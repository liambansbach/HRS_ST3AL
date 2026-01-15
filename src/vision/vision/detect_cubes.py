#!/usr/bin/env python3

"""
Docstring for vision.vision.detect_cubes

histogram → global blob → initial window → CamShift tracking → optional optical-flow fine-tune
"""

from pathlib import Path

import numpy as np
import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from rclpy.callback_groups import ReentrantCallbackGroup

class CubeDetector(Node):
    def __init__(self):
        super().__init__('cube_detector')
        self.cwd = Path.cwd()

        self.cb_group = ReentrantCallbackGroup()

        self.bridge = CvBridge()
        self.frame = None

        self.red_hist_path = "/home/maalonjochmann/HRS_ST3AL/src/vision/vision/maalon/hist_red.npy"
        self.green_hist_path = "/home/maalonjochmann/HRS_ST3AL/src/vision/vision/maalon/hist_green.npy"
        self.blue_hist_path = "/home/maalonjochmann/HRS_ST3AL/src/vision/vision/maalon/hist_blue.npy"

        self.red_hist = np.load(self.red_hist_path)
        self.green_hist = np.load(self.green_hist_path)
        self.blue_hist = np.load(self.blue_hist_path)

        self.green_window = None
        self.red_window = None
        self.blue_window = None

        self.trackers = {
            "green": {"hist": np.load(self.green_hist_path), "window": None,
                    "prev_area": None, "flow_pts": None, "flow_prev_gray": None},
            "red":   {"hist": np.load(self.red_hist_path),   "window": None,
                    "prev_area": None, "flow_pts": None, "flow_prev_gray": None},
            "blue":  {"hist": np.load(self.blue_hist_path),  "window": None,
                    "prev_area": None, "flow_pts": None, "flow_prev_gray": None},
        }

        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        self.flow_pts = None
        self.flow_prev_gray = None
        self.flow_min_pts = 25

        self.feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=5, blockSize=7)

        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        # QoS: Reliable to ensure camera_info is received
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub_distorted = self.create_subscription(
            Image,
            'camera_image/undistorted',
            self.camera_cb,
            sensor_qos,
            callback_group=self.cb_group,
        )

    def camera_cb(self, msg: Image):
        # Convert image to OpenCV BGR
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        except Exception as e:
            self.get_logger().warn(f"Failed to convert image: {e}")
            return
        
        self.frame = frame


    """
    ===========================
            CAMSHIFT 
    ===========================
    """

    def back_projection(self, hist):
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        mask_sv = cv2.inRange(hsv, (0, 40, 40), (179, 255, 255))
        bp = cv2.calcBackProject([h], [0], hist, [0, 180], 1)
        bp = cv2.GaussianBlur(bp, (7, 7), 0)        # blur reduces jitter
        bp = cv2.bitwise_and(bp, bp, mask=mask_sv)
        _, bp = cv2.threshold(bp, 30, 255, cv2.THRESH_TOZERO)  # tune 20–60
        return bp
    
    def find_biggest_blob(self, bp, min_area=300):        # maybe add min_area?
        """
        Finds biggest blob
        
        :param bp: backprojection depending on color histogram
        returns: bounding rect of biggest blob
        """

        bw = (bp > 0).astype(np.uint8) * 255 

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)

        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < min_area:
            return None
        return cv2.boundingRect(c)



    def apply_camshift(self):
        """
        applies camshift to detect the cubes and - if possible - improves tracking with optical flow
        
        :param self: Description
        """
        vis = self.frame.copy()
        bp = self.back_projection(hist=self.green_hist)
        
        #self.green_window = self.find_biggest_blob(bp)
        if self.green_window is None:
            self.green_window = self.find_biggest_blob(bp)
            if self.green_window is None:
                self.get_logger().warn("[green] no blob found to init window")
                return vis        

        rotated_rect, new_window = cv2.CamShift(bp, self.green_window, self.term_crit)

        self.green_window = new_window
        x, y, w, h = self.green_window

        prev_area = getattr(self, "_prev_green_area", None)
        new_area = w * h
        if prev_area is not None and new_area > prev_area * 1.5:  # tune 1.5–3.0
            self.get_logger().warn("[green] window exploded -> reset")
            self.green_window = None
            self._prev_green_area = None
            return vis
        self._prev_green_area = new_area

        #optional
        H, W = bp.shape[:2]
        x = max(0, min(x, W-1))
        y = max(0, min(y, H-1))
        w = max(1, min(w, W-x))
        h = max(1, min(h, H-y))
        self.green_window = (x, y, w, h)

        roi = bp[y:y+h, x:x+w]
        conf = float(np.mean(roi)) if roi.size else 0.0
        
        ###
        # remember size
        prev_w = getattr(self, "_prev_w", w)
        prev_h = getattr(self, "_prev_h", h)

        # CamShift-rotated size ist oft stabiler als axis-aligned w/h
        (ccx, ccy), (rw, rh), _ = rotated_rect
        scale = 1.1  # kleiner Puffer um Würfel
        w_new = max(1.0, rw * scale)
        h_new = max(1.0, rh * scale)

        # low confidence -> keep size
        conf_freeze = 11.0  # tune: 10–20 (bei occlusion meist niedriger)
        if conf < conf_freeze:
            w_new, h_new = float(prev_w), float(prev_h)
        else:
            # limit increase in size per frame
            # clamp shrink & grow pro frame
            max_grow = 1.30     # tune
            max_shrink = 0.85   # tune (0.7–0.9)
            w_new = min(w_new, prev_w * max_grow)
            h_new = min(h_new, prev_h * max_grow)
            w_new = max(w_new, prev_w * max_shrink)
            h_new = max(h_new, prev_h * max_shrink)

        # smoothing gegen jitter (low pass filter)
        alpha = 0.2
        w_s = getattr(self, "_w_smooth", w_new)
        h_s = getattr(self, "_h_smooth", h_new)
        w_s = (1 - alpha) * w_s + alpha * w_new
        h_s = (1 - alpha) * h_s + alpha * h_new
        self._w_smooth, self._h_smooth = w_s, h_s

        # axis-aligned window neu um CamShift-center bauen
        w2, h2 = int(w_s), int(h_s)
        x2 = int(ccx - w2 / 2)
        y2 = int(ccy - h2 / 2)

        # clamp to image
        x2 = max(0, min(x2, W - 1))
        y2 = max(0, min(y2, H - 1))
        w2 = max(1, min(w2, W - x2))
        h2 = max(1, min(h2, H - y2))

        self.green_window = (x2, y2, w2, h2)
        self._prev_w, self._prev_h = w2, h2
        
        if conf < 8.0:  # tune
            self.get_logger().warn(f"[green] LOST conf={conf:.1f} -> reset window")
            self.green_window = None
            return vis

        # Check if optical flow works
        if self.flow_pts is None or self.flow_prev_gray is None:
            ok = self.init_flow_in_window(self.green_window, bp=bp)
            if not ok:
                # flow not available, CamShift continues alone
                pass
        else:
            ok = self.flow_step_update_window()
            if ok:
                # convert tracked points -> updated window
                pts2 = self.flow_pts.reshape(-1, 2).astype(np.float32)

                # check if flow box viel größer als cam shift
                xC, yC, wC, hC = self.green_window
                areaC = wC * hC

                xF, yF, wF, hF = cv2.boundingRect(pts2)
                areaF = wF * hF

                if areaC > 0 and areaF > areaC * 1.6:  # tune 1.3–2.5
                    self.get_logger().warn("[flow] exploded vs camshift -> reset flow only")
                    self.flow_pts = None
                    self.flow_prev_gray = None
                else:
                    # größe bleibt, flow bestimmt nur cx cy neu
                    mx = float(np.mean(pts2[:, 0]))
                    my = float(np.mean(pts2[:, 1]))

                    x2 = int(mx - wC / 2.0)
                    y2 = int(my - hC / 2.0)

                    # clamp to image
                    x2 = max(0, min(x2, W - 1))
                    y2 = max(0, min(y2, H - 1))
                    w2 = max(1, min(int(wC), W - x2))
                    h2 = max(1, min(int(hC), H - y2))

                    self.green_window = (x2, y2, w2, h2)

        pts = cv2.boxPoints(rotated_rect).astype(np.int32)
        cv2.polylines(vis, [pts], True, (0, 120, 0), 2)

        x, y, w, h = self.green_window
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 80, 0), 2)

        return vis
    
    """
    ===================
        OPTICAL FLOW
    ===================
    """

    def init_flow_in_window(self, window, bp=None):
        if window is None or self.frame is None:
            return False

        x, y, w, h = window
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        # only features in window
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[y:y+h, x:x+w] = 255 

        # only features in color hist
        if bp is not None:
            mask_bp = (bp > 50).astype(np.uint8) * 255
            mask = cv2.bitwise_and(mask, mask_bp)

        pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
        if pts is None or len(pts) < self.flow_min_pts:
            return False

        self.flow_pts = pts.astype(np.float32)
        self.flow_prev_gray = gray
        return True
    
    def flow_step_update_window(self):
        if self.flow_pts is None or self.flow_prev_gray is None or self.frame is None:
            return False
        
        curr_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(
        self.flow_prev_gray, curr_gray, self.flow_pts, None, **self.lk_params
        )

        # works?
        if p1 is None or st is None:
            self.flow_pts = None
            self.flow_prev_gray = None
            return False

        # only keep successfully tracked points
        good_new = p1[st == 1]
        if good_new.shape[0] < self.flow_min_pts:
            self.flow_pts = None
            self.flow_prev_gray = None
            return False

        # update tracker state
        self.flow_pts = good_new.reshape(-1, 1, 2).astype(np.float32)
        self.flow_prev_gray = curr_gray
        return True


    def display_loop(self):
        """Main loop to display detected cubes."""
        while rclpy.ok():
            if self.frame is not None:
                vis = self.frame.copy()
                """for color in ("red", "green", "blue"):
                    vis = self.track_one(color, vis)"""
                vis = self.apply_camshift()
                cv2.imshow("camshift", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
                
            rclpy.spin_once(self, timeout_sec=0.01)

        cv2.destroyAllWindows()

def main():
    rclpy.init()
    node = CubeDetector()

    try:
        #rclpy.spin(node)
        node.display_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
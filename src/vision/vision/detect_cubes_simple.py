#!/usr/bin/env python3
"""
ROS2 CubeDetector Node (Histogram Backprojection + Blob + (optional) Optical Flow)
Axis-aligned, square bounding boxes with EMA smoothing (no rotated bboxes).

Pipeline per color:
1) Histogram backprojection (Hue) -> bp map
2) Find biggest blob (prefer ROI around last window for stability)
3) Convert to square bbox + padding
4) EMA smoothing on center + size
5) Optional optical flow refinement: update center only, keep size
6) Publish CubeBBoxList (angle=0)

[TUNE] params are marked below.
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
from ainex_interfaces.msg import CubeBBox, CubeBBoxList


class CubeDetector(Node):
    def __init__(self):
        super().__init__('cube_detector')
        self.cwd = Path.cwd()

        self.cb_group = ReentrantCallbackGroup()
        self.bridge = CvBridge()
        self.frame = None

        # Histogram paths
        self.red_hist_path = Path.joinpath(self.cwd, "src/vision/vision/maalon/hist_red.npy")
        self.green_hist_path = Path.joinpath(self.cwd, "src/vision/vision/maalon/hist_green.npy")
        self.blue_hist_path = Path.joinpath(self.cwd, "src/vision/vision/maalon/hist_blue.npy")

        # --- [TUNE] global parameters ---
        self.min_blob_area = 250
        self.conf_thresh = 8.0           # min mean bp in bbox to accept
        self.square_scale = 0.93          # padding scale for square bbox
        self.ema_alpha = 0.2             # smoothing strength (higher = more responsive)
        self.roi_search_scale = 1.63       # search region size around last_window (multiplier)
        self.keep_last_on_fail = True     # publish last_window even if lost this frame

        self.bp_tight_thr = 30              # [TUNE] 60..120 tighter bbox threshold
        self.jump_center_px = 100           # [TUNE] max center jump allowed before reset
        self.jump_area_ratio = 3.5          # [TUNE] if area changes too much -> reset

        # Optical flow parameters
        self.flow_min_pts = 25
        self.feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=5, blockSize=7)
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
    
        self.use_optical_flow = False


        self.DRAW = {
            "red":   {"box": (0, 0, 255)},
            "green": {"box": (0, 120, 0)},
            "blue":  {"box": (255, 0, 0)},
        }

        self.trackers = {
            "red":   {"hist": np.load(self.red_hist_path)},
            "green": {"hist": np.load(self.green_hist_path)},
            "blue":  {"hist": np.load(self.blue_hist_path)},
        }
        # init per-color state
        for c in self.trackers:
            self._init_tracker_state(self.trackers[c])

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

        self.pub_boxes = self.create_publisher(CubeBBoxList, "cubes_position", 10)

    def _init_tracker_state(self, t: dict):
        t.update({
            "window": None,
            "last_window": None,
            "tracked": False,
            # EMA state
            "ema_cx": None,
            "ema_cy": None,
            "ema_s": None,
            # optical flow
            "flow_pts": None,
            "flow_prev_gray": None,
        })

    def camera_cb(self, msg: Image):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"Failed to convert image: {e}")
            return

    def publish_boxes(self):
        msg = CubeBBoxList()
        msg.cubes = []

        for color, t in self.trackers.items():
            win = t["window"] if t["tracked"] else t["last_window"]
            if win is None:
                continue

            x, y, w, h = win
            cx = x + w / 2.0
            cy = y + h / 2.0

            b = CubeBBox()
            b.id = color
            b.cx = float(cx)
            b.cy = float(cy)
            b.w = float(w)
            b.h = float(h)
            b.angle = 0.0
            msg.cubes.append(b)

        self.pub_boxes.publish(msg)

    # ----------------------------
    # Backprojection + Blob
    # ----------------------------
    def back_projection(self, hist, hsv):
        h, s, v = cv2.split(hsv)

        # [TUNE] mask to suppress gray/dark pixels
        mask_sv = cv2.inRange(hsv, (0, 30, 60), (179, 255, 255))

        bp = cv2.calcBackProject([h], [0], hist, [0, 180], 1)
        bp = cv2.GaussianBlur(bp, (5, 5), 0)  # [TUNE]
        bp = cv2.bitwise_and(bp, bp, mask=mask_sv)

        # [TUNE] remove low responses
        _, bp = cv2.threshold(bp, 50, 255, cv2.THRESH_TOZERO)
        return bp

    def find_biggest_blob(self, bp, min_area=300):
        bw = (bp > 0).astype(np.uint8) * 255

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # [TUNE]
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)

        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < min_area:
            return None
        return cv2.boundingRect(c)
    
    def tight_bbox_from_bp(self, bp, search_win=None, thr=80, min_area=300):
        """
        Compute a tighter bbox by thresholding bp harder and taking largest contour bbox.
        If search_win is given, restrict to that region and return bbox in full-image coords.
        """
        H, W = bp.shape[:2]

        if search_win is None:
            x0, y0, w0, h0 = 0, 0, W, H
        else:
            x0, y0, w0, h0 = search_win

        roi = bp[y0:y0+h0, x0:x0+w0]
        if roi.size == 0:
            return None

        # hard threshold -> tighter mask
        _, bw = cv2.threshold(roi, thr, 255, cv2.THRESH_BINARY)  # [TUNE] thr

        # small morphology to remove speckles but NOT inflate too much
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # [TUNE]
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)

        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < min_area:
            return None

        x, y, w, h = cv2.boundingRect(c)
        return (x + x0, y + y0, w, h)


    def conf_in_window(self, bp, win):
        if win is None:
            return 0.0
        x, y, w, h = win
        roi = bp[y:y+h, x:x+w]
        return float(np.mean(roi)) if roi.size else 0.0

    # ----------------------------
    # Square + EMA
    # ----------------------------
    def square_box(self, x, y, w, h, W, H, scale=1.15):
        cx = x + w / 2.0
        cy = y + h / 2.0
        s = int(max(1, max(w, h) * scale))

        x2 = int(cx - s / 2)
        y2 = int(cy - s / 2)

        x2 = max(0, min(x2, W - 1))
        y2 = max(0, min(y2, H - 1))
        s = max(1, min(s, W - x2, H - y2))
        return (x2, y2, s, s)

    def ema_window(self, t, win, alpha=0.25):
        x, y, w, h = win
        cx, cy = x + w / 2.0, y + h / 2.0
        s = float(w)  # square => w==h

        if t["ema_cx"] is None:
            t["ema_cx"], t["ema_cy"], t["ema_s"] = cx, cy, s
        else:
            t["ema_cx"] = (1 - alpha) * t["ema_cx"] + alpha * cx
            t["ema_cy"] = (1 - alpha) * t["ema_cy"] + alpha * cy
            t["ema_s"] = (1 - alpha) * t["ema_s"] + alpha * s

        return t["ema_cx"], t["ema_cy"], t["ema_s"]

    def clamp_square_from_center(self, cx, cy, s, W, H):
        s2 = int(max(1, s))
        x2 = int(cx - s2 / 2)
        y2 = int(cy - s2 / 2)

        x2 = max(0, min(x2, W - 1))
        y2 = max(0, min(y2, H - 1))
        s2 = max(1, min(s2, W - x2, H - y2))
        return (x2, y2, s2, s2)

    # ----------------------------
    # Optical Flow (center only)
    # ----------------------------
    def init_flow_in_window(self, t, window, bp=None):
        if window is None or self.frame is None:
            return False

        x, y, w, h = window
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[y:y+h, x:x+w] = 255

        if bp is not None:
            mask_bp = (bp > 50).astype(np.uint8) * 255  # [TUNE]
            mask = cv2.bitwise_and(mask, mask_bp)

        pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
        if pts is None or len(pts) < self.flow_min_pts:
            return False

        t["flow_pts"] = pts.astype(np.float32)
        t["flow_prev_gray"] = gray
        return True

    def flow_step_update(self, t):
        if t["flow_pts"] is None or t["flow_prev_gray"] is None or self.frame is None:
            return False

        curr_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            t["flow_prev_gray"], curr_gray, t["flow_pts"], None, **self.lk_params
        )

        if p1 is None or st is None:
            t["flow_pts"] = None
            t["flow_prev_gray"] = None
            return False

        good_new = p1[st == 1]
        if good_new.shape[0] < self.flow_min_pts:
            t["flow_pts"] = None
            t["flow_prev_gray"] = None
            return False

        t["flow_pts"] = good_new.reshape(-1, 1, 2).astype(np.float32)
        t["flow_prev_gray"] = curr_gray
        return True

    def reset_tracker(self, t, keep_last=True):
        t["window"] = None
        t["tracked"] = False
        # EMA
        t["ema_cx"] = None
        t["ema_cy"] = None
        t["ema_s"] = None
        # flow
        t["flow_pts"] = None
        t["flow_prev_gray"] = None
        if not keep_last:
            t["last_window"] = None


    # ----------------------------
    # Tracking per color
    # ----------------------------
    def roi_around_last(self, last_win, W, H, scale=2.0):
        x, y, w, h = last_win
        cx, cy = x + w/2.0, y + h/2.0
        s = max(w, h) * scale
        s = int(max(1, s))
        x2 = int(cx - s/2)
        y2 = int(cy - s/2)
        x2 = max(0, min(x2, W-1))
        y2 = max(0, min(y2, H-1))
        s = max(1, min(s, W-x2, H-y2))
        return (x2, y2, s, s)

    def track_color(self, color, vis, hsv):
        t = self.trackers[color]
        t["tracked"] = False

        bp = self.back_projection(hist=t["hist"], hsv=hsv)
        H, W = bp.shape[:2]

        # 1) define a search region: ROI around last only if it exists
        roi_win = None
        if t["last_window"] is not None:
            roi_win = self.roi_around_last(t["last_window"], W, H, scale=self.roi_search_scale)

        # 2) tight bbox from bp (prefer ROI, fallback full frame)
        win = None
        if roi_win is not None:
            win = self.tight_bbox_from_bp(
                bp, search_win=roi_win, thr=self.bp_tight_thr, min_area=self.min_blob_area
            )

        if win is None:
            win = self.tight_bbox_from_bp(
                bp, search_win=None, thr=self.bp_tight_thr, min_area=self.min_blob_area
            )

        if win is None:
            # lost this frame
            if not self.keep_last_on_fail:
                self.reset_tracker(t, keep_last=False)
            return vis

        # 3) confidence check (mean in bbox)
        conf = self.conf_in_window(bp, win)
        if conf < self.conf_thresh:
            if not self.keep_last_on_fail:
                self.reset_tracker(t, keep_last=False)
            return vis

        # 4) jump detection vs last_window -> reset EMA+Flow and use this detection immediately
        if t["last_window"] is not None:
            lx, ly, lw, lh = t["last_window"]
            lc = (lx + lw/2.0, ly + lh/2.0)
            x, y, w, h = win
            c = (x + w/2.0, y + h/2.0)

            dist = ((c[0]-lc[0])**2 + (c[1]-lc[1])**2) ** 0.5
            area = w*h
            larea = max(1, lw*lh)
            area_ratio = max(area/larea, larea/area)

            if dist > self.jump_center_px or area_ratio > self.jump_area_ratio:
                # big discontinuity (bag jump) -> reset smoothing/flow so it snaps immediately
                self.reset_tracker(t, keep_last=True)
                # keep last_window though; we'll overwrite below with the new one

        x, y, w, h = win

        # 5) square bbox with MUCH smaller padding
        win = self.square_box(x, y, w, h, W, H, scale=self.square_scale)

        # 6) EMA smoothing (but after a reset it will re-init and snap)
        cx, cy, s_ema = self.ema_window(t, win, alpha=self.ema_alpha)
        win = self.clamp_square_from_center(cx, cy, s_ema, W, H)


        # 7) optical flow: only if already initialized and we did not just reset
        if self.use_optical_flow:
            if t["flow_pts"] is None or t["flow_prev_gray"] is None:
                self.init_flow_in_window(t, win, bp=bp)
            else:
                ok = self.flow_step_update(t)
                if ok:
                    pts2 = t["flow_pts"].reshape(-1, 2)
                    mx, my = float(np.mean(pts2[:, 0])), float(np.mean(pts2[:, 1]))
                    _, _, s, _ = win
                    win = self.clamp_square_from_center(mx, my, s, W, H)

        # draw
        cv2.rectangle(
            vis, (win[0], win[1]), (win[0] + win[2], win[1] + win[3]),
            self.DRAW[color]["box"], 2
        )

        # save
        t["window"] = win
        t["last_window"] = win
        t["tracked"] = True
        return vis


    def display_loop(self):
        while rclpy.ok():
            if self.frame is not None:
                hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
                vis = self.frame.copy()

                for color in ("red", "green", "blue"):
                    vis = self.track_color(color, vis, hsv)

                self.publish_boxes()
                cv2.imshow("cube_detector", vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

            rclpy.spin_once(self, timeout_sec=0.01)

        cv2.destroyAllWindows()


def main():
    rclpy.init()
    node = CubeDetector()
    try:
        node.display_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

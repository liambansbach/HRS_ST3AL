#!/usr/bin/env python3
"""
ROS2 CubeDetector Node (Histogram Backprojection + Blob + (optional) Optical Flow)
Axis-aligned, square bounding boxes with EMA smoothing (no rotated bboxes).

NOW: subscribes to DISTORTED JPEG stream:
  - sensor_msgs/CompressedImage on `camera_image/compressed`
"""

from pathlib import Path
from collections import deque

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import CompressedImage
from ainex_interfaces.msg import CubeBBox, CubeBBoxList


class CubeDetector(Node):
    def __init__(self):
        super().__init__('cube_detector')
        self.cwd = Path.cwd()

        self.cb_group = ReentrantCallbackGroup()
        self.frame = None

        # Histogram paths
        self.red_hist_path = Path.joinpath(self.cwd, "src/vision/histograms/hist_red_front-face_new2.npy")
        self.green_hist_path = Path.joinpath(self.cwd, "src/vision/histograms/hist_green_front-face_new2.npy")
        self.blue_hist_path = Path.joinpath(self.cwd, "src/vision/histograms/hist_blue_front-face_new2.npy")

        # --- [TUNE] global parameters ---
        self.min_blob_area = 1250 #750
        self.conf_thresh = 10.0  #10
        self.square_scale = 0.9  #0.9
        self.ema_alpha = 0.3     #0.3
        self.roi_search_scale = 1.5 #1.5
        self.keep_last_on_fail = True

        self.bp_tight_thr = 5 # 0.5
        self.jump_center_px = 100 # 100
        self.jump_area_ratio = 2.0 # 2.0

        # Optical flow parameters
        self.flow_min_pts = 25
        self.feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=5, blockSize=7)
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        self.use_optical_flow = False

        # temporal smoothing
        self.hist_len = 10
        self.use_history_median = True

        # per-color params
        self.color_params = {
            "red": {
                "sv_min": (0, 50, 30), # 0, 70, 50
                "blur": (3, 3),
                "bp_tozero": 20,
                "tight_thr": 70,
            },
            "green": {
                "sv_min": (0, 25, 25), # 0, 35, 35
                "blur": (5, 5),
                "bp_tozero": 30,
                "tight_thr": 80,
            },
            "blue": {
                "sv_min": (0, 255*0.12, 255*0.60), # 0, 35, 35
                "blur": (5, 5),
                "bp_tozero": 30,
                "tight_thr": 60,
            },
        }

        self.DRAW = {
            "red":   {"box": (0, 0, 255)},
            "green": {"box": (0, 120, 0)},
            "blue":  {"box": (255, 0, 0)},
        }

        self.trackers = {
            "red":   {"hue_hist": np.load(self.red_hist_path)},
            "green": {"hue_hist": np.load(self.green_hist_path)},
            "blue":  {"hue_hist": np.load(self.blue_hist_path)},
        }
        for c in self.trackers:
            self._init_tracker_state(self.trackers[c])

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ✅ Subscribe to distorted JPEG stream
        self.sub_distorted = self.create_subscription(
            CompressedImage,
            "camera_image/compressed",
            self.camera_cb_compressed,
            sensor_qos,
            callback_group=self.cb_group,
        )

        self.pub_boxes = self.create_publisher(CubeBBoxList, "cubes_position", 10)

    # ----------------------------
    # State init
    # ----------------------------
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
            # history for median smoothing
            "win_hist": deque(maxlen=self.hist_len),
        })

    # ----------------------------
    # Image callback (CompressedImage)
    # ----------------------------
    def camera_cb_compressed(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                self.get_logger().warn("cv2.imdecode returned None")
                return
            self.frame = frame
        except Exception as e:
            self.get_logger().warn(f"Failed to decode compressed image: {e}")

    # ----------------------------
    # Publish bboxes
    # ----------------------------
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
    def back_projection(self, color, hist, hsv):
        h, s, v = cv2.split(hsv)

        p = self.color_params[color]
        sv_min = p["sv_min"]
        blur_k = p["blur"]
        tozero = p["bp_tozero"]

        mask_sv = cv2.inRange(hsv, sv_min, (356, 255, 255)) #179

        bp = cv2.calcBackProject([h], [0], hist, [0, 180], 1)
        bp = cv2.GaussianBlur(bp, blur_k, 0)
        bp = cv2.bitwise_and(bp, bp, mask=mask_sv)

        _, bp = cv2.threshold(bp, tozero, 255, cv2.THRESH_TOZERO)
        return bp

    def tight_bbox_from_bp(self, bp, search_win=None, thr=80, min_area=300):
        H, W = bp.shape[:2]

        if search_win is None:
            x0, y0, w0, h0 = 0, 0, W, H
        else:
            x0, y0, w0, h0 = search_win

        roi = bp[y0:y0+h0, x0:x0+w0]
        if roi.size == 0:
            return None

        _, bw = cv2.threshold(roi, thr, 255, cv2.THRESH_BINARY)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
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
    # Square + EMA + history
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
        s = float(w)

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

    def smooth_with_history(self, t, win, W, H):
        x, y, w, h = win
        cx = x + w / 2.0
        cy = y + h / 2.0
        s = float(w)

        t["win_hist"].append((cx, cy, s))
        arr = np.array(t["win_hist"], dtype=np.float32)

        cx_m = float(np.median(arr[:, 0]))
        cy_m = float(np.median(arr[:, 1]))
        s_m  = float(np.median(arr[:, 2]))

        return self.clamp_square_from_center(cx_m, cy_m, s_m, W, H)

    # ----------------------------
    # Tracking per color
    # ----------------------------
    def roi_around_last(self, last_win, W, H, scale=2.0):
        x, y, w, h = last_win
        cx, cy = x + w/2.0, y + h/2.0
        s = int(max(1, max(w, h) * scale))
        x2 = int(cx - s/2)
        y2 = int(cy - s/2)
        x2 = max(0, min(x2, W-1))
        y2 = max(0, min(y2, H-1))
        s = max(1, min(s, W-x2, H-y2))
        return (x2, y2, s, s)

    def reset_tracker(self, t, keep_last=True):
        t["window"] = None
        t["tracked"] = False
        t["ema_cx"] = None
        t["ema_cy"] = None
        t["ema_s"] = None
        t["flow_pts"] = None
        t["flow_prev_gray"] = None
        t["win_hist"].clear()
        if not keep_last:
            t["last_window"] = None

    def track_color(self, color, vis, hsv):
        t = self.trackers[color]
        t["tracked"] = False

        bp = self.back_projection(color=color, hist=t["hue_hist"], hsv=hsv)
        H, W = bp.shape[:2]
        thr = self.color_params[color]["tight_thr"]

        roi_win = None
        if t["last_window"] is not None:
            roi_win = self.roi_around_last(t["last_window"], W, H, scale=self.roi_search_scale)

        win = None
        if roi_win is not None:
            win = self.tight_bbox_from_bp(bp, search_win=roi_win, thr=thr, min_area=self.min_blob_area)
        if win is None:
            win = self.tight_bbox_from_bp(bp, search_win=None, thr=thr, min_area=self.min_blob_area)

        if win is None:
            if not self.keep_last_on_fail:
                self.reset_tracker(t, keep_last=False)
            return vis

        conf = self.conf_in_window(bp, win)
        if conf < self.conf_thresh:
            if not self.keep_last_on_fail:
                self.reset_tracker(t, keep_last=False)
            return vis

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
                self.reset_tracker(t, keep_last=True)

        x, y, w, h = win

        win = self.square_box(x, y, w, h, W, H, scale=self.square_scale)

        if self.use_history_median:
            win = self.smooth_with_history(t, win, W, H)

        cx, cy, s_ema = self.ema_window(t, win, alpha=self.ema_alpha)
        win = self.clamp_square_from_center(cx, cy, s_ema, W, H)

        cv2.rectangle(
            vis, (win[0], win[1]), (win[0] + win[2], win[1] + win[3]),
            self.DRAW[color]["box"], 2
        )

        t["window"] = win
        t["last_window"] = win
        t["tracked"] = True
        return vis

    # ----------------------------
    # Display loop
    # ----------------------------
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

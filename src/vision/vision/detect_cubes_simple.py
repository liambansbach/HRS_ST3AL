#!/usr/bin/env python3
"""
CubeDetector Node (Histogram Backprojection + Blob)
Axis-aligned, square bounding boxes with EMA + optional history median smoothing
(no rotated bboxes).

Input:
  - sensor_msgs/CompressedImage on `camera_image/compressed` (distorted stream)

Output:
  - ainex_interfaces/CubeBBoxList on `cubes_position`
    Each bbox is represented by its center (cx, cy) and size (w, h) in pixel space.

Main pipeline (per color):
  1) Convert BGR -> HSV
  2) Build SV mask to suppress dark / low-saturation regions
  3) Histogram backprojection on Hue channel
  4) Threshold + morphology -> blob extraction -> tight bbox
  5) Convert to square bbox + temporal smoothing (median history + EMA)
  6) Publish bounding boxes
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
    """
    Detects and tracks colored cubes (red/green/blue) using per-color Hue histograms.

    The tracker keeps a 'last_window' per color and searches in a ROI around it to
    stabilize detections. When a detection is found, it is made square and smoothed
    over time (median history + EMA).
    """

    def __init__(self):
        """Initialize parameters, load histograms, set up ROS interfaces."""
        super().__init__('cube_detector')
        self.cwd = Path.cwd()

        # Callback group for subscription + event loop
        self.cb_group = ReentrantCallbackGroup()

        # Latest received frame (BGR)
        self.frame = None

        # Histogram paths (Hue histograms stored as .npy)
        self.red_hist_path = Path.joinpath(self.cwd, "src/vision/histograms/hist_red_front-face_new3.npy")
        self.green_hist_path = Path.joinpath(self.cwd, "src/vision/histograms/hist_green_front-face_new2.npy")
        self.blue_hist_path = Path.joinpath(self.cwd, "src/vision/histograms/hist_blue_front-face_new3.npy")

        # --- [TUNE] global parameters ---
        self.min_blob_area = 1250
        self.conf_thresh = 10
        self.square_scale = 1.0
        self.ema_alpha = 0.3
        self.roi_search_scale = 1.5
        self.keep_last_on_fail = True

        # Reject sudden jumps between frames (stability heuristic)
        self.jump_center_px = 100
        self.jump_area_ratio = 2.0

        # Temporal smoothing (history median)
        self.hist_len = 10
        self.use_history_median = True

        # Per-color params (HSV thresholding + backprojection thresholding)
        self.color_params = {
            "red": {
                "sv_min": (0, 255 * 0.18, 255 * 0.33),
                "blur": (3, 3),
                "bp_tozero": 20,
                "tight_thr": 70,
            },
            "green": {
                "sv_min": (0, 255 * 0.38, 255 * 0.25),
                "blur": (3, 3),
                "bp_tozero": 30,
                "tight_thr": 80,
            },
            "blue": {
                "sv_min": (0, 255 * 0.55, 255 * 0.08),
                "blur": (3, 3),
                "bp_tozero": 20,
                "tight_thr": 70,
            },
        }

        # Visualization colors (BGR)
        self.DRAW = {
            "red": {"box": (0, 0, 255)},
            "green": {"box": (0, 120, 0)},
            "blue": {"box": (255, 0, 0)},
        }

        # Load per-color histograms and initialize tracker state
        self.trackers = {
            "red": {"hue_hist": np.load(self.red_hist_path)},
            "green": {"hue_hist": np.load(self.green_hist_path)},
            "blue": {"hue_hist": np.load(self.blue_hist_path)},
        }
        for c in self.trackers:
            self._init_tracker_state(self.trackers[c])

        # Best-effort sensor QoS for camera streams
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribe to distorted stream
        self.sub_distorted = self.create_subscription(
            CompressedImage,
            "camera_image/compressed",
            self.camera_cb_compressed,
            sensor_qos,
            callback_group=self.cb_group,
        )

        # Publish list of detected cube bounding boxes
        self.pub_boxes = self.create_publisher(CubeBBoxList, "cubes_position", 10)

    # ----------------------------
    # Tracker state init/reset
    # ----------------------------
    def _init_tracker_state(self, t: dict) -> None:
        """
        Initialize per-color tracking state.

        Args:
            t: Tracker dict to be updated in-place.

        Returns:
            None
        """
        t.update(
            {
                "window": None,       # current window (x, y, w, h)
                "last_window": None,  # last valid window
                "tracked": False,     # updated this frame?
                # EMA state (center + size)
                "ema_cx": None,
                "ema_cy": None,
                "ema_s": None,
                # History for median smoothing (store cx, cy, s)
                "win_hist": deque(maxlen=self.hist_len),
            }
        )

    def reset_tracker(self, t: dict, keep_last: bool = True) -> None:
        """
        Reset a tracker's internal state.

        Args:
            t: Tracker dict (modified in-place).
            keep_last: If False, also clears last_window.

        Returns:
            None
        """
        t["window"] = None
        t["tracked"] = False
        t["ema_cx"] = None
        t["ema_cy"] = None
        t["ema_s"] = None
        t["win_hist"].clear()
        if not keep_last:
            t["last_window"] = None

    # ----------------------------
    # Image callback (CompressedImage)
    # ----------------------------
    def camera_cb_compressed(self, msg: CompressedImage) -> None:
        """
        Decode a JPEG CompressedImage into an OpenCV BGR frame.

        Args:
            msg: ROS2 CompressedImage message containing JPEG bytes.

        Returns:
            None (updates self.frame)
        """
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
    def publish_boxes(self) -> None:
        """
        Publish current (or last known) windows for each color as CubeBBoxList.

        Uses last_window when a color was not tracked in the current frame.

        Returns:
            None
        """
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
            b.angle = 0.0  # axis-aligned (no rotation estimate)
            msg.cubes.append(b)

        self.pub_boxes.publish(msg)

    # ----------------------------
    # Backprojection + Blob
    # ----------------------------
    def back_projection(self, color: str, hist: np.ndarray, hsv: np.ndarray) -> np.ndarray:
        """
        Compute backprojection image for the given color using Hue histogram.

        Steps:
          1) Build SV mask (remove low saturation/value pixels)
          2) Backproject Hue channel
          3) Blur + apply mask
          4) THRESH_TOZERO to suppress weak responses

        Args:
            color: "red" | "green" | "blue"
            hist: Hue histogram for the color
            hsv: HSV image (uint8)

        Returns:
            Backprojection response image (uint8).
        """
        h, _, _ = cv2.split(hsv)

        p = self.color_params[color]
        sv_min = p["sv_min"]
        blur_k = p["blur"]
        tozero = p["bp_tozero"]

        # Note: upper Hue bound is ignored by inRange; kept as-is from original code.
        mask_sv = cv2.inRange(hsv, sv_min, (356, 255, 255))

        bp = cv2.calcBackProject([h], [0], hist, [0, 180], 1)
        bp = cv2.GaussianBlur(bp, blur_k, 0)
        bp = cv2.bitwise_and(bp, bp, mask=mask_sv)

        _, bp = cv2.threshold(bp, tozero, 255, cv2.THRESH_TOZERO)
        return bp

    def tight_bbox_from_bp(
        self,
        bp: np.ndarray,
        search_win: tuple | None = None,
        thr: int = 80,
        min_area: int = 300,
    ) -> tuple | None:
        """
        Extract the largest blob bbox from a backprojection image.

        Args:
            bp: Backprojection image (uint8).
            search_win: Optional ROI window (x, y, w, h). If None, uses full image.
            thr: Binary threshold on bp.
            min_area: Minimum contour area to accept.

        Returns:
            (x, y, w, h) bbox in full-image coordinates, or None if no valid blob.
        """
        H, W = bp.shape[:2]

        if search_win is None:
            x0, y0, w0, h0 = 0, 0, W, H
        else:
            x0, y0, w0, h0 = search_win

        roi = bp[y0:y0 + h0, x0:x0 + w0]
        if roi.size == 0:
            return None

        # Threshold + morphology to get clean blobs
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

    def conf_in_window(self, bp: np.ndarray, win: tuple | None) -> float:
        """
        Compute mean backprojection confidence inside a window.

        Args:
            bp: Backprojection image (uint8).
            win: Window (x, y, w, h) or None.

        Returns:
            Mean ROI intensity as float (0 if invalid).
        """
        if win is None:
            return 0.0
        x, y, w, h = win
        roi = bp[y:y + h, x:x + w]
        return float(np.mean(roi)) if roi.size else 0.0

    # ----------------------------
    # Square + EMA + history
    # ----------------------------
    def square_box(self, x: int, y: int, w: int, h: int, W: int, H: int, scale: float = 1.15) -> tuple:
        """
        Convert an axis-aligned bbox to a square bbox (clamped to image bounds).

        Args:
            x, y, w, h: Input bbox.
            W, H: Image width/height.
            scale: Scale factor applied to max(w, h).

        Returns:
            Square window (x, y, s, s).
        """
        cx = x + w / 2.0
        cy = y + h / 2.0
        s = int(max(1, max(w, h) * scale))

        x2 = int(cx - s / 2)
        y2 = int(cy - s / 2)

        x2 = max(0, min(x2, W - 1))
        y2 = max(0, min(y2, H - 1))
        s = max(1, min(s, W - x2, H - y2))
        return (x2, y2, s, s)

    def ema_window(self, t: dict, win: tuple, alpha: float = 0.25) -> tuple[float, float, float]:
        """
        Apply EMA smoothing on window center and size.

        Args:
            t: Tracker dict (EMA state stored in-place).
            win: Current window (x, y, w, h).
            alpha: EMA update factor.

        Returns:
            Smoothed (cx, cy, s) as floats.
        """
        x, y, w, _ = win
        cx, cy = x + w / 2.0, y + w / 2.0
        s = float(w)

        if t["ema_cx"] is None:
            t["ema_cx"], t["ema_cy"], t["ema_s"] = cx, cy, s
        else:
            t["ema_cx"] = (1 - alpha) * t["ema_cx"] + alpha * cx
            t["ema_cy"] = (1 - alpha) * t["ema_cy"] + alpha * cy
            t["ema_s"] = (1 - alpha) * t["ema_s"] + alpha * s

        return t["ema_cx"], t["ema_cy"], t["ema_s"]

    def clamp_square_from_center(self, cx: float, cy: float, s: float, W: int, H: int) -> tuple:
        """
        Convert (center,size) to a clamped square window in image coordinates.

        Args:
            cx, cy: Center in pixels.
            s: Side length in pixels.
            W, H: Image width/height.

        Returns:
            Window (x, y, s, s) clamped to bounds.
        """
        s2 = int(max(1, s))
        x2 = int(cx - s2 / 2)
        y2 = int(cy - s2 / 2)

        x2 = max(0, min(x2, W - 1))
        y2 = max(0, min(y2, H - 1))
        s2 = max(1, min(s2, W - x2, H - y2))
        return (x2, y2, s2, s2)

    def smooth_with_history(self, t: dict, win: tuple, W: int, H: int) -> tuple:
        """
        Median smoothing over recent windows (center + size).

        Args:
            t: Tracker dict containing 'win_hist'.
            win: Current window (x, y, w, h).
            W, H: Image width/height.

        Returns:
            Smoothed window (x, y, s, s) clamped to bounds.
        """
        x, y, w, _ = win
        cx = x + w / 2.0
        cy = y + w / 2.0
        s = float(w)

        t["win_hist"].append((cx, cy, s))
        arr = np.array(t["win_hist"], dtype=np.float32)

        cx_m = float(np.median(arr[:, 0]))
        cy_m = float(np.median(arr[:, 1]))
        s_m = float(np.median(arr[:, 2]))

        return self.clamp_square_from_center(cx_m, cy_m, s_m, W, H)

    # ----------------------------
    # Tracking per color
    # ----------------------------
    def roi_around_last(self, last_win: tuple, W: int, H: int, scale: float = 2.0) -> tuple:
        """
        Build a square ROI around the last known window (to speed up search).

        Args:
            last_win: Last window (x, y, w, h).
            W, H: Image width/height.
            scale: Scaling factor for ROI size.

        Returns:
            ROI window (x, y, s, s) clamped to bounds.
        """
        x, y, w, h = last_win
        cx, cy = x + w / 2.0, y + h / 2.0
        s = int(max(1, max(w, h) * scale))
        x2 = int(cx - s / 2)
        y2 = int(cy - s / 2)
        x2 = max(0, min(x2, W - 1))
        y2 = max(0, min(y2, H - 1))
        s = max(1, min(s, W - x2, H - y2))
        return (x2, y2, s, s)

    def track_color(self, color: str, vis: np.ndarray, hsv: np.ndarray) -> np.ndarray:
        """
        Track a single color in the current frame and draw its window on 'vis'.

        Steps:
          1) Backprojection + blob extraction (ROI-first, then full image)
          2) Confidence check
          3) Jump rejection (center/area changes)
          4) Square bbox + temporal smoothing (history median + EMA)
          5) Update tracker state + draw rectangle

        Args:
            color: "red" | "green" | "blue"
            vis: Visualization image (BGR) to draw on.
            hsv: HSV image of the same frame.

        Returns:
            Updated visualization image.
        """
        t = self.trackers[color]
        t["tracked"] = False

        bp = self.back_projection(color=color, hist=t["hue_hist"], hsv=hsv)
        H, W = bp.shape[:2]
        thr = self.color_params[color]["tight_thr"]

        # Prefer ROI search around last position
        roi_win = None
        if t["last_window"] is not None:
            roi_win = self.roi_around_last(t["last_window"], W, H, scale=self.roi_search_scale)

        win = None
        if roi_win is not None:
            win = self.tight_bbox_from_bp(bp, search_win=roi_win, thr=thr, min_area=self.min_blob_area)
        if win is None:
            win = self.tight_bbox_from_bp(bp, search_win=None, thr=thr, min_area=self.min_blob_area)

        # No detection: optionally keep last result
        if win is None:
            if not self.keep_last_on_fail:
                self.reset_tracker(t, keep_last=False)
            return vis

        # Basic confidence filter
        conf = self.conf_in_window(bp, win)
        if conf < self.conf_thresh:
            if not self.keep_last_on_fail:
                self.reset_tracker(t, keep_last=False)
            return vis

        # Reject sudden jumps (likely false positives)
        if t["last_window"] is not None:
            lx, ly, lw, lh = t["last_window"]
            lc = (lx + lw / 2.0, ly + lh / 2.0)

            x, y, w, h = win
            c = (x + w / 2.0, y + h / 2.0)

            dist = ((c[0] - lc[0]) ** 2 + (c[1] - lc[1]) ** 2) ** 0.5
            area = w * h
            larea = max(1, lw * lh)
            area_ratio = max(area / larea, larea / area)

            if dist > self.jump_center_px or area_ratio > self.jump_area_ratio:
                self.reset_tracker(t, keep_last=True)

        # Square + smoothing
        x, y, w, h = win
        win = self.square_box(x, y, w, h, W, H, scale=self.square_scale)

        if self.use_history_median:
            win = self.smooth_with_history(t, win, W, H)

        cx, cy, s_ema = self.ema_window(t, win, alpha=self.ema_alpha)
        win = self.clamp_square_from_center(cx, cy, s_ema, W, H)

        # Draw bbox
        cv2.rectangle(
            vis,
            (win[0], win[1]),
            (win[0] + win[2], win[1] + win[3]),
            self.DRAW[color]["box"],
            2,
        )

        # Update state
        t["window"] = win
        t["last_window"] = win
        t["tracked"] = True
        return vis

    # ----------------------------
    # Display loop
    # ----------------------------
    def display_loop(self) -> None:
        """
        Main node loop:
          - Process latest frame (if available)
          - Track all colors
          - Publish bboxes
          - Show visualization window
          - spin_once for ROS callbacks

        Returns:
            None
        """
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


def main() -> None:
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

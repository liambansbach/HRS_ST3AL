#!/usr/bin/env python3

"""
ROS2 CubeDetector Node (CamShift + Optical Flow)

Track up to three colored cubes (red/green/blue) in an RGB camera stream and publish their
2D image-space bounding boxes as well (s. src/ainex_interfaces/msg/CubeBBox.msg)

Pipeline:
1) Histogram backprojection (Hue channel) -> probability map
2) Global blob detection to initialize window if needed
3) CamShift update (robust color-based tracking) -> rotated rectangle + updated window
4) Axis-aligned window stabilization
   - size clamped per frame (max grow/shrink)
   - size frozen when confidence is low (occlusions)
   - low-pass smoothing of w/h to reduce jitter
5) Optional optical-flow refinement (Lucas-Kanade)
   - track feature points inside the window
   - if flow "explodes" (box much larger than CamShift), reset flow only
   - use flow to update center while keeping CamShift-derived size
6) Publish CubeBBoxList 

Tuning:
All parameters that typically need tuning are grouped and marked with:  # [TUNE]
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
    """
    Node that tracks colored cubes using CamShift + optical flow.

    Subscribes
    - camera_image/undistorted (sensor_msgs/Image): BGR8 image stream.

    Publishes
    - cubes_position (src/ainex_interfaces/msg/CubeBBoxList.msg): 
    list of CubeBBox messages (id, cx, cy, w, h, angle), one per color.

    Notes
    - For each color we keep a small tracker state dict in `self.trackers`.
    - `window` is the *current* tracked (x,y,w,h), where (x,y) is the top left corner of the window.
    - `last_window` is the last known valid window (used if tracking fails in a frame).
    """

    def __init__(self):
        """Initialize parameters, ROS interfaces, and per-color tracker states."""
        super().__init__('cube_detector')
        self.cwd = Path.cwd()

        # ROS/IO
        self.cb_group = ReentrantCallbackGroup()
        self.bridge = CvBridge()
        self.frame = None

        # TODO adapt paths and update
        # Histogram paths
        self.red_hist_path = "/home/hrs2025/Desktop/HRS_ST3AL/src/vision/histograms/hist_red.npy"
        self.green_hist_path = "/home/hrs2025/Desktop/HRS_ST3AL/src/vision/histograms/hist_green.npy"
        self.blue_hist_path = "/home/hrs2025/Desktop/HRS_ST3AL/src/vision/histograms/hist_blue.npy"

        self.trackers = {
            "red":   {
                "hist": np.load(self.red_hist_path),   "window": None, "prev_area": None,
                "prev_w": None, "prev_h": None, "w_smooth": None, "h_smooth": None,
                "flow_pts": None, "flow_prev_gray": None,
                "last_window": None, "tracked": False, "rotated_rect": None, "last_rotated_rect": None,
                },

            "green": {
                "hist": np.load(self.green_hist_path), "window": None, "prev_area": None,
                "prev_w": None, "prev_h": None, "w_smooth": None, "h_smooth": None,
                "flow_pts": None, "flow_prev_gray": None,
                "last_window": None, "tracked": False, "rotated_rect": None, "last_rotated_rect": None,
            },
            
            "blue":  {
                "hist": np.load(self.blue_hist_path),  "window": None, "prev_area": None,
                "prev_w": None, "prev_h": None, "w_smooth": None, "h_smooth": None,
                "flow_pts": None, "flow_prev_gray": None,
                "last_window": None, "tracked": False, "rotated_rect": None, "last_rotated_rect": None,
            }
        }

        # Tracking parameters
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        # Optical flow parameters
        self.flow_min_pts = 25
        self.feature_params = dict(
            maxCorners=200, qualityLevel=0.01, minDistance=5, blockSize=7
            )
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        # Drawing colors (bgr)
        self.DRAW = {
            "red":   {"camshift": (0, 0, 255), "of": (0, 0, 200)},
            "green": {"camshift": (0, 120, 0), "of": (0, 80, 0)},
            "blue":  {"camshift": (255, 0, 0), "of": (200, 0, 0)},
        }

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

        self.pub_boxes = self.create_publisher(CubeBBoxList, "cubes_position", 10)

        # TODO slow down publishing rate for debugging
        #self.pub_every_n = 20
        #self._pub_ctr = 0

    def camera_cb(self, msg: Image):
        """
        Subscription callback: store the most recent frame.
        
        :type msg: sensor_msgs/Image (BGR8 image)
        """
        # Convert image to OpenCV BGR
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"Failed to convert image: {e}")
            return
        self.frame = frame

    def publish_boxes(self):
        """
        Publish CubeBBoxList message for all colors.

        - For each color:
          - If tracked in current frame: publish `window`
          - Else: publish `last_window` (last known position), if available.
        """
        # TODO slow down publishing rate for debugging
        #self._pub_ctr += 1
        #if self._pub_ctr % self.pub_every_n != 0:
        #    return
        
        msg = CubeBBoxList()
        msg.cubes = []

        for color, t in self.trackers.items():
            rr = t["rotated_rect"] if t["tracked"] else t["last_rotated_rect"] # camshift output 
            if rr is None:
                continue

            (cx, cy), (w, h), angle = rr

            b = CubeBBox()
            b.id = color
            b.cx = float(cx)
            b.cy = float(cy)
            b.w  = float(w)
            b.h  = float(h)
            b.angle = float(angle)

            msg.cubes.append(b)

        self.get_logger().info(
            "".join([f"\n{b.id}: cx={b.cx} cy={b.cy}" for b in msg.cubes])
        )

        self.pub_boxes.publish(msg)

    """
    ===========================
            CAMSHIFT 
    ===========================
    """

    def back_projection(self, hist, hsv):
        """
        Compute a backprojection probability map from a Hue histogram.

        Inputs
        hist: np.ndarray
            Hue histogram for the target color.
        hsv: np.ndarray
            HSV image (same size as current frame).

        Outputs
        np.ndarray
            Backprojection map (uint8).
        """
        h, s, v = cv2.split(hsv)
        
        # Mask low saturation/value pixels to reduce false positives
        mask_sv = cv2.inRange(hsv, (0, 40, 40), (179, 255, 255))        # [TUNE]        
        
        bp = cv2.calcBackProject([h], [0], hist, [0, 180], 1)
        bp = cv2.GaussianBlur(bp, (7, 7), 0)        # [TUNE] kernel size: blur reduces jitter
        bp = cv2.bitwise_and(bp, bp, mask=mask_sv)
        
        # Remove low probability values for stability
        _, bp = cv2.threshold(bp, 30, 255, cv2.THRESH_TOZERO)       # [TUNE] 
        return bp
    
    def find_biggest_blob(self, bp, min_area=300):        # maybe add min_area?
        """
        Finds biggest blob.
        
        Inputs
        bp: np.ndarray
            Backprojection map depending on color
        min_area: int
            Minimum contour area to accept a blob.

        Outputs
        Window
            Bounding rectangle (x,y,w,h) of the largest blob, or None if not found.
        """

        # Binary map: any positive backprojection counts as foreground
        bw = (bp > 0).astype(np.uint8) * 255 

        # Morphology to remove noise and fill holes
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))        # [TUNE]
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)      # [TUNE]
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)     # [TUNE]

        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < min_area:       # [TUNE]
            return None
        return cv2.boundingRect(c)


    def track_color(self, color, vis, hsv):
        """
        Applies camshift to detect the cubes and - if possible - improves tracking with optical flow

        Inputs
        color: str
            One of {"red","green","blue"}.
        vis: np.ndarray
            Visualization image (BGR) to draw on.
        hsv: np.ndarray
            HSV conversion of current frame.

        Outputs
        np.ndarray
            Updated visualization image with drawn windows.
        """
        color_dict = self.trackers[color]
        color_dict["tracked"] = False
                
        hist = color_dict["hist"]
        window = color_dict["window"]
        prev_area = color_dict["prev_area"]
        prev_w = color_dict["prev_w"]
        prev_h = color_dict["prev_h"]

        # bp is grayscale map describing how well each pixel matches the histogram:
        bp = self.back_projection(hist=hist, hsv=hsv)

        # If needed: initialize window
        if window is None:
            window = self.find_biggest_blob(bp)
            if window is None:
                #self.get_logger().warn(f"[{color}] no blob found to init window")
                self.reset_tracker(color_dict)
                return vis  

        # Applying camshift update: rotated rect + new window
        rotated_rect, new_window = cv2.CamShift(bp, window, self.term_crit)
        window = new_window
        x, y, w, h = window

        # reset tracker if window suddenly grows too much (explode protection/area jump)
        # can happen if camshift suddenly focuses on a huge region, like hand or background objects
        new_area = w * h
        if prev_area is not None and new_area > prev_area * 1.5:   # [TUNE] adapt acceptable growth between consecutive frames
            #self.get_logger().warn(f"[{color}] window exploded -> reset")
            self.reset_tracker(color_dict)
            return vis
        color_dict["prev_area"] = new_area

        # optional: clamp window to image (stay inside valid bounds)
        H, W = bp.shape[:2]
        x = max(0, min(x, W-1))
        y = max(0, min(y, H-1))
        w = max(1, min(w, W-x))
        h = max(1, min(h, H-y))
        window = (x, y, w, h)

        # confidence from mean backprojection (probabililty map) inside the window
        roi = bp[y:y+h, x:x+w]
        conf = float(np.mean(roi)) if roi.size else 0.0        
        if conf < 8.0:  # tune
            #self.get_logger().warn(f"[{color}] LOST conf={conf:.1f} -> reset window")
            self.reset_tracker(color_dict)
            return vis
        
        # Stabilize axis-aligned size using CamShift rotated rect dimensions (rotated rect was often more stable)
        # remember size
        prev_w = color_dict["prev_w"] if color_dict["prev_w"] is not None else w
        prev_h = color_dict["prev_h"] if color_dict["prev_h"] is not None else h

        (ccx, ccy), (rw, rh), _ = rotated_rect
        scale = 1.0    # [TUNE] Padding around cube
        w_new = max(1.0, rw * scale)
        h_new = max(1.0, rh * scale)


        # low confidence -> keep size (eg when cube is occluded)
        conf_freeze = 11.0      # [TUNE] 10–20
        if conf < conf_freeze:
            w_new, h_new = float(prev_w), float(prev_h)
        else:
            # limit increase in size per frame
            max_grow = 1.30     # [TUNE]
            max_shrink = 0.85   # [TUNE] (0.7–0.9)
            w_new = min(w_new, prev_w * max_grow)
            h_new = min(h_new, prev_h * max_grow)
            w_new = max(w_new, prev_w * max_shrink)
            h_new = max(h_new, prev_h * max_shrink)

        # Low pass filter to reduce jittering
        alpha = 0.2
        w_s = color_dict["w_smooth"] if color_dict["w_smooth"] is not None else w_new
        h_s = color_dict["h_smooth"] if color_dict["h_smooth"] is not None else h_new
        w_s = (1 - alpha) * w_s + alpha * w_new
        h_s = (1 - alpha) * h_s + alpha * h_new
        color_dict["w_smooth"] = w_s
        color_dict["h_smooth"] = h_s

        # axis-aligned window around center
        w2, h2 = int(w_s), int(h_s)
        x2 = int(ccx - w2 / 2)
        y2 = int(ccy - h2 / 2)

        # clamp to image
        x2 = max(0, min(x2, W - 1))
        y2 = max(0, min(y2, H - 1))
        w2 = max(1, min(w2, W - x2))
        h2 = max(1, min(h2, H - y2))

        window = (x2, y2, w2, h2)
        color_dict["prev_w"] = w2 
        color_dict["prev_h"] = h2
        
        # Optical flow refinement
        # Check if optical flow works (per color)
        if color_dict["flow_pts"] is None or color_dict["flow_prev_gray"] is None:
            ok = self.init_flow_in_window(color_dict, window, bp=bp)
        else:
            ok = self.flow_step_update_window(color_dict)

            if ok:
                pts2 = color_dict["flow_pts"].reshape(-1, 2).astype(np.float32)

                # If flow box explodes relative to CamShift window, reset flow only
                xC, yC, wC, hC = window
                areaC = wC * hC
                xF, yF, wF, hF = cv2.boundingRect(pts2)
                areaF = wF * hF

                if areaC > 0 and areaF > areaC * 1.6:       # [TUNE]
                    color_dict["flow_pts"] = None
                    color_dict["flow_prev_gray"] = None
                else:
                    # Update only the center by flow, keep CamShift size (wC,hC)
                    mx = float(np.mean(pts2[:, 0]))
                    my = float(np.mean(pts2[:, 1]))
                    x2 = int(mx - wC / 2.0)
                    y2 = int(my - hC / 2.0)

                    x2 = max(0, min(x2, W - 1))
                    y2 = max(0, min(y2, H - 1))
                    w2 = max(1, min(int(wC), W - x2))
                    h2 = max(1, min(int(hC), H - y2))

                    window = (x2, y2, w2, h2)

        # Draw results
        c_camshift = self.DRAW[color]["camshift"]
        c_of       = self.DRAW[color]["of"]

        pts = cv2.boxPoints(rotated_rect).astype(np.int32)
        cv2.polylines(vis, [pts], True, c_camshift, 2)      # rotated camshift box

        x, y, w, h = window
        #cv2.rectangle(vis, (x, y), (x + w, y + h), c_of, 2)     # axis aligned box

        # Save tracker state
        color_dict["window"] = window
        color_dict["last_window"] = window      # keep last known position

        color_dict["rotated_rect"] = rotated_rect
        color_dict["last_rotated_rect"] = rotated_rect  # keep last known position

        color_dict["tracked"] = True
        return vis
    
    """
    ===================
        OPTICAL FLOW
    ===================
    """

    def init_flow_in_window(self, tracker, window, bp=None):
        """
        Initialize optical flow points inside the given window.
        
        Inputs
        tracker: dict
            Per-color tracker dict to update.
        window: Window
            Current window (x,y,w,h) to restrict feature detection.
        bp: Optional[np.ndarray]
            Backprojection map (optional) used to restrict features to likely target pixels.

        Outputs
        bool
            True if enough features were found and flow was initialized, else False.
        """
        if window is None or self.frame is None:
            return False

        x, y, w, h = window
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        # only features in the window
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[y:y+h, x:x+w] = 255

        # only features in the hist
        if bp is not None:
            mask_bp = (bp > 50).astype(np.uint8) * 255      # [TUNE] adapt threshold
            mask = cv2.bitwise_and(mask, mask_bp)

        pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
        if pts is None or len(pts) < self.flow_min_pts:
            return False

        tracker["flow_pts"] = pts.astype(np.float32)
        tracker["flow_prev_gray"] = gray
        return True
    
    def flow_step_update_window(self, tracker):
        """
        Run one Lucas-Kanade optical flow step to update tracked feature points.

        Inputs
        tracker: dict
            Per-color tracker dict to update.

        Outputs
        bool
            True if flow succeeded and enough points remain, else False (tracker flow is reset).
        """
        if tracker["flow_pts"] is None or tracker["flow_prev_gray"] is None or self.frame is None:
            return False

        curr_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(
            tracker["flow_prev_gray"], curr_gray, tracker["flow_pts"], None, **self.lk_params
        )

        # works?
        if p1 is None or st is None:
            tracker["flow_pts"] = None
            tracker["flow_prev_gray"] = None
            return False

        # only keep successfully tracked points
        good_new = p1[st == 1]
        if good_new.shape[0] < self.flow_min_pts:
            tracker["flow_pts"] = None
            tracker["flow_prev_gray"] = None
            return False

        # update tracker state
        tracker["flow_pts"] = good_new.reshape(-1, 1, 2).astype(np.float32)
        tracker["flow_prev_gray"] = curr_gray
        return True


    def reset_tracker(self, t):
        """
        Helper function to reset the per-color tracker dict.
        """
        # current tracking state
        t["window"] = None
        t["tracked"] = False
        t["rotated_rect"] = None

        # smoothing
        t["prev_area"] = None   
        t["prev_w"] = None
        t["prev_h"] = None
        t["w_smooth"] = None
        t["h_smooth"] = None
        
        # optical flow
        t["flow_pts"] = None
        t["flow_prev_gray"] = None

    def display_loop(self):
        """
        Main processing loop:

        - Wait for a frame
        - Convert to HSV once per frame
        - Track each color sequentially
        - Publish tracked boxes
        - Show visualization window
        """
        while rclpy.ok():
            if self.frame is not None:

                hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
                vis = self.frame.copy()
                for color in ("red", "green", "blue"):
                    vis = self.track_color(color, vis, hsv)
                
                self.publish_boxes()
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

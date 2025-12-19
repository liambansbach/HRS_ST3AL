#!/usr/bin/env python3

"""
Docstring for vision.vision.detect_cubes
"""

from pathlib import Path

import numpy as np
import cv2
from cv_bridge import CvBridge
import yaml
from ament_index_python.packages import get_package_share_directory
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

        self.hsv_ranges = {                # TODO ggf boundaries tighter
        "green": [((60, 70, 40), (90, 255, 255))],
        "blue":  [((95, 60, 40), (125, 255, 255))],
        "red":   [((0, 70, 40), (10, 255, 255)),
                ((165, 70, 40), (179, 255, 255))],
        }

        # Noise filters (tune)
        self.min_area = 800
        self.max_area = 30000000

        self.min_w = 15
        self.max_w = 8000000
        
        self.min_h = 15
        self.max_h = 8000000

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
    TODO
    - assign colors to cubes based on inner square color
    - publish cube poses and colors
    - optional: improve detection (e.g., filter by size, use contours better)
    """

    def bgr_to_hsv(self, bgr):
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    def make_color_mask(self, hsv, color_ranges):
        """creates a binary mask from hsv; depends on input color_range"""
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in color_ranges:
            m = cv2.inRange(
                hsv,
                np.array(lower, dtype=np.uint8),
                np.array(upper, dtype=np.uint8),
            )
            mask = cv2.bitwise_or(mask, m)
        return mask

    def extract_blobs(self, binary_mask):
        """extracts blobs; receives binary mask depending on input color from make_color_mask()"""

        """
        TODO
        maybe adapt kernel sizes as like this erosion/dilation is quite strong
        """
        initial_erosion = cv2.erode(binary_mask, np.ones((3,3), np.uint8), iterations = 2)
        dilation = cv2.dilate(initial_erosion, np.ones((3,3), np.uint8), iterations = 6)
        post_erosion = cv2.erode(dilation, np.ones((3,3), np.uint8), iterations = 2)
        
        # optional: filter very small blobs (noise) by accessing the area from the stats
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(post_erosion, connectivity=8, ltype=cv2.CV_32S)

        filtered = np.zeros_like(binary_mask, dtype=np.uint8)

        """
        TODO
        adapt threshold for area, min_w and min_h
        """

        min_area = self.min_area
        max_area = self.max_area

        min_w = self.min_w
        max_w = self.max_w

        min_h = self.min_h
        max_h = self.max_h

        for i in range(1, num_labels):  # 0 is background
            area = stats[i, cv2.CC_STAT_AREA]
            w    = stats[i, cv2.CC_STAT_WIDTH]
            h    = stats[i, cv2.CC_STAT_HEIGHT]

            # noise filter
            if not (min_area <= area <= max_area):
                continue
            if w < min_w or h < min_h:
                continue
            if w > max_w or h > max_h:
                continue
            # keep this blob
            filtered[labels == i] = 255

        return filtered

    def find_contours(self, binary_mask):
        """finds contours and filters noise; receives cleaned mask from extract_blob() """
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

      
    def find_contour_center(self, contour):
        """returns center points of one contour"""
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return (-1, -1)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)

    def bbox(self, contour):
        rect = cv2.minAreaRect(contour)          # ((cx,cy),(w,h),angle)
        (cx, cy), (w, h), angle = rect
        box = cv2.boxPoints(rect)                # 4x2 float
        box = box.astype(np.int32)
        return (cx, cy), (w, h), angle, box

    """
    TODO
    maybe add metrics and is_cube_candidate later for more accuracte shape validation
    """
    def shape_metrics(self, contours):
        pass


    def is_cube_candidate(self, contour) -> bool:
        area = cv2.contourArea(contour)
        if area < self.min_area or area > self.max_area:
            return False

        # Rotated bounding box
        (_, _), (w, h), _ = cv2.minAreaRect(contour)
        w, h = float(w), float(h)
        if w < 1 or h < 1:
            return False    

        # Kriterien zur besseren shape validation von chat
        aspect = max(w, h) / min(w, h)           # >= 1
        rect_area = w * h
        extent = area / max(rect_area, 1e-6)     # how well it fills its rotated box

        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / max(hull_area, 1e-6)

        
        if aspect > 1.8:        # too skinny to be a cube face blob
            return False
        if extent < 0.55:       # blob does not look rectangular
            return False
        if solidity < 0.85:     # too jagged / noisy
            return False

        return True

    def detect_color(self, bgr, color_name, color_ranges):
        hsv = self.bgr_to_hsv(bgr)
        mask = self.make_color_mask(hsv, color_ranges)
        cleaned = self.extract_blobs(mask)
        contours = self.find_contours(cleaned)

        dets = []
        for c in contours:
            if not self.is_cube_candidate(c):
                continue
            center = self.find_contour_center(c)
            bbox_center, (w, h), angle, box = self.bbox(c)

            # Von chat vorgeschlagen, vielleicht später relevant
            dets.append({
                "color": color_name,
                "contour": c,
                "center": center,
                "box": box,
                "bbox_center": bbox_center,
                "size": (w, h),
                "angle": angle,
                "metrics": None,
            })

        return dets, cleaned

    def detect_all_colors(self, bgr):
        results = {}
        combined = np.zeros(bgr.shape[:2], dtype=np.uint8)

        for color, ranges in self.hsv_ranges.items():
            dets, cleaned = self.detect_color(bgr, color, ranges)  
            results[color] = dets                                   # vllt später relevant
            combined = cv2.bitwise_or(combined, cleaned)

        return results, combined


    def draw_detections(self, bgr, results_by_color):
        out = bgr.copy()
        for color, dets in results_by_color.items():
            for d in dets:
                box = d["box"]
                cx, cy = d["center"]
                cv2.drawContours(out, [box], 0, (0, 255, 0), 2)
                if cx != -1:
                    #cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(out, color, (cx + 6, cy - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return out


    def display_loop(self):
        """Main loop to display detected cubes."""
        while rclpy.ok():
            if self.frame is not None:
                frame = self.frame.copy()

                results, blob_mask = self.detect_all_colors(frame)
                vis = self.draw_detections(frame, results)

                cv2.imshow("Camera", frame)
                cv2.imshow("Blobs", blob_mask)
                cv2.imshow("Detections", vis)
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
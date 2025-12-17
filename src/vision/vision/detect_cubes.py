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

        # QoS: Reliable to ensure camera_info is received
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub_compressed = self.create_subscription(
            Image,
            'camera_image/undistorted',
            self.camera_cb,
            sensor_qos,
            callback_group=self.cb_group,
        )

    def camera_cb(self, msg: Image):
        # Convert compressed image to OpenCV BGR
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"Failed to convert image: {e}")
            return
        

    def find_squares(self, img):
        """Detect squares in the image."""
        squares = []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            cnt_len = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                area = cv2.contourArea(approx)
                if area > 1000:
                    squares.append(approx)
        return squares
    

    def display_loop(self):
        """Main loop to display detected cubes."""
        while rclpy.ok():
            rclpy.spin_once(self)
            if self.frame is not None:
                squares = self.find_squares(self.frame)
                display_frame = self.frame.copy()
                cv2.drawContours(display_frame, squares, -1, (0, 255, 0), 3)
                cv2.imshow("Detected Cubes", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
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
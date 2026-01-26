#!/usr/bin/env python3
"""
TUM - ICS AiNex CameraSubscriberCompressed Demo for ROS 2 Jazzy
----------------------------------------
Subscribes to compressed camera images and camera info,
Requires:
  sudo apt install python3-numpy python3-opencv

Msgs:
    sensor_msgs/CompressedImage
    sensor_msgs/CameraInfo


Group B:
    Liam Bansbach
    Marius Moe Rena
    Niklas Peter
    Tobias Töws
    Maalon Jochmann
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
from sensor_msgs.msg import CompressedImage, CameraInfo, Image
from rclpy.callback_groups import ReentrantCallbackGroup

class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_sub')
        self.cwd = Path.cwd()
        
        self.frame = None
        self.camera_k = None
        self.camera_d = None
        self.camera_width = None
        self.camera_heigth = None

        self.camera_info_received = False
        self.show_compressed = False

        self.cb_group = ReentrantCallbackGroup()

        # QoS: Reliable to ensure camera_info is received
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Load calibration file parameter
        pkg_share = Path(get_package_share_directory('vision'))
        default_calib = pkg_share / 'config' / 'calibration.yaml'
        self.declare_parameter('calibration_file', str(default_calib))
        calib_file = self.get_parameter('calibration_file').get_parameter_value().string_value
        self.get_logger().info(f"Using calibration file: {calib_file}") 

        # Bridge for CompressedImage -> OpenCV
        self.bridge = CvBridge()

        # Load camera calibration
        self.camera_matrix, self.dist_coeffs = self._load_calibration(calib_file)

        if self.camera_matrix is None or self.dist_coeffs is None:
            self.get_logger().warn(
                "Camera calibration could not be loaded. "
                "Node will still run but without undistortion."
            )

        # Subscribe compressed images
        self.sub_compressed = self.create_subscription(
            CompressedImage,
            'camera_image/compressed',
            self.image_callback_compressed,
            sensor_qos,
            callback_group=self.cb_group,
        )
        self.sub_compressed

        # Subscribe camera info
        self.sub_camerainfo = self.create_subscription(
            CameraInfo,
            'camera_info',
            self.camera_info_callback,
            sensor_qos,
            callback_group=self.cb_group
        )
        self.sub_camerainfo

        self.camera_undist_pub = self.create_publisher(Image, "camera_image/undistorted", 10)


    def camera_info_callback(self, msg: CameraInfo):
        '''
        Docstring for camera_info_callback
        
        :param self: Description
        :param msg: Description
        :type msg: CameraInfo
        '''
        if not self.camera_info_received:
            self.get_logger().info(
                f'Camera Info received: {msg.width}x{msg.height}\n'
                f'K: {msg.k}\n'
                f'D: {msg.d}'
            )
            print(f'Camera Info received: {msg.width}x{msg.height}')
            print(f'Intrinsic matrix K: {msg.k}')
            print(f'Distortion coeffs D: {msg.d}')
            self.camera_info_received = True

            self.camera_k = msg.k
            self.camera_d = msg.d
            self.camera_width = msg.width
            self.camera_heigth = msg.height

    def _load_calibration(self, calib_path: str):
        """
        Load camera_matrix and distortion_coefficients from a YAML file.
        """
        calib_path = Path(calib_path)
        if not calib_path.is_file():
            self.get_logger().error(f"Calibration file not found: {calib_path}")
            return None, None

        try:
            with calib_path.open('r') as f:
                calib = yaml.safe_load(f)
        except Exception as e:
            self.get_logger().error(f"Failed to read calibration file: {e}")
            return None, None

        try:
            # camera_matrix: list of 3 lists (3x3)
            K_list = calib['camera_matrix']
            # distortion_coefficients: list with one list inside
            D_list = calib['distortion_coefficients']
            K = np.array(K_list, dtype=np.float32).reshape(3, 3)
            D = np.array(D_list, dtype=np.float32).ravel()
        except KeyError as e:
            self.get_logger().error(f"Missing key in calibration file: {e}")
            return None, None

        self.get_logger().info("Loaded camera calibration.")
        return K, D


    def image_callback_compressed(self, msg: CompressedImage):
        '''
        Docstring for image_callback_compressed
        
        :param self: Description
        :param msg: Description
        :type msg: CompressedImage
        '''
        # Convert compressed image to OpenCV BGR
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"Failed to convert image: {e}")
            return

        # Undistort if calibration is available
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            self.frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

        else:
            self.frame = frame

        # Keep numpy frame for display
        self.undist = self.frame

        # Convert to ROS Image and publish
        img_msg = self.bridge.cv2_to_imgmsg(self.frame, encoding='bgr8')
        img_msg.header = msg.header          # keep timestamp/frame_id from incoming image
        self.camera_undist_pub.publish(img_msg)

    def process_key(self):
        '''
        
        '''
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False  # Quit
        if key == ord('c'):
            self.show_compressed = True
            self.get_logger().info('Switched to compressed image')
        return True


    def display_loop(self):
        '''
        
        '''
        while rclpy.ok():
            if self.frame is not None:
                # Display the compressed image 
                cv2.imshow('Camera Compressed undistorted', self.undist)
                #input("press key to continue...")
            if not self.process_key():
                break

            rclpy.spin_once(self, timeout_sec=0.01)

        cv2.destroyAllWindows()

def main():
    rclpy.init()
    node = CameraSubscriber()
    node.get_logger().info('CameraSubscriber node started')

    try:
        # NOT Visualize Display
        rclpy.spin(node)
        # Visualize Display
        #node.display_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
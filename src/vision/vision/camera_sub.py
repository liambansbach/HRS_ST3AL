#!/usr/bin/env python3
"""
ROS 2 Camera Subscriber (CompressedImage -> undistorted Image publisher)

This node subscribes to:
  - sensor_msgs/CompressedImage on `camera_image/compressed`
  - sensor_msgs/CameraInfo on `camera_info`

It loads a camera calibration YAML (intrinsics + distortion coefficients),
undistorts incoming compressed frames using OpenCV, publishes the undistorted
result as `sensor_msgs/Image` on `camera_image/undistorted`, and optionally
displays the undistorted stream in an OpenCV window.

Key bindings (OpenCV window):
  - 'q': quit
  - 'c': toggles an internal flag (kept for compatibility; display remains undistorted)

Notes:
  - QoS is BEST_EFFORT with depth=1 for sensor-style topics.
  - If calibration cannot be loaded, frames are forwarded without undistortion.

HRS 2025 - Group B:
    Liam Bansbach
    Marius Moe Rena
    Niklas Peter
    Tobias Töws
    Maalon Jochmann
"""

from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image


class CameraSubscriber(Node):
    """
    Subscribe to compressed camera images, undistort them, publish as Image, and optionally display.

    Responsibilities:
      - Declare and read a `calibration_file` parameter (defaults to vision/config/calibration.yaml)
      - Load camera intrinsics/distortion from YAML
      - Subscribe to `camera_image/compressed` and `camera_info`
      - Undistort frames (if calibration is available) and publish `camera_image/undistorted`
      - Provide an OpenCV display loop with basic key handling
    """

    def __init__(self) -> None:
        """
        Initialize the ROS 2 node, parameters, calibration, subscribers, and publisher.

        This constructor keeps the original behavior intact:
          - BEST_EFFORT QoS with depth=1
          - Loads calibration YAML from a parameter
          - Subscribes to compressed images and camera info
          - Publishes undistorted images as `sensor_msgs/Image`
        """
        super().__init__("camera_sub")
        self.cwd = Path.cwd()

        self.frame = None
        self.camera_k = None
        self.camera_d = None
        self.camera_width = None
        self.camera_heigth = None

        self.camera_info_received = False
        self.show_compressed = False

        self.cb_group = ReentrantCallbackGroup()

        # QoS: BEST_EFFORT with depth=1 for typical sensor topics
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Load calibration file parameter
        pkg_share = Path(get_package_share_directory("vision"))
        default_calib = pkg_share / "config" / "calibration.yaml"
        self.declare_parameter("calibration_file", str(default_calib))
        calib_file = (
            self.get_parameter("calibration_file").get_parameter_value().string_value
        )
        self.get_logger().info(f"Using calibration file: {calib_file}")

        # Bridge for CompressedImage -> OpenCV and OpenCV -> Image
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
            "camera_image/compressed",
            self.image_callback_compressed,
            sensor_qos,
            callback_group=self.cb_group,
        )
        self.sub_compressed  # keep reference

        # Subscribe camera info
        self.sub_camerainfo = self.create_subscription(
            CameraInfo,
            "camera_info",
            self.camera_info_callback,
            sensor_qos,
            callback_group=self.cb_group,
        )
        self.sub_camerainfo  # keep reference

        # Publish undistorted image as raw Image
        self.camera_undist_pub = self.create_publisher(
            Image, "camera_image/undistorted", 10
        )

    def camera_info_callback(self, msg: CameraInfo) -> None:
        """
        Handle incoming `sensor_msgs/CameraInfo`.

        Logs camera resolution and calibration parameters once (on first receipt),
        and stores the info into instance attributes for later reference.

        Args:
            msg: Incoming CameraInfo message containing width/height and calibration data.
        """
        if not self.camera_info_received:
            self.get_logger().info(
                f"Camera Info received: {msg.width}x{msg.height}\n"
                f"K: {msg.k}\n"
                f"D: {msg.d}"
            )
            print(f"Camera Info received: {msg.width}x{msg.height}")
            print(f"Intrinsic matrix K: {msg.k}")
            print(f"Distortion coeffs D: {msg.d}")
            self.camera_info_received = True

            self.camera_k = msg.k
            self.camera_d = msg.d
            self.camera_width = msg.width
            self.camera_heigth = msg.height

    def _load_calibration(self, calib_path: str):
        """
        Load camera intrinsics (K) and distortion coefficients (D) from a YAML file.

        The YAML is expected to contain:
          - `camera_matrix`: a 3x3 nested list (or list-like) describing the intrinsic matrix.
          - `distortion_coefficients`: list-like describing distortion parameters.

        Args:
            calib_path: Path to the calibration YAML file.

        Returns:
            (K, D) where:
              - K is a (3, 3) float32 numpy array
              - D is a (N,) float32 numpy array (raveled)
            If loading fails, returns (None, None).
        """
        calib_path = Path(calib_path)
        if not calib_path.is_file():
            self.get_logger().error(f"Calibration file not found: {calib_path}")
            return None, None

        try:
            with calib_path.open("r") as f:
                calib = yaml.safe_load(f)
        except Exception as e:
            self.get_logger().error(f"Failed to read calibration file: {e}")
            return None, None

        try:
            # camera_matrix: list of 3 lists (3x3)
            K_list = calib["camera_matrix"]
            # distortion_coefficients: list with one list inside
            D_list = calib["distortion_coefficients"]
            K = np.array(K_list, dtype=np.float32).reshape(3, 3)
            D = np.array(D_list, dtype=np.float32).ravel()
        except KeyError as e:
            self.get_logger().error(f"Missing key in calibration file: {e}")
            return None, None

        self.get_logger().info("Loaded camera calibration.")
        return K, D

    def image_callback_compressed(self, msg: CompressedImage) -> None:
        """
        Handle incoming `sensor_msgs/CompressedImage`.

        Converts the compressed image into an OpenCV BGR frame, undistorts it if
        calibration is available, stores it for display, then converts it back
        into a ROS `sensor_msgs/Image` and publishes on `camera_image/undistorted`.

        Args:
            msg: Incoming compressed image message.
        """
        # Convert compressed image to OpenCV BGR
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )
        except Exception as e:
            self.get_logger().warn(f"Failed to convert image: {e}")
            return

        # Undistort if calibration is available
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            self.frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
        else:
            self.frame = frame

        #self.frame = frame

        # Keep numpy frame for display
        self.undist = self.frame

        # Convert to ROS Image and publish
        img_msg = self.bridge.cv2_to_imgmsg(self.frame, encoding="bgr8")
        img_msg.header = msg.header  # keep timestamp/frame_id from incoming image
        self.camera_undist_pub.publish(img_msg)

    def process_key(self) -> bool:
        """
        Process keyboard input from the OpenCV window.

        Behavior:
          - 'q' quits the display loop
          - 'c' sets an internal `show_compressed` flag and logs (kept as-is)

        Returns:
            False to stop the display loop, True to continue.
        """
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False  # Quit
        if key == ord("c"):
            self.show_compressed = True
            self.get_logger().info("Switched to compressed image")
        return True

    def display_loop(self) -> None:
        """
        Run an interactive display loop that shows the latest undistorted frame.

        The loop:
          - Displays `self.undist` when available
          - Checks for key presses via `process_key()`
          - Calls `rclpy.spin_once()` to service callbacks

        Exits when:
          - ROS shuts down, or
          - User presses 'q'
        """
        while rclpy.ok():
            if self.frame is not None:
                # Display the compressed image (undistorted)
                cv2.imshow("Camera Compressed undistorted", self.undist)
                # input("press key to continue...")

            if not self.process_key():
                break

            rclpy.spin_once(self, timeout_sec=0.01)

        cv2.destroyAllWindows()


def main() -> None:

    rclpy.init()
    node = CameraSubscriber()
    node.get_logger().info("CameraSubscriber node started")

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


if __name__ == "__main__":
    main()

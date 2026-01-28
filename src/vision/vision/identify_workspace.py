#!/usr/bin/env python3
"""
TUM - ICS AiNex CameraSubscriberCompressed Demo for ROS 2 Jazzy

Group B:
    Liam Bansbach
    Marius Moe Rena
    Niklas Peter
    Tobias Töws
    Maalon Jochmann
"""
from typing import Tuple
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, CameraInfo
from rclpy.callback_groups import ReentrantCallbackGroup
from pathlib import Path
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation as R


class CameraSubscriber(Node):
    """
    ROS 2 Node for detecting ArUco markers in camera images, computing their 3D positions,
    and defining a workspace boundary based on marker locations.
    
    This node:
    - Subscribes to compressed camera images and camera info
    - Detects ArUco markers (DICT_6X6_250)
    - Calculates 3D pose of each marker using camera calibration
    - Defines a convex hull workspace boundary from marker centers
    - Publishes TF transforms for each detected marker
    - Provides workspace boundary checking functionality
    """

    def __init__(self):
        super().__init__('camera_subscriber')
        self.cwd = Path.cwd()

        # Camera calibration parameters (received from camera_info topic)
        self.camera_k = None  # Intrinsic camera matrix (3x3)
        self.camera_d = None  # Distortion coefficients
        self.camera_width = None
        self.camera_heigth = None

        # TF broadcaster for publishing marker transforms
        self.br = TransformBroadcaster(self)

        # Use reentrant callback group to allow parallel callback execution
        self.cb_group = ReentrantCallbackGroup()

        # QoS Profile: Best effort for real-time camera data
        # BEST_EFFORT allows dropping frames if processing is slow
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,  # Keep only the latest message
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

        # State variables
        self.camera_info_received = False

        # Image frames
        self.frame = None  # Current BGR frame from camera
        self.aruco_frame = None  # Frame with ArUco detections drawn

        # 3D pose estimation results
        self.tvecs = None  # Translation vectors (position) of markers
        self.rvecs = None  # Rotation vectors (orientation) of markers
        
        # Low-pass filter state for smooth pose tracking
        self.last_tvec = None  # Previous translation for filtering
        self.last_rvec = None  # Previous rotation for filtering
        self.alpha_pos = 0.2   # Position filter weight (lower = smoother, slower response)
        self.alpha_rot = 0.2   # Rotation filter weight (lower = smoother, slower response)

    def camera_info_callback(self, msg: CameraInfo):
        """
        Callback for camera_info topic. Stores camera calibration parameters.
        Only processes the first received message since camera parameters are static.
        
        :param msg: CameraInfo message containing calibration data
        """
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

            # Store calibration parameters for 3D pose estimation
            self.camera_k = msg.k  # 3x3 intrinsic matrix (flattened)
            self.camera_d = msg.d  # Distortion coefficients
            self.camera_width = msg.width
            self.camera_heigth = msg.height

    def image_callback_compressed(self, msg: CompressedImage):
        """
        Callback for compressed camera images. Decodes JPEG data to BGR format.
        
        :param msg: CompressedImage message containing JPEG encoded image
        """
        try:
            # Convert compressed image bytes to numpy array
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            
            # Decode JPEG to OpenCV BGR format
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                self.get_logger().warn('JPEG decode returned None')
                return

            # Store frame for processing in display loop
            self.frame = frame                  
        except Exception as exc:
            self.get_logger().error(f'Decode error in compressed image: {exc}')

    def process_key(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False  # Quit
        if key == ord('c'):
            self.show_compressed = True
            self.get_logger().info('Switched to compressed image')
        return True

    def display_loop(self):
        """
        Main processing loop that runs until shutdown.
        Processes frames, detects markers, computes poses, and displays results.
        """
        while rclpy.ok():
            if self.frame is not None:
                # Create a copy for ArUco visualization
                self.aruco_frame = self.frame.copy()

                # Detect ArUco markers in the frame
                corners, ids, rejected = self.aruco_marker()
                
                # Process only if exactly 4 markers are detected (workspace corners)
                if len(corners) == 4:
                    # Define workspace boundary from marker positions
                    self.define_boundary_box(corners)
                    
                    # Calculate 3D positions if camera is calibrated
                    if self.camera_info_received:
                        self.calculate_3d_positions(corners)
                    
                    # Publish TF transforms for detected markers
                    if self.tvecs is not None and self.rvecs is not None:
                        self.publish_transforms()

                        # TEST: Verify workspace boundary checking function
                        # Calculate centroid of all marker centers
                        centers_of_centers = self.centers.mean(axis=0)
                        #print("centers_of_centers: ", centers_of_centers)
                        # Test point inside workspace (should be True)
                        test = self.check_if_in_workspace((centers_of_centers[0], centers_of_centers[1]))
                        #print("is the point in the workspace? ", test)
                        # Test point far outside workspace (should be False)
                        test = self.check_if_in_workspace((centers_of_centers[0]+1000, centers_of_centers[1]+1000))
                        #print("is the point in the workspace? ", test)

                cv2.imshow('Camera Subscrber', self.frame)
                cv2.imshow('3D marker position', self.aruco_frame)

            # Check for key press
            # maybe remove this, as this could lead to unintended closing of the node?
            if not self.process_key():
                break

            rclpy.spin_once(self, timeout_sec=0.01)

        cv2.destroyAllWindows()

    def aruco_marker(self):
        """
        Detects ArUco markers in the current frame and draws them.

        :return:
            corners: 2D coordinates of each edge of each detected marker corners, hence size Nx4
            ids: IDs of detected markers
            rejected: Rejected candidates during detection
        """
        # Check if frame is empty
        if self.aruco_frame is None or self.aruco_frame.size == 0:
            return

        gray = cv2.cvtColor(self.aruco_frame, cv2.COLOR_BGR2GRAY)

        # Make it stable for different versions of OpenCV
        DICT_ID = cv2.aruco.DICT_6X6_250  # Markers used in the HRS lab
        # DICT_ID = cv2.aruco.DICT_5X5_50  # Alternative: markers on the pizza carton
        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_ID)
        else:
            aruco_dict = cv2.aruco.Dictionary_get(DICT_ID)

        # DetectorParameters: handle different OpenCV versions
        if hasattr(cv2.aruco, "DetectorParameters_create"):
            parameters = cv2.aruco.DetectorParameters_create()
        else:
            parameters = cv2.aruco.DetectorParameters()

        # Detection: handle different OpenCV versions
        if hasattr(cv2.aruco, "ArucoDetector"):  # OpenCV >= 4.7
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, rejected = detector.detectMarkers(gray)
        else:  # Older versions
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, aruco_dict, parameters=parameters
            )

        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(self.aruco_frame, corners, ids)
        return corners, ids, rejected

    def calc_centers(self, corners):
        """
        Calculate the center point of each detected ArUco marker.
        
        :param corners: List of marker corners, each with shape (1, 4, 2)
        :return: Array of marker centers with shape (N, 2) where N is number of markers
        """
        centers = []
        for marker in corners:
            pts = marker[0]            # Extract 4 corner points (4, 2)
            center = pts.mean(axis=0)  # Calculate mean to get center (2,)
            centers.append(center)
        return np.array(centers, dtype=np.float32)

    def define_boundary_box(self, corners):
        """
        Define the workspace boundary as a convex hull around detected marker centers.
        Draws the boundary polygon and marker centers on the visualization frame.
        
        :param corners: Detected marker corners from aruco_marker()
        """
        if corners is None or len(corners) == 0:
            return

        # Step 1: Calculate center point of each marker
        self.centers = self.calc_centers(corners)  # Shape: (N, 2) float32

        # Step 2: Compute convex hull to define workspace boundary
        # The hull is the smallest convex polygon containing all marker centers
        # Alternative: cv2.minAreaRect(self.centers) for minimum bounding rectangle
        self.hull = cv2.convexHull(self.centers)  # Shape: (M, 1, 2) float32
        hull_i32 = self.hull.astype(np.int32)     # Convert to int32 for drawing

        # Step 3: Draw workspace boundary polygon (red line)
        cv2.polylines(
                self.aruco_frame,
                [hull_i32],
                isClosed=True,
                color=(0, 0, 255),  # Red (BGR format)
                thickness=2
            )
        
        # Visualize marker centers as green circles
        for (x, y) in self.centers.astype(np.int32):
            cv2.circle(self.aruco_frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    def calculate_3d_positions(self, corners, marker_length=0.05):
        """
        Calculate the 3D pose (position and orientation) of detected ArUco markers
        using camera calibration parameters.

        :param corners: Detected marker corners from aruco_marker()
        :param marker_length: Physical size of the marker's side in meters (default: 5cm)
        :return: Sets self.rvecs (rotation) and self.tvecs (translation) as numpy arrays
        """
        if self.camera_k is None or self.camera_d is None:
            self.get_logger().warn("Camera parameters not set. Cannot compute 3D positions.")
            return None, None, None

        # Convert flattened camera parameters to proper matrix formats
        camera_matrix = np.array(self.camera_k).reshape((3, 3))  # 3x3 intrinsic matrix
        dist_coeffs = np.array(self.camera_d)  # Distortion coefficients

        # Prepare lists to hold rotation and translation vectors
        rvecs = []  # Rotation vectors (axis-angle representation)
        tvecs = []  # Translation vectors (3D position in camera frame)

        # Estimate pose for each detected marker
        for i, corner in enumerate(corners):
            # Solve PnP (Perspective-n-Point) problem to get marker pose
            # Returns rotation vector (rvec) and translation vector (tvec)
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corner,
                marker_length,
                camera_matrix,
                dist_coeffs
            )
            rvecs.append(rvec)
            tvecs.append(tvec)

        # Store results as numpy arrays for further processing
        self.rvecs = np.array(rvecs)
        self.tvecs = np.array(tvecs)
        # NOTE: May require joint_state update since robot coordinate frame differs from camera

    def publish_transforms(self):
        """
        Publish TF transforms for all detected ArUco markers.
        Applies low-pass filtering for smooth tracking and converts from OpenCV
        camera frame to ROS standard coordinate frame (camera_link).
        """
        # Get raw pose data from detection (OpenCV camera coordinate frame)
        tvec = self.tvecs.astype(float)  # Translation [tx, ty, tz]
        rvec = self.rvecs.astype(float)  # Rotation (Rodrigues vector)
        
        # === Apply Low-Pass Filter to Translation (Position) ===
        # Smooths jittery detections using exponential moving average
        if self.last_tvec is None:
            tvec_f = tvec  # First frame: no filtering
        else:
            # Weighted average: alpha controls smoothing (lower = smoother)
            tvec_f = self.alpha_pos * tvec + (1.0 - self.alpha_pos) * self.last_tvec
        self.last_tvec = tvec_f

        # === Apply Low-Pass Filter to Rotation (Orientation) ===
        if self.last_rvec is None:
            rvec_f = rvec  # First frame: no filtering
        else:
            # Weighted average for rotation vector
            rvec_f = self.alpha_rot * rvec + (1.0 - self.alpha_rot) * self.last_rvec
        self.last_rvec = rvec_f

        # === Publish TF Transform for Each Marker ===
        for i in range(tvec_f.shape[0]):
            tvec_f_i = tvec_f[i].flatten()
            rvec_f_i = rvec_f[i].flatten()

            # Extract filtered translation components
            tx = tvec_f_i[0]
            ty = tvec_f_i[1]
            tz = tvec_f_i[2]

            # Create TF message
            t_msg = TransformStamped()
            t_msg.header.stamp = self.get_clock().now().to_msg()
            t_msg.header.frame_id = "base_link"  # Parent frame (robot base)
            t_msg.child_frame_id = "aruco_marker_" + str(i)  # Child frame (marker)

            # === Coordinate Frame Transformation: OpenCV -> ROS ===
            # OpenCV camera frame: Z forward, X right, Y down
            # ROS camera_link frame: X forward, Y left, Z up
            #    # Forward (camera Z -> ROS X)
            #   # Left/Right (camera -X -> ROS Y)
            #  # Up/Down (camera -Y -> ROS Z)
            cam_point = np.array([tz, -tx, -ty, 1.0], dtype=np.float32)
            
            # your first attempt for transforming the frames in here:
            """
            T_cam_to_base = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0430140359009206],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.152356120938238],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            cam_point = T_cam_to_base @ cam_point
            """
            
            #@liam just copy this:
            """
            def define_homogenous_transform(translation, rotation_angle):
                # For now, we use a fixed transformation from camera to base
                # In future, include head joint angles for dynamic transform
                T = np.eye(4, dtype=np.float32)
                T[0:3, 3] = translation
                c1 = np.cos(rotation_angle[0])
                s1 = np.sin(rotation_angle[0])
                c2 = np.cos(rotation_angle[1])
                s2 = np.sin(rotation_angle[1])
                c3 = np.cos(rotation_angle[2])s3 = np.sin(rotation_angle[2])
                # Rotation matrices for each axis
                Rx = np.array([[1, 0, 0],
                               [0, c1, -s1],
                               [0, s1, c1]], dtype=np.float32)
                Ry = np.array([[c2, 0, s2],
                               [0, 1, 0],
                               [-s2, 0, c2]], dtype=np.float32)
                Rz = np.array([[c3, -s3, 0],
                               [s3, c3, 0],
                               [0, 0, 1]], dtype=np.float32)
                # Combined rotation matrix
                R_combined = Rz @ Ry @ Rx
                T[0:3, 0:3] = R_combined
                return T
            head_yaw_angle = 0.0 #np.pi/2  # Placeholder value
            head_pitch_angle = 0.0    # Placeholder value
            rotation_angles = [head_yaw_angle, head_pitch_angle, 0.0]  # roll is zero
            T_cam_to_base = define_homogenous_transform(translation=np.array([0.0430140359009206, 0.0, 0.152356120938238]), rotation_angle=rotation_angles)
            cam_point = T_cam_to_base @ cam_point
            """

            

            t_msg.transform.translation.x = float(cam_point[0])
            t_msg.transform.translation.y = float(cam_point[1])
            t_msg.transform.translation.z = float(cam_point[2])
            # === Convert Rotation Vector to Quaternion ===
            # Convert Rodrigues rotation vector to rotation matrix
            R_cv, _ = cv2.Rodrigues(rvec_f_i)
            # Convert rotation matrix to quaternion using scipy
            r = R.from_matrix(R_cv)
            qx, qy, qz, qw = r.as_quat()  # Returns [x, y, z, w] format

            t_msg.transform.rotation.x = float(qx)
            t_msg.transform.rotation.y = float(qy)
            t_msg.transform.rotation.z = float(qz)
            t_msg.transform.rotation.w = float(qw)



            # Broadcast the transform
            self.br.sendTransform(t_msg)

    # TODO: Position robot and scan for markers (requires joint_state update in RViz)

    def check_if_in_workspace(self, point: Tuple[float, float]) -> bool:
        """
        Check if a 2D point lies within the defined workspace boundary.
        
        :param point: Tuple of (x, y) coordinates in image space (pixels)
        :return: True if point is inside or on the boundary, False if outside
        """
        # Prepare convex hull as contour for point-in-polygon test
        # cv2.pointPolygonTest expects contour with shape (N, 1, 2) or (N, 2)
        contour = self.hull.astype(np.float32)

        # Perform point-in-polygon test
        # Returns: +1 if inside, 0 if on edge, -1 if outside (when measureDist=False)
        res = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False)
        
        # Return True if point is inside or on the boundary
        return res >= 0


def main(args=None):
    rclpy.init(args=args)
    node = CameraSubscriber()
    node.get_logger().info('CameraSubscriber node started')

    try:
        node.display_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

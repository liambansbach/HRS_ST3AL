#!/usr/bin/env python3
"""
TUM - ICS AiNex CameraSubscriberCompressed Demo for ROS 2 Jazzy
----------------------------------------
Subscribes to JPEG-compressed images and raw images on /camera_image/compressed and /camera_image,
shows frames with OpenCV, and displays CameraInfo.

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
from typing import Tuple
import numpy as np
#import matplotlib.pyplot as plt
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, CameraInfo
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Point, Vector3
from pathlib import Path

class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        self.cwd = Path.cwd()
        #self.track_window = None

        #self.optical_flow_flag = False
        # self.reference_img_gray = None      # Referenzbild (grau)
        # self.reference_pts = None       # Punkte aus dem Referenzbild
        # self.prev_frame_gray = None     # Vorframe (grau)
        # self.current_frame_pts = None           # Punkte im aktuellen Frame (Nx1x2)
        # self.mask = None          # Layer für Trajektorien
        # self.random_colors = np.random.randint(0,255,(300,3)).tolist()

        self.camera_k = None
        self.camera_d = None
        self.camera_width = None
        self.camera_heigth = None


        # gleich laden:
        #self.reference_img = cv2.imread(str(Path.cwd()) + "/screenshots/tutorial_3/templateImg2.png") 
        self.aruco_video = cv2.VideoCapture(str(Path.cwd()) + "/screenshots/tutorial_3/aruco_marker_video.mp4") 


        self.cb_group = ReentrantCallbackGroup()

        # QoS: Reliable to ensure camera_info is received
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
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

        self.frame = None # BGR Frame
        self.aruco_frame = None

        # self.image_shown = False
        # self.normalized_roi_hist = self.read_show_image()   # returns hist now
        


    def camera_info_callback(self, msg: CameraInfo):
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

    def image_callback_compressed(self, msg: CompressedImage):
        try:
            # Decode the compressed image
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                self.get_logger().warn('JPEG decode returned None')
                return

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
        while rclpy.ok():
            if self.frame is not None:
                # Display the compressed image 
                #self.shift_frame = self.frame.copy()
                #self.optical_flow_frame = self.frame.copy()   
                self.aruco_frame = self.frame.copy() #aruco detection from ainex cam
                #ret, self.aruco_frame = self.aruco_video.read()  # aruco detection from tet video

                # TODO comment out the parts that dont need to be displayed.
                #self.read_show_image()
                #self.back_projection()
                #self.apply_meanshift()
                #self.apply_camshift()
                #self.optical_flow_step()
                corners, ids, rejected = self.aruco_marker()
                self.define_boundary_box(corners)
                self.calculate_3d_positions(corners)

                cv2.imshow('Camera Subscrber', self.frame)
                cv2.imshow('3D marker position', self.aruco_frame)
                #cv2.imshow('cam/mean shift', self.shift_frame)
                #cv2.imshow('optical flow: ', self.optical_flow_frame)
            
            # Check for key press
            # maybe remove this, as this could lead to unintended closing of the node?
            if not self.process_key():
                break

            rclpy.spin_once(self, timeout_sec=0.01)

        cv2.destroyAllWindows()

    def aruco_marker(self):
        """
        Detects ArUco markers in the current frame and draws them.

        :param self: none
        
        :return: 
        corners: 2D coordinates of each edge of each detected marker corners, hence size Nx4 
        ids: IDs of detected markers
        rejected: Rejected candidates during detection
        """


        # check if frame is empty
        if self.aruco_frame is None or self.aruco_frame.size == 0:
            return

        gray = cv2.cvtColor(self.aruco_frame, cv2.COLOR_BGR2GRAY)

        # make it stable for different versions of opencv because the syntax differs for the aruco lib
        DICT_ID = cv2.aruco.DICT_6X6_250 #markers used in the hrs lab
        #DICT_ID = cv2.aruco.DICT_5X5_50 #markers on the pizza carton
        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_ID)
        else:
            aruco_dict = cv2.aruco.Dictionary_get(DICT_ID)

        # DetectorParameters -> make it stable for different versions of opencv because the syntax differs for the aruco lib
        if hasattr(cv2.aruco, "DetectorParameters_create"):
            parameters = cv2.aruco.DetectorParameters_create()
        else:
            parameters = cv2.aruco.DetectorParameters()
           

        # Detektion -> make it stable for different versions of opencv because the syntax differs for the aruco lib
        if hasattr(cv2.aruco, "ArucoDetector"):                   # OpenCV >= 4.7
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, rejected = detector.detectMarkers(gray)
        else:                                                      # Older Versions
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, aruco_dict, parameters=parameters
            )
        print("ids: ", ids, "corners: ", corners)

        cv2.aruco.drawDetectedMarkers(self.aruco_frame, corners, ids)
        #print("ids: ", ids)
        #print("type of corners: ", type(corners))
        #print("corners: ", corners)
        return corners, ids, rejected
    
    def calc_centers(self, corners):
        centers = []
        for marker in corners:
            pts = marker[0]            # (4,2)
            center = pts.mean(axis=0)  # (2,)
            centers.append(center)
        return np.array(centers, dtype=np.float32)
    

    def define_boundary_box(self, corners):

        if corners is None or len(corners) == 0:
            return

        # 1) calculate marker centers
        centers = self.calc_centers(corners)  # (N,2) float32

        # Need at least 3 points for a hull polygon
        if centers.shape[0] != 4:
            print("Not enough markers detected to define a boundary.")
            return

        # 2) convex hull on centers
        hull = cv2.convexHull(centers)          # (M,1,2) float32
        hull_i32 = hull.astype(np.int32)        # required for drawing
    
        # 3) draw (close polygon)
        cv2.polylines(
                self.aruco_frame,
                [hull_i32],
                isClosed=True,
                color=(0, 0, 255),
                thickness=2
            )
        
        # Optional: visualize the centers themselves
        for (x, y) in centers.astype(np.int32):
            cv2.circle(self.aruco_frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    def calculate_3d_positions(self, corners, marker_length=0.05):
        """
        Calculate the 3D positions of detected ArUco markers.

        :param corners: Detected marker corners from aruco_marker()
        :param marker_length: Length of the marker's side in meters
        :return: Tuple of rotation vectors, translation vectors, and IDs
        """
        if self.camera_k is None or self.camera_d is None:
            self.get_logger().warn("Camera parameters not set. Cannot compute 3D positions.")
            return None, None, None

        # Convert camera parameters to numpy arrays
        camera_matrix = np.array(self.camera_k).reshape((3, 3))
        dist_coeffs = np.array(self.camera_d)

        # Prepare lists to hold rotation and translation vectors
        rvecs = []
        tvecs = []
        #ids_list = []

        # Iterate over each detected marker
        for i, corner in enumerate(corners):
            # Estimate pose of each marker
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corner,
                marker_length,
                camera_matrix,
                dist_coeffs
            )
            rvecs.append(rvec)
            tvecs.append(tvec)

        print("3D positions: ", tvecs)
        return np.array(rvecs), np.array(tvecs)
    
    # publish 3d positions as plane in rviz?

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

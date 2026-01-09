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
    def __init__(self):
        super().__init__('camera_subscriber')
        self.cwd = Path.cwd()

        self.camera_k = None
        self.camera_d = None
        self.camera_width = None
        self.camera_heigth = None

        # TF broadcaster
        self.br = TransformBroadcaster(self)

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

        self.tvecs = None
        self.rvecs = None
        self.last_tvec = None
        self.last_rvec = None
        self.alpha_pos = 0.2   # 0.1, smaller = smoother
        self.alpha_rot = 0.2      

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
                self.aruco_frame = self.frame.copy() #aruco detection from ainex cam

                # comment out the parts that dont need to be displayed.
                corners, ids, rejected = self.aruco_marker()
                if len(corners) == 4:
                    self.define_boundary_box(corners)
                    if self.camera_info_received:
                        #self.rvecs, self.tvecs = 
                        self.calculate_3d_positions(corners)
                        #print("rvecs: ", self.rvecs, "tvecs: ", self.tvecs)
                    if self.tvecs is not None and self.rvecs is not None:
                        self.publish_transforms()#self.rvecs, self.tvecs, self.br)
                        # calculate mean of centers of markers
                        centers_of_centers = self.centers.mean(axis=0)
                        print("centers_of_centers: ", centers_of_centers)
                        test = self.check_if_in_workspace((centers_of_centers[0], centers_of_centers[1]))
                        print("is the point in the workspace? ", test)
                        test = self.check_if_in_workspace((centers_of_centers[0]+1000, centers_of_centers[1]+1000))
                        print("is the point in the workspace? ", test)



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
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(self.aruco_frame, corners, ids)
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
        self.centers = self.calc_centers(corners)  # (N,2) float32

        # 2) convex hull on centers
        # maybe use cv2.minAreaRect(self.centers) instead?
        # hull = centers, just in another order (maybe it orders it clockwise?)
        self.hull = cv2.convexHull(self.centers)          # (M,1,2) float32
        hull_i32 = self.hull.astype(np.int32)        # required for drawing
        print("hull_i32: ", hull_i32)


        # 3) draw (close polygon)
        cv2.polylines(
                self.aruco_frame,
                [hull_i32],
                isClosed=True,
                color=(0, 0, 255),
                thickness=2
            )
        
        # Optional: visualize the centers themselves
        for (x, y) in self.centers.astype(np.int32):
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

        self.rvecs = np.array(rvecs)
        self.tvecs = np.array(tvecs)
    # would require an joint_state update, as robot stands upwards in rviz but not in reality


    def publish_transforms(self):#, rvecs, tvecs, broadcaster):
        # Nimm den ersten Marker (falls du mehrere willst, musst du hier erweitern)
        tvec = self.tvecs.astype(float)   # [tx, ty, tz] im OpenCV-Kameraframe
        rvec = self.rvecs.astype(float)
        # === Low-pass Filter auf Translation ===
        if self.last_tvec is None:
            tvec_f = tvec
        else:
            tvec_f = self.alpha_pos * tvec + (1.0 - self.alpha_pos) * self.last_tvec
        self.last_tvec = tvec_f

        # === Low-pass Filter auf Rotation (auf rvec) ===
        if self.last_rvec is None:
            rvec_f = rvec
        else:
            rvec_f = self.alpha_rot * rvec + (1.0 - self.alpha_rot) * self.last_rvec
        self.last_rvec = rvec_f

        # === Koordinatensystem-Anpassung ===
        for i in range(tvec_f.shape[0]):
            tvec_f_i = tvec_f[i].flatten()
            rvec_f_i = rvec_f[i].flatten()

            tx = tvec_f_i[0]
            ty = tvec_f_i[1]
            tz = tvec_f_i[2]

            t_msg = TransformStamped()
            t_msg.header.stamp = self.get_clock().now().to_msg()
            t_msg.header.frame_id = "camera_link"
            t_msg.child_frame_id = "aruco_marker_" + str(i)

            t_msg.transform.translation.x = float(tz)    # vor der Kamera
            t_msg.transform.translation.y = float(-tx)   # links/rechts
            t_msg.transform.translation.z = float(-ty)   # hoch/runter

            # Rotation aus gefiltertem rvec
            R_cv, _ = cv2.Rodrigues(rvec_f_i)
            r = R.from_matrix(R_cv)
            qx, qy, qz, qw = r.as_quat()  # [x, y, z, w]

            t_msg.transform.rotation.x = float(qx)
            t_msg.transform.rotation.y = float(qy)
            t_msg.transform.rotation.z = float(qz)
            t_msg.transform.rotation.w = float(qw)

            # TF senden
            self.br.sendTransform(t_msg)
        

    # maybe TODO: bring robot in a specific position (including updateing the joint_state in rviz), then let it search for the markers (move head)
    

    def check_if_in_workspace(self, point: Tuple[float, float]) -> bool:
        # cv2.pointPolygonTest expects contour as (N,1,2) or (N,2)
        contour = self.hull.astype(np.float32)

        # returns: +1 inside, 0 on edge, -1 outside (when measureDist=False)
        res = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False)
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

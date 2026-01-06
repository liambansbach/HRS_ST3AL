
from typing import Tuple
import numpy as np
#import matplotlib.pyplot as plt
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, CameraInfo
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Point, Vector3, Polygon
from pathlib import Path


class IdentifyWorkspace(Node):
    
    # TODO define init

    # TODO define camera callback

    # TODO define display loop?

    # TODO Subscribe camera_image/undistorted topic

    # TODO Detect Aruco markers

    # TODO save coordinates of each corner and create a square/polygon 

    # TODO Use this square as the workspace boundary box

    # TODO Broadcast the Boundary box coordinates relative to the camera to other nodes via a ROS topic (not sure if this is accually needed)

    def aruco_marker(self):
        # check if frame is empty
        if self.aruco_frame is None or self.aruco_frame.size == 0:
            return

        gray = cv2.cvtColor(self.aruco_frame, cv2.COLOR_BGR2GRAY)

        # make it stable for different versions of opencv because the syntax differs for the aruco lib
        #DICT_ID = cv2.aruco.DICT_6X6_250 #markers used in the hrs lab
        DICT_ID = cv2.aruco.DICT_5X5_50 #markers on the pizza carton
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

        cv2.aruco.drawDetectedMarkers(self.aruco_frame, corners, ids)

        return corners, ids, rejected
    
    def define_boundary_box(self, corners):
        #TODO use the corners to define the boundary box of the workspace for later "collision avoidance"
        
        
        # TODO define the boundary box based on the detected markers and draw it on the image
        # maybe use vision_msgs/BoundingBox3D.msg
        
        # I think the corners are in a corner of the marker, maybe add some offset, to get the center of the marker?
        # corner_centers = []
        # for corner in corners:
        #     corner_array = corner[0]
        #     center_x = int(np.mean(corner_array[:, 0]))
        #     center_y = int(np.mean(corner_array[:, 1]))
        #     corner_centers.append((center_x, center_y))

        conrners_np = np.array(corners)
        
        if len(conrners_np) > 4:
            hull = cv2.convexHull(conrners_np)
        else: 
            rect = cv2.minAreaRect(conrners_np)
            hull = cv2.boxPoints(rect)
            hull = np.int0(hull)

        color_conture = (0, 0, 255)
        cv2.drawContours(self.aruco_frame,[hull],0,color_conture) # can adjust thickness here if needed
        
        # TODO publish the boundary box coordinates via a ROS topic
        if len(corners) > 4: 
            #convert corners to geometry_msgs/Polygon.msg
            polygon = Polygon()

            # first convert corners to list of geometry_msgs/Point32
            for c in corners:
                c 

            polygon.points = c
        else:



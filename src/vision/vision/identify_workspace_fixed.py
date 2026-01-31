#!/usr/bin/env python3
"""
AiNex ArUco Marker TF Broadcaster (ROS 2)

Publishes:
  head_tilt_link -> hrs_camera_link   (fixed mount offset)
  hrs_camera_link -> aruco_<id>        (pose from OpenCV, converted optical->ROS)

Why:
- head_tilt_link in RViz follows head motion correctly (x forward, y left, z up).
- OpenCV ArUco pose is in camera optical convention (x right, y down, z forward).
- Therefore we must convert optical -> ROS before publishing under a robot-link frame.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import cv2


import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import CompressedImage, CameraInfo
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


@dataclass
class MarkerFilterState:
    t: Optional[np.ndarray] = None  # (3,)
    q: Optional[np.ndarray] = None  # (4,) [x,y,z,w]


class ArucoMarkerTfBroadcaster(Node):
    def __init__(self):
        super().__init__('aruco_marker_tf_broadcaster')

        # -------------------------
        # Parameters
        # -------------------------
        self.declare_parameter('image_topic', 'camera_image/compressed')
        self.declare_parameter('camera_info_topic', 'camera_info')

        # We use head_tilt_link as stable moving head frame
        self.declare_parameter('head_frame', 'head_tilt_link')

        # Create a virtual camera frame under head_frame
        self.declare_parameter('camera_frame', 'hrs_camera_link')
        self.declare_parameter('camera_offset_xyz', [0.0, 0.019, 0.016])  # meters
        self.declare_parameter('camera_offset_rpy', [0.0, 0.0, 0.0])      # rad (keep 0 if rotation matches)

        self.declare_parameter('marker_frame_prefix', 'aruco_')
        self.declare_parameter('marker_length', 0.05)  # meters

        self.declare_parameter('aruco_dict', 'DICT_6X6_250')

        self.declare_parameter('show_debug', True)
        self.declare_parameter('debug_window_name', 'Aruco TF Broadcaster')

        self.declare_parameter('enable_filter', True)
        self.declare_parameter('alpha_pos', 0.25)
        self.declare_parameter('alpha_rot', 0.25)

        # -------------------------
        # Internals
        # -------------------------
        self.cb_group = ReentrantCallbackGroup()
        self.br = TransformBroadcaster(self)

        self.camera_info_received = False
        self.camera_k: Optional[np.ndarray] = None
        self.camera_d: Optional[np.ndarray] = None

        self.latest_frame_bgr: Optional[np.ndarray] = None

        self.filters: Dict[int, MarkerFilterState] = {}

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        image_topic = self.get_parameter('image_topic').value
        caminfo_topic = self.get_parameter('camera_info_topic').value

        self.sub_image = self.create_subscription(
            CompressedImage, image_topic, self._on_image, sensor_qos, callback_group=self.cb_group
        )
        self.sub_caminfo = self.create_subscription(
            CameraInfo, caminfo_topic, self._on_caminfo, sensor_qos, callback_group=self.cb_group
        )

        # steady processing timer
        self.timer = self.create_timer(0.03, self._process, callback_group=self.cb_group)

        self.get_logger().info(
            f"Started ArucoMarkerTfBroadcaster.\n"
            f" image_topic: {image_topic}\n"
            f" camera_info: {caminfo_topic}\n"
            f" head_frame:  {self.get_parameter('head_frame').value}\n"
            f" camera_frame:{self.get_parameter('camera_frame').value}\n"
        )

    def _on_caminfo(self, msg: CameraInfo):
        if self.camera_info_received:
            return
        try:
            self.camera_k = np.array(msg.k, dtype=np.float64).reshape((3, 3))
            self.camera_d = np.array(msg.d, dtype=np.float64)
            self.camera_info_received = True
            self.get_logger().info(f"CameraInfo received. K set, D length={len(msg.d)}.")
        except Exception as e:
            self.get_logger().error(f"Failed to parse CameraInfo: {e}")

    def _on_image(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                self.get_logger().warn("cv2.imdecode returned None")
                return
            self.latest_frame_bgr = frame
        except Exception as e:
            self.get_logger().error(f"Image decode error: {e}")

    def _process(self):
        frame = self.latest_frame_bgr
        if frame is None:
            return

        show_debug = bool(self.get_parameter('show_debug').value)
        if (not self.camera_info_received) or (self.camera_k is None) or (self.camera_d is None):
            if show_debug:
                cv2.imshow(self.get_parameter('debug_window_name').value, frame)
                cv2.waitKey(1)
            return

        # 1) Always publish the virtual camera frame under the head frame
        self._publish_camera_mount_tf()

        # 2) Detect markers
        debug_frame = frame.copy() if show_debug else None
        corners, ids = self._detect_aruco(frame, debug_frame)
        if ids is None or len(ids) == 0:
            if show_debug and debug_frame is not None:
                cv2.imshow(self.get_parameter('debug_window_name').value, debug_frame)
                cv2.waitKey(1)
            return

        marker_length = float(self.get_parameter('marker_length').value)

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, self.camera_k, self.camera_d
        )

        if show_debug and debug_frame is not None:
            for i in range(len(ids)):
                cv2.drawFrameAxes(debug_frame, self.camera_k, self.camera_d, rvecs[i], tvecs[i], marker_length * 0.5)

        # 3) Publish marker TFs under hrs_camera_link (ROS convention!)
        camera_frame = self.get_parameter('camera_frame').value
        prefix = self.get_parameter('marker_frame_prefix').value

        for i in range(len(ids)):
            marker_id = int(ids[i][0])

            rvec = rvecs[i].reshape(3)
            tvec_opt = tvecs[i].reshape(3)  # OpenCV optical convention

            # rotation optical
            R_opt, _ = cv2.Rodrigues(rvec)
            q_opt = R.from_matrix(R_opt).as_quat()

            # Convert optical pose -> ROS (x fwd, y left, z up)
            t_ros, q_ros = self._optical_to_ros_pose(tvec_opt, q_opt)

            # Filter per marker id
            t_f, q_f = self._filter_pose(marker_id, t_ros, q_ros)

            # Broadcast marker TF
            self._send_tf(camera_frame, f"{prefix}{marker_id}", t_f, q_f)

        if show_debug and debug_frame is not None:
            cv2.imshow(self.get_parameter('debug_window_name').value, debug_frame)
            cv2.waitKey(1)

    def _publish_camera_mount_tf(self):
        head_frame = self.get_parameter('head_frame').value
        camera_frame = self.get_parameter('camera_frame').value

        off = np.array(self.get_parameter('camera_offset_xyz').value, dtype=np.float64).reshape(3)
        rpy = np.array(self.get_parameter('camera_offset_rpy').value, dtype=np.float64).reshape(3)

        q = R.from_euler('xyz', rpy).as_quat()

        self._send_tf(head_frame, camera_frame, off, q)

    def _detect_aruco(self, frame_bgr: np.ndarray, debug_frame: Optional[np.ndarray]):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        dict_name = self.get_parameter('aruco_dict').value
        dict_id = getattr(cv2.aruco, dict_name, cv2.aruco.DICT_6X6_250)

        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        else:
            aruco_dict = cv2.aruco.Dictionary_get(dict_id)

        if hasattr(cv2.aruco, "DetectorParameters_create"):
            params = cv2.aruco.DetectorParameters_create()
        else:
            params = cv2.aruco.DetectorParameters()

        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(aruco_dict, params)
            corners, ids, _rej = detector.detectMarkers(gray)
        else:
            corners, ids, _rej = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

        if debug_frame is not None and corners is not None and len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(debug_frame, corners, ids)

        return corners, ids

    def _filter_pose(self, marker_id: int, t: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not bool(self.get_parameter('enable_filter').value):
            return t, q

        alpha_pos = float(self.get_parameter('alpha_pos').value)
        alpha_rot = float(self.get_parameter('alpha_rot').value)

        st = self.filters.get(marker_id)
        if st is None:
            st = MarkerFilterState()
            self.filters[marker_id] = st

        if st.t is None:
            t_f = t
        else:
            t_f = alpha_pos * t + (1.0 - alpha_pos) * st.t

        if st.q is None:
            q_f = q
        else:
            r0 = R.from_quat(st.q)
            r1 = R.from_quat(q)
            slerp = Slerp([0.0, 1.0], R.concatenate([r0, r1]))
            q_f = slerp([alpha_rot])[0].as_quat()

        st.t = t_f
        st.q = q_f
        return t_f, q_f

    def _optical_to_ros_pose(self, t_opt: np.ndarray, q_opt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optical (OpenCV): x right, y down, z forward
        ROS link (REP-103): x forward, y left, z up

        Mapping for vectors:
          x_ros =  z_opt
          y_ros = -x_opt
          z_ros = -y_opt

        Rotation basis change:
          R_ros = B * R_opt * B^{-1}
        """
        B = np.array([
            [0.0, 0.0, 1.0],   # x = z
            [-1.0, 0.0, 0.0],  # y = -x
            [0.0, -1.0, 0.0],  # z = -y
        ], dtype=np.float64)

        t_ros = (B @ t_opt.reshape(3)).astype(np.float64)

        R_opt = R.from_quat(q_opt).as_matrix()
        R_ros = B @ R_opt @ np.linalg.inv(B)
        q_ros = R.from_matrix(R_ros).as_quat().astype(np.float64)

        return t_ros, q_ros

    def _send_tf(self, parent: str, child: str, t: np.ndarray, q: np.ndarray):
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = parent
        msg.child_frame_id = child

        msg.transform.translation.x = float(t[0])
        msg.transform.translation.y = float(t[1])
        msg.transform.translation.z = float(t[2])

        msg.transform.rotation.x = float(q[0])
        msg.transform.rotation.y = float(q[1])
        msg.transform.rotation.z = float(q[2])
        msg.transform.rotation.w = float(q[3])

        self.br.sendTransform(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoMarkerTfBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
from __future__ import annotations

from collections import deque
from typing import Dict, Optional, Tuple, List

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from ainex_interfaces.msg import CubeBBoxList
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformListener, TransformBroadcaster


# -----------------------------
# Math helpers
# -----------------------------
def quat_to_rotmat(x, y, z, w):
    n = x*x + y*y + z*z + w*w
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    return np.array(
        [
            [1 - (y*y + z*z) * s, (x*y - z*w) * s,     (x*z + y*w) * s],
            [(x*y + z*w) * s,     1 - (x*x + z*z) * s, (y*z - x*w) * s],
            [(x*z - y*w) * s,     (y*z + x*w) * s,     1 - (x*x + y*y) * s],
        ],
        dtype=np.float64,
    )


def rotmat_to_quat(R):
    tr = float(np.trace(R))
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    qn = np.linalg.norm(q)
    if qn < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    q /= qn
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def invert_rt(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Invert rigid transform p_tgt = R p_src + t  ->  p_src = R^T p_tgt - R^T t"""
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv


# -----------------------------
# Node
# -----------------------------
class WorkspaceProjectionNode(Node):
    def __init__(self):
        super().__init__("workspace_projection")

        # Frames
        self.declare_parameter("preferred_world_frames", ["body_link", "base_link", "head_tilt_link"])
        self.declare_parameter("camera_frame", "hrs_camera_link")

        # Leave as you had it (because it fixed your mirroring empirically)
        self.declare_parameter("invert_world_cam_tf", False)

        # CameraInfo topic (so we don't read YAML intrinsics anymore)
        self.declare_parameter("camera_info_topic", "camera_info")

        # Distortion handling for distorted pixel coords
        self.declare_parameter("undistort_pixels", True)

        # Markers used for plane
        self.declare_parameter("marker_ids", [0, 1, 2, 3])
        self.declare_parameter("marker_prefix", "aruco_")

        # Outputs
        self.declare_parameter("plane_frame", "hrs_workspace_plane")
        self.declare_parameter("cube_prefix", "hrs_cube_")

        # Cube / bbox
        self.declare_parameter("cube_size", 0.05)
        self.declare_parameter("bbox_timeout", 1.0)

        # Pixel anchor tuning
        self.declare_parameter("u_anchor", 0.50)
        self.declare_parameter("v_anchor", 0.5)

        # Smoothing
        self.declare_parameter("marker_hist_len", 8)
        self.declare_parameter("cube_hist_len", 6)
        self.declare_parameter("ema_alpha", 0.25)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.br = TransformBroadcaster(self)

        # Camera intrinsics from CameraInfo
        self.camera_info_received = False
        self.K: Optional[np.ndarray] = None
        self.D: Optional[np.ndarray] = None
        self.K_inv: Optional[np.ndarray] = None

        # Optical -> ROS camera_link basis change for directions:
        # x_ros =  z_opt, y_ros = -x_opt, z_ros = -y_opt
        self.R_cam_opt = np.array(
            [[0.0, 0.0, 1.0],
             [-1.0, 0.0, 0.0],
             [0.0, -1.0, 0.0]],
            dtype=np.float64,
        )

        # Histories
        self.marker_hist_len = int(self.get_parameter("marker_hist_len").value)
        self.marker_hist: Dict[int, deque] = {
            int(mid): deque(maxlen=self.marker_hist_len)
            for mid in self.get_parameter("marker_ids").value
        }

        self.cube_hist_len = int(self.get_parameter("cube_hist_len").value)
        self.cubes = {
            c: {"cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0, "last_ns": 0,
                "hist": deque(maxlen=self.cube_hist_len), "ema": None}
            for c in ["red", "green", "blue"]
        }
        self._last_bbox_seen_ns = 0

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subs
        self.create_subscription(CubeBBoxList, "cubes_position", self._on_boxes, 10)
        self.create_subscription(
            CameraInfo,
            self.get_parameter("camera_info_topic").value,
            self._on_caminfo,
            sensor_qos,
        )

        self.create_timer(1.0 / 20.0, self._tick)

        self.get_logger().info("workspace_projection node running (CameraInfo-based intrinsics)")

    # -----------------------------
    # CameraInfo
    # -----------------------------
    def _on_caminfo(self, msg: CameraInfo):
        if self.camera_info_received:
            return
        try:
            K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            D = np.array(msg.d, dtype=np.float64).ravel()

            # Some drivers publish empty D; handle gracefully
            if D.size == 0:
                D = None

            self.K = K
            self.D = D
            self.K_inv = np.linalg.inv(K)

            self.camera_info_received = True
            self.get_logger().info(
                f"CameraInfo received. Using K from message. D={'None' if D is None else len(D)} coeffs."
            )
        except Exception as e:
            self.get_logger().error(f"Failed to parse CameraInfo: {e}")

    # -----------------------------
    # BBoxes
    # -----------------------------
    def _on_boxes(self, msg: CubeBBoxList):
        now = self.get_clock().now().nanoseconds
        self._last_bbox_seen_ns = now
        for b in msg.cubes:
            if b.id in self.cubes:
                s = self.cubes[b.id]
                s["cx"], s["cy"] = float(b.cx), float(b.cy)
                s["w"], s["h"] = float(b.w), float(b.h)
                s["last_ns"] = now

    # -----------------------------
    # TF utils
    # -----------------------------
    def _can_T(self, target, source) -> bool:
        return self.tf_buffer.can_transform(target, source, rclpy.time.Time())

    def _lookup_RT(self, target, source) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns (R, t) mapping a point from 'source' into 'target':
            p_target = R @ p_source + t
        """
        if not self._can_T(target, source):
            return None
        tf = self.tf_buffer.lookup_transform(target, source, rclpy.time.Time())
        Rm = quat_to_rotmat(
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
            tf.transform.rotation.w,
        )
        t = np.array(
            [tf.transform.translation.x,
             tf.transform.translation.y,
             tf.transform.translation.z],
            dtype=np.float64,
        )
        return Rm, t

    def _broadcast(self, parent: str, child: str, p: np.ndarray, Rm: Optional[np.ndarray] = None):
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = parent
        msg.child_frame_id = child

        msg.transform.translation.x = float(p[0])
        msg.transform.translation.y = float(p[1])
        msg.transform.translation.z = float(p[2])

        if Rm is None:
            q = (0.0, 0.0, 0.0, 1.0)
        else:
            q = rotmat_to_quat(Rm)

        msg.transform.rotation.x = float(q[0])
        msg.transform.rotation.y = float(q[1])
        msg.transform.rotation.z = float(q[2])
        msg.transform.rotation.w = float(q[3])

        self.br.sendTransform(msg)

    # -----------------------------
    # Geometry
    # -----------------------------
    def _pixel_to_ray_cam(self, u: float, v: float) -> np.ndarray:
        """
        Ray direction in CAMERA FRAME (hrs_camera_link, ROS convention).

        If undistort_pixels=True and D is available:
            Use cv2.undistortPoints to get normalized optical coords (x,y),
            then d_opt=[x,y,1].
        Else:
            d_opt = K^-1 [u v 1]

        Then convert optical -> hrs_camera_link:
            d_cam = R_cam_opt * d_opt
        """
        if (not self.camera_info_received) or (self.K is None) or (self.K_inv is None):
            # fallback safe direction
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)

        use_undist = bool(self.get_parameter("undistort_pixels").value) and (self.D is not None)

        if use_undist:
            # undistortPoints returns normalized image coordinates if P is None
            pts = np.array([[[u, v]]], dtype=np.float64)
            und = cv2.undistortPoints(pts, self.K, self.D, P=None)
            x = float(und[0, 0, 0])
            y = float(und[0, 0, 1])
            d_opt = np.array([x, y, 1.0], dtype=np.float64)
        else:
            d_opt = self.K_inv @ np.array([u, v, 1.0], dtype=np.float64)

        n = np.linalg.norm(d_opt)
        d_opt = d_opt / n if n > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=np.float64)

        d_cam = self.R_cam_opt @ d_opt
        n2 = np.linalg.norm(d_cam)
        return d_cam / n2 if n2 > 1e-12 else np.array([1.0, 0.0, 0.0], dtype=np.float64)

    @staticmethod
    def _fit_plane(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        c = points.mean(axis=0)
        _, _, Vt = np.linalg.svd(points - c)
        n = Vt[-1]
        nn = np.linalg.norm(n)
        n = n / nn if nn > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if n[2] < 0:
            n = -n
        return c, n

    @staticmethod
    def _ray_plane(o: np.ndarray, d: np.ndarray, p0: np.ndarray, n: np.ndarray) -> Optional[np.ndarray]:
        denom = float(np.dot(n, d))
        if abs(denom) < 1e-9:
            return None
        t = float(np.dot(n, (p0 - o)) / denom)
        return (o + t * d) if t > 0 else None

    @staticmethod
    def _plane_frame_from_normal(n: np.ndarray) -> np.ndarray:
        z = n / (np.linalg.norm(n) + 1e-12)
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(ref, z)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x = np.cross(ref, z)
        x /= (np.linalg.norm(x) + 1e-12)
        y = np.cross(z, x)
        y /= (np.linalg.norm(y) + 1e-12)
        return np.column_stack([x, y, z])

    @staticmethod
    def _cube_center_from_front_bottom_edge(hit: np.ndarray, cam_origin: np.ndarray, n: np.ndarray, cube_size: float) -> np.ndarray:
        v_to_cam = cam_origin - hit
        v_plane = v_to_cam - n * float(np.dot(v_to_cam, n))
        norm = float(np.linalg.norm(v_plane))
        if norm > 1e-9:
            dir_away = -v_plane / norm
        else:
            dir_away = np.zeros(3, dtype=np.float64)

        p_bottom_center = hit + dir_away * (cube_size / 2.0)
        p_center = p_bottom_center + n * (cube_size / 2.0)
        return p_center

    # -----------------------------
    # World frame selection
    # -----------------------------
    def _choose_world_frame(self, cam_frame: str) -> Optional[str]:
        preferred: List[str] = list(self.get_parameter("preferred_world_frames").value)
        for w in preferred:
            if self._can_T(w, cam_frame):
                return w
        return None

    # -----------------------------
    # Main tick
    # -----------------------------
    def _tick(self):
        if not self.camera_info_received:
            self.get_logger().warn("Waiting for CameraInfo (K,D)...", throttle_duration_sec=2.0)
            return

        cam = self.get_parameter("camera_frame").value

        world = self._choose_world_frame(cam)
        if world is None:
            self.get_logger().warn(
                f"Missing TF from any of {self.get_parameter('preferred_world_frames').value} -> {cam}.",
                throttle_duration_sec=2.0,
            )
            return

        # world <- cam (target=world, source=cam)
        Twc = self._lookup_RT(world, cam)
        if Twc is None:
            self.get_logger().warn(f"Missing TF {world} <- {cam}", throttle_duration_sec=2.0)
            return

        R_wc, t_wc = Twc

        # keep your workaround
        if bool(self.get_parameter("invert_world_cam_tf").value):
            R_wc, t_wc = invert_rt(R_wc, t_wc)

        cam_origin_world = t_wc

        # 1) Plane from markers in world (direct TF world <- aruco_<id>)
        marker_prefix = self.get_parameter("marker_prefix").value
        marker_ids = [int(x) for x in self.get_parameter("marker_ids").value]

        pts = []
        for mid in marker_ids:
            marker_frame = f"{marker_prefix}{mid}"
            Twm = self._lookup_RT(world, marker_frame)
            if Twm is None:
                continue
            _, p_w = Twm

            self.marker_hist[mid].append(p_w)
            p_med = np.median(np.array(self.marker_hist[mid]), axis=0)
            pts.append(p_med)

        if len(pts) < 3:
            self.get_logger().warn("Not enough markers for plane fit (need >=3).", throttle_duration_sec=2.0)
            return

        pts = np.array(pts, dtype=np.float64)
        p0, n = self._fit_plane(pts)

        plane_frame = self.get_parameter("plane_frame").value
        R_plane = self._plane_frame_from_normal(n)
        self._broadcast(world, plane_frame, p0, R_plane)

        # 2) Project cubes
        now = self.get_clock().now().nanoseconds
        bbox_timeout = float(self.get_parameter("bbox_timeout").value)
        u_anchor = float(self.get_parameter("u_anchor").value)
        v_anchor = float(self.get_parameter("v_anchor").value)
        cube_size = float(self.get_parameter("cube_size").value)
        ema_alpha = float(self.get_parameter("ema_alpha").value)
        cube_prefix = self.get_parameter("cube_prefix").value

        any_cube = False
        for color, s in self.cubes.items():
            if (now - int(s["last_ns"])) > int(bbox_timeout * 1e9):
                continue

            u = float(s["cx"]) + (u_anchor - 0.5) * float(s["w"])
            v = float(s["cy"]) + v_anchor * float(s["h"])

            d_cam = self._pixel_to_ray_cam(u, v)
            d_w = R_wc @ d_cam

            hit = self._ray_plane(cam_origin_world, d_w, p0, n)
            if hit is None:
                continue

            p_cube = self._cube_center_from_front_bottom_edge(hit, cam_origin_world, n, cube_size)

            s["hist"].append(p_cube)
            p_med = np.median(np.array(s["hist"]), axis=0)

            if s["ema"] is None:
                s["ema"] = p_med
            else:
                a = ema_alpha
                s["ema"] = (1.0 - a) * s["ema"] + a * p_med

            self._broadcast(world, cube_prefix + color, s["ema"], None)
            any_cube = True

        if not any_cube:
            if (now - int(self._last_bbox_seen_ns)) > int(1.0 * 1e9):
                self.get_logger().warn("No cube TF: no recent cubes_position.", throttle_duration_sec=2.0)
            else:
                self.get_logger().warn("No cube TF: ray-plane miss.", throttle_duration_sec=2.0)


def main():
    rclpy.init()
    node = WorkspaceProjectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


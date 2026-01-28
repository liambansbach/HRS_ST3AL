#!/usr/bin/env python3
"""
GrabCubeNode (head-aware camera frame, with history reset on head motion)

Key fix:
- When head moves, marker/cube median history makes plane/cubes "stick".
- We detect camera pose change and CLEAR histories so plane/cubes follow head immediately.

Pipeline:
1) Build base->hrs_camera_link from base->head_tilt_link:
   - translation: head_tilt position + (optional) offset rotated by head
   - rotation: head_tilt rotation * R_offset
2) Publish hrs_camera_optical as well (for debugging + clean ray math)
3) Convert ArUco marker positions (camera_link -> aruco_marker_i) into base_link and rebroadcast as hrs_aruco_marker_*
4) Fit workspace plane in base_link and broadcast hrs_workspace_plane (z=normal)
5) Project cube pixels (from cubes_position) onto plane and broadcast hrs_cube_{color}

Camera intrinsics:
- detect_cubes uses 'camera_image/undistorted' => use K_eff, D_eff from calibration.yaml
"""

from pathlib import Path
from collections import deque
import numpy as np
import yaml
import rclpy
from rclpy.node import Node

from ainex_interfaces.msg import CubeBBoxList
from tf2_ros import TransformListener, Buffer, TransformBroadcaster
from geometry_msgs.msg import TransformStamped


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


def rpy_to_rotmat(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp],
                   [0, 1, 0],
                   [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0],
                   [sy, cy, 0],
                   [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def rot_angle_deg(R1, R2):
    """Angle between rotations (degrees)."""
    R = R1.T @ R2
    # clamp trace
    tr = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    ang = np.arccos(tr)
    return float(np.degrees(ang))


# -----------------------------
# Node
# -----------------------------
class ProjectCubesOntoWorkspaceNode(Node):
    def __init__(self):
        super().__init__("project_cubes_onto_workspace")

        # ---- frames ----
        self.base_frame = "base_link"

        # where the aruco node publishes marker transforms from:
        self.aruco_camera_frame = "camera_link"
        self.marker_frames = [f"aruco_marker_{i}" for i in range(4)]

        # head kinematic frame (moves with pan/tilt chain)
        self.head_frame = "head_tilt_link"

        # our debug camera frames (moves with head)
        self.hrs_camera_frame = "hrs_camera_link"
        self.hrs_camera_optical_frame = "hrs_camera_optical"
        self.publish_hrs_camera = True

        # rebroadcast markers in base (should change with camera/head motion while the original aruco_marker_* stay fixed in camera_link)
        self.marker_rebroadcast_prefix = "hrs_aruco_marker_"
        self.publish_marker_rebroadcast = True

        # plane frame (calculated plane through aruco markers)
        self.plane_frame = "hrs_workspace_plane"
        self.publish_plane_frame = True

        # cube frames (center of every cube volume)
        self.cube_tf_prefix = "hrs_cube_"

        # ---- camera extrinsics relative to head_tilt_link ----
        CAMERA_OFFSET_XYZ = np.array([0.0, 0.019, 0.016], dtype=np.float64)

        # optional rotation offset (usually 0 unless axes mismatch)
        CAMERA_OFFSET_RPY = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        self.R_hc = rpy_to_rotmat(*CAMERA_OFFSET_RPY)
        self.t_hc = CAMERA_OFFSET_XYZ

        # ---- IMPORTANT: reset histories on head motion ----
        # Tune these:
        self.reset_rot_deg = 2.0     # if camera rotates more than this -> reset histories
        self.reset_trans_m = 0.01    # if camera translates more than this -> reset histories

        self._last_R_bc = None
        self._last_t_bc = None

        # ---- cube params ----
        self.cube_size = 0.05 # cube size in meters. Relevant for projecting to plane
        self.bbox_timeout = 1.0  # seconds before we consider a bbox stale

        # --- pixel anchor tuning ---
        # 0.50 = bottom edge of bbox (cy + 0.5*h)
        # try 0.55 .. 0.70 if depth is too far away (because bbox includes top face)
        self.u_anchor = 0.50   # 0.5=center horizontally
        self.v_anchor = 3.30   # global default (0.5=bottom edge), tune per setup :: Higher = cube projection is closer to robot than real :: Lower = cube projection is further from robot than real
        

        # ---- smoothing ----
        self.marker_hist_len = 10   # reduced (was 15)
        self.cube_hist_len = 6      # reduced (was 10)
        self.ema_alpha = 0.2

        # ---- calibration ----
        self.calib_path = Path.cwd() / "src/vision/config/calibration.yaml"
        self.camera_matrix_name = "K_eff"
        self.camera_distortion_name = "D_eff"
        self.K, self.D = self._load_calibration(self.calib_path)
        self.K_inv = np.linalg.inv(self.K)

        # ---- TF ----
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_br = TransformBroadcaster(self)

        # ---- states ----
        self.markers = {
            m: {"hist": deque(maxlen=self.marker_hist_len), "p_med": None}
            for m in self.marker_frames
        }
        self.cubes = {
            c: {
                "cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0,
                "last_ns": 0,
                "hist": deque(maxlen=self.cube_hist_len),
                "ema": None
            } for c in ["red", "green", "blue"]
        }

        # REP-103 optical <-> camera_link rotation
        # optical: x right, y down, z forward
        # link:    x forward, y left, z up
        # d_link = R_link_opt * d_opt
        self.R_link_opt = np.array(
            [[0.0, 0.0, 1.0],
             [-1.0, 0.0, 0.0],
             [0.0, -1.0, 0.0]],
            dtype=np.float64,
        )
        # d_opt = R_opt_link * d_link
        self.R_opt_link = self.R_link_opt.T

        # ---- ROS ----
        self.create_subscription(CubeBBoxList, "cubes_position", self._on_boxes, 10) # from detect_cubes node
        self.create_timer(1.0 / 20.0, self._tick) # 20 Hz 

        self.get_logger().info("GrabCubeNode running (with history reset on head motion)")

    # -----------------------------
    # Calibration
    # -----------------------------
    def _load_calibration(self, path: Path):
        if not path.exists():
            self.get_logger().error(f"Calibration missing at {path}, using identity.")
            return np.eye(3, dtype=np.float64), np.zeros((5,), dtype=np.float64)

        with path.open() as f:
            data = yaml.safe_load(f)

        if self.camera_matrix_name in data:
            K = np.array(data[self.camera_matrix_name], dtype=np.float64).reshape(3, 3)
        else:
            self.get_logger().warn("No K_eff found in YAML, using identity.")
            K = np.eye(3, dtype=np.float64)

        if self.camera_distortion_name in data:
            D = np.array(data[self.camera_distortion_name], dtype=np.float64).reshape(-1)
        else:
            D = np.zeros((5,), dtype=np.float64)

        self.get_logger().info(f"Loaded K ({self.camera_matrix_name}):\n{K}")
        self.get_logger().info(f"Loaded D ({self.camera_distortion_name}): {D.tolist()}")
        return K, D

    # -----------------------------
    # Sub callback
    # -----------------------------
    def _on_boxes(self, msg: CubeBBoxList):
        now = self.get_clock().now().nanoseconds
        for b in msg.cubes:
            if b.id in self.cubes:
                s = self.cubes[b.id]
                s["cx"], s["cy"] = float(b.cx), float(b.cy)
                s["w"], s["h"] = float(b.w), float(b.h)
                s["last_ns"] = now

    # -----------------------------
    # TF helpers
    # -----------------------------
    def _can_T(self, parent, child) -> bool:
        return self.tf_buffer.can_transform(parent, child, rclpy.time.Time())

    def _get_T(self, parent, child):
        if not self._can_T(parent, child):
            return None
        tf = self.tf_buffer.lookup_transform(parent, child, rclpy.time.Time())
        R = quat_to_rotmat(
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
        return R, t

    def _broadcast_tf(self, child, p, q_xyzw=(0.0, 0.0, 0.0, 1.0), parent=None):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent if parent is not None else self.base_frame
        t.child_frame_id = child
        t.transform.translation.x = float(p[0])
        t.transform.translation.y = float(p[1])
        t.transform.translation.z = float(p[2])
        t.transform.rotation.x = float(q_xyzw[0])
        t.transform.rotation.y = float(q_xyzw[1])
        t.transform.rotation.z = float(q_xyzw[2])
        t.transform.rotation.w = float(q_xyzw[3])
        self.tf_br.sendTransform(t)

    # -----------------------------
    # Reset histories if camera pose changed
    # -----------------------------
    def _maybe_reset_on_camera_motion(self, R_bc, t_bc):
        if self._last_R_bc is None:
            self._last_R_bc = R_bc.copy()
            self._last_t_bc = t_bc.copy()
            return

        d_rot = rot_angle_deg(self._last_R_bc, R_bc)
        d_trans = float(np.linalg.norm(self._last_t_bc - t_bc))

        if d_rot > self.reset_rot_deg or d_trans > self.reset_trans_m:
            # Reset marker history
            for st in self.markers.values():
                st["hist"].clear()
                st["p_med"] = None
            # Reset cube history/ema
            for s in self.cubes.values():
                s["hist"].clear()
                s["ema"] = None

            self.get_logger().info(
                f"Head/camera moved (Δrot={d_rot:.2f}deg, Δtrans={d_trans:.3f}m) -> reset histories"
            )

        self._last_R_bc = R_bc.copy()
        self._last_t_bc = t_bc.copy()

    # -----------------------------
    # Base -> hrs_camera_link (head driven)
    # -----------------------------
    def _get_base_to_hrs_camera(self):
        Tb_h = self._get_T(self.base_frame, self.head_frame)
        if Tb_h is None:
            return None

        R_bh, t_bh = Tb_h

        # camera translation: head position + rotated offset
        t_bc = t_bh + (R_bh @ self.t_hc)

        # camera rotation: head rotation * optional offset
        R_bc = R_bh @ self.R_hc

        # Detect motion and reset histories if needed
        self._maybe_reset_on_camera_motion(R_bc, t_bc)

        if self.publish_hrs_camera:
            q = rotmat_to_quat(R_bc)
            self._broadcast_tf(self.hrs_camera_frame, t_bc, q_xyzw=q, parent=self.base_frame)

            # also publish optical frame (same translation, rotated)
            R_bo = R_bc @ self.R_opt_link
            qopt = rotmat_to_quat(R_bo)
            self._broadcast_tf(self.hrs_camera_optical_frame, t_bc, q_xyzw=qopt, parent=self.base_frame)

        return R_bc, t_bc

    # -----------------------------
    # Geometry
    # -----------------------------
    def _pixel_to_ray_optical(self, u, v):
        d = self.K_inv @ np.array([u, v, 1.0], dtype=np.float64)
        n = np.linalg.norm(d)
        if n < 1e-12:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return d / n

    @staticmethod
    def _fit_plane(points):
        c = points.mean(axis=0)
        _, _, Vt = np.linalg.svd(points - c)
        n = Vt[-1]
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            n = n / nn
        if n[2] < 0:
            n = -n
        return c, n

    @staticmethod
    def _ray_plane(o, d, p0, n):
        denom = float(np.dot(n, d))
        if abs(denom) < 1e-9:
            return None
        t = float(np.dot(n, (p0 - o)) / denom)
        return (o + t * d) if t > 0 else None

    @staticmethod
    def _plane_frame_from_normal(n):
        z = n / (np.linalg.norm(n) + 1e-12)
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(ref, z)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x = np.cross(ref, z)
        x /= (np.linalg.norm(x) + 1e-12)
        y = np.cross(z, x)
        y /= (np.linalg.norm(y) + 1e-12)
        return np.column_stack([x, y, z])

    # -----------------------------
    # Marker positions in base using hrs_camera pose
    # -----------------------------
    def _marker_pos_in_base(self, marker_frame, R_bc, t_bc):
        Tc_m = self._get_T(self.aruco_camera_frame, marker_frame)
        if Tc_m is None:
            Tb_m = self._get_T(self.base_frame, marker_frame)
            return None if Tb_m is None else Tb_m[1]
        _, p_c = Tc_m
        return R_bc @ p_c + t_bc

    # -----------------------------
    # Main loop
    # -----------------------------
    def _tick(self):
        Tb_c = self._get_base_to_hrs_camera()
        if Tb_c is None:
            self.get_logger().warn(
                f"Missing TF {self.base_frame}->{self.head_frame}. (joint_states / robot_state_publisher?)",
                throttle_duration_sec=2.0,
            )
            return

        R_bc, o = Tb_c  # camera_link origin in base

        # 1) markers -> base -> plane
        got_any = False
        for i, m in enumerate(self.marker_frames):
            p_b = self._marker_pos_in_base(m, R_bc, o)
            if p_b is None:
                continue

            got_any = True
            st = self.markers[m]
            st["hist"].append(p_b)
            st["p_med"] = np.median(st["hist"], axis=0)

            if self.publish_marker_rebroadcast:
                self._broadcast_tf(self.marker_rebroadcast_prefix + str(i), st["p_med"])

        pts = [s["p_med"] for s in self.markers.values() if s["p_med"] is not None]
        if len(pts) < 3:
            if not got_any:
                self.get_logger().warn(
                    f"No aruco markers found under {self.aruco_camera_frame}->aruco_marker_*",
                    throttle_duration_sec=2.0,
                )
            else:
                self.get_logger().warn("Not enough marker points for plane (need >=3).", throttle_duration_sec=2.0)
            return

        p0, n = self._fit_plane(np.array(pts, dtype=np.float64))

        if self.publish_plane_frame:
            R_plane = self._plane_frame_from_normal(n)
            q_plane = rotmat_to_quat(R_plane)
            self._broadcast_tf(self.plane_frame, p0, q_xyzw=q_plane)

        # 2) cubes: pixel ray -> plane
        now = self.get_clock().now().nanoseconds
        any_cube = False

        for color, s in self.cubes.items():
            if (now - int(s["last_ns"])) > int(self.bbox_timeout * 1e9):
                continue

            # u = float(s["cx"])
            # v = float(s["cy"]) + 0.5 * float(s["h"])  # bottom-center (change to cy if needed)

            u = float(s["cx"]) + (self.u_anchor - 0.5) * float(s["w"])
            v = float(s["cy"]) + self.v_anchor * float(s["h"])


            # ray in optical
            d_opt = self._pixel_to_ray_optical(u, v)

            # optical -> camera_link
            d_cam = self.R_link_opt @ d_opt

            # camera_link -> base
            d_base = R_bc @ d_cam

            hit = self._ray_plane(o, d_base, p0, n)
            if hit is None:
                continue

            p = hit + n * (self.cube_size / 2.0)

            s["hist"].append(p)
            p_med = np.median(np.array(s["hist"]), axis=0)
            if s["ema"] is None:
                s["ema"] = p_med
            else:
                a = float(self.ema_alpha)
                s["ema"] = (1.0 - a) * s["ema"] + a * p_med

            self._broadcast_tf(self.cube_tf_prefix + color, s["ema"])
            any_cube = True

        if not any_cube:
            self.get_logger().warn(
                "No cube TF published. Check cubes_position / bbox_timeout / (cy+h/2) convention.",
                throttle_duration_sec=2.0,
            )


def main():
    rclpy.init()
    node = ProjectCubesOntoWorkspaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

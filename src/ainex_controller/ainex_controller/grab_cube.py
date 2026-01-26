#!/usr/bin/env python3
"""
GrabCubeNode

- Subscribes:  cubes_position (CubeBBoxList)
- Reads TFs:   aruco_marker_{0..3} in base_link
- Reads TF:    base_link -> camera_optical_frame (preferred) OR base_link -> camera_link
- Computes:    workspace plane from ArUco markers
- Projects:    cube pixel rays onto plane
- Publishes:   TF frames hrs_cube_red / green / blue (in base_link)

IMPORTANT:
- Camera intrinsics K correspond to the *optical* camera frame (REP-103):
    optical: x right, y down, z forward
  If TF provides only camera_link, we convert optical ray -> camera_link using a fixed rotation.
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


def quat_to_rotmat(x, y, z, w):
    n = x*x + y*y + z*z + w*w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1-(y*y+z*z)*s, (x*y-z*w)*s, (x*z+y*w)*s],
        [(x*y+z*w)*s, 1-(x*x+z*z)*s, (y*z-x*w)*s],
        [(x*z-y*w)*s, (y*z+x*w)*s, 1-(x*x+y*y)*s]
    ], dtype=np.float64)


class GrabCubeNode(Node):
    def __init__(self):
        super().__init__("grab_cube")

        # ---- frames ----
        self.base_frame = "base_link"

        # Prefer optical frames (common names). We auto-pick the first that exists in TF.
        self.camera_optical_candidates = [
            "camera_optical_frame",
            "camera_color_optical_frame",
            "camera_rgb_optical_frame",
            "camera_link_optical",
        ]

        self.camera_matrix_name = "K_eff"

        # Fallback if no optical exists
        self.camera_link_frame = "camera_link"

        self.marker_frames = [f"aruco_marker_{i}" for i in range(4)]

        self.cube_tf_prefix = "hrs_cube_"
        self.plane_tf_frame = "hrs_workspace_plane"

        # ---- cube params ----
        self.cube_size = 0.05      # meters
        self.bbox_timeout = 1.0    # seconds (etwas großzügiger)

        # ---- smoothing ----
        self.marker_hist_len = 15
        self.cube_hist_len = 10
        self.ema_alpha = 0.35

        # ---- camera calibration ----
        self.K = self._load_camera_matrix()
        self.K_inv = np.linalg.inv(self.K)

        # ---- TF ----
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_br = TransformBroadcaster(self)

        # ---- marker state ----
        self.markers = {
            m: {"hist": deque(maxlen=self.marker_hist_len), "p_med": None}
            for m in self.marker_frames
        }

        # ---- cube state ----
        self.cubes = {
            c: {
                "cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0,
                "last_ns": 0,
                "hist": deque(maxlen=self.cube_hist_len),
                "ema": None
            } for c in ["red", "green", "blue"]
        }

        # ---- REP103: optical -> camera_link rotation ----
        # camera_link: x forward, y left, z up
        # optical:     x right,   y down, z forward
        # link_x = z_opt
        # link_y = -x_opt
        # link_z = -y_opt
        self.R_link_opt = np.array([
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ], dtype=np.float64)

        # ---- subs / timers ----
        self.create_subscription(CubeBBoxList, "cubes_position", self._on_boxes, 10)
        self.create_timer(1.0 / 20.0, self._tick)

        self.get_logger().info("GrabCubeNode running")

    # -----------------------------
    # Calibration
    # -----------------------------
    def _load_camera_matrix(self):
        calib = Path.cwd() / "src/vision/config/calibration.yaml"
        if not calib.exists():
            self.get_logger().error("Calibration missing, using identity")
            return np.eye(3, dtype=np.float64)

        with calib.open() as f:
            data = yaml.safe_load(f)
        K = np.array(data[self.camera_matrix_name], dtype=np.float64).reshape(3, 3)
        self.get_logger().info(f"Loaded camera {self.camera_matrix_name}:\n{K}")
        return K

    # -----------------------------
    # Sub callback
    # -----------------------------
    def _on_boxes(self, msg: CubeBBoxList):
        now = self.get_clock().now().nanoseconds
        for b in msg.cubes:
            if b.id in self.cubes:
                s = self.cubes[b.id]
                s["cx"], s["cy"] = float(b.cx), float(b.cy)
                s["w"], s["h"] = float(b.w) * 1, float(b.h) * 1
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
            tf.transform.rotation.w
        )
        t = np.array([
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z
        ], dtype=np.float64)
        return R, t

    def _broadcast_tf_point(self, child, p):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.base_frame
        t.child_frame_id = child
        t.transform.translation.x = float(p[0])
        t.transform.translation.y = float(p[1])
        t.transform.translation.z = float(p[2])
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_br.sendTransform(t)

    # -----------------------------
    # Geometry
    # -----------------------------
    def _pixel_to_ray_optical(self, u, v):
        """Pixel (u,v) -> unit ray in OPTICAL frame."""
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
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            n = n / n_norm
        # orient "up" (assumes base z up)
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

    # -----------------------------
    # Camera frame selection
    # -----------------------------
    def _pick_camera_frame(self):
        """Return tuple (mode, frame_name).
        mode = 'optical' if an optical frame exists else 'link' (camera_link fallback).
        """
        for f in self.camera_optical_candidates:
            if self._can_T(self.base_frame, f):
                return ("optical", f)
        if self._can_T(self.base_frame, self.camera_link_frame):
            return ("link", self.camera_link_frame)
        return (None, None)

    # -----------------------------
    # Main loop
    # -----------------------------
    def _tick(self):
        # 1) update markers
        for m in self.marker_frames:
            T = self._get_T(self.base_frame, m)
            if T:
                self.markers[m]["hist"].append(T[1])
                self.markers[m]["p_med"] = np.median(self.markers[m]["hist"], axis=0)

        pts = [m["p_med"] for m in self.markers.values() if m["p_med"] is not None]
        if len(pts) < 3:
            self.get_logger().warn("Not enough marker points for plane (need >=3).", throttle_duration_sec=2.0)
            return

        p0, n = self._fit_plane(np.array(pts, dtype=np.float64))
        self._broadcast_tf_point(self.plane_tf_frame, p0)

        # 2) choose camera frame
        mode, cam_frame = self._pick_camera_frame()
        if cam_frame is None:
            self.get_logger().warn("No TF base_link -> camera frame found (optical/camera_link).", throttle_duration_sec=2.0)
            return

        Tc = self._get_T(self.base_frame, cam_frame)
        if not Tc:
            return
        R_bcam, o = Tc  # camera origin in base

        now = self.get_clock().now().nanoseconds
        any_cube = False

        # 3) per cube: intersection + lift
        for color, s in self.cubes.items():
            if (now - int(s["last_ns"])) > int(self.bbox_timeout * 1e9):
                continue

            # bottom-center pixel
            u = float(s["cx"])
            v = float(s["cy"]) + 0.5 * float(s["h"])

            d_opt = self._pixel_to_ray_optical(u, v)

            # convert ray to the chosen cam_frame
            if mode == "optical":
                d_cam = d_opt
            else:
                # mode == "link": TF gives camera_link, but ray is optical -> convert
                d_cam = self.R_link_opt @ d_opt

            d_base = R_bcam @ d_cam

            hit = self._ray_plane(o, d_base, p0, n)
            if hit is None:
                continue

            # cube center (lift by half cube height)
            p = hit + n * (self.cube_size / 2.0)

            # smooth (median + EMA)
            s["hist"].append(p)
            p_med = np.median(np.array(s["hist"]), axis=0)
            if s["ema"] is None:
                s["ema"] = p_med
            else:
                a = float(self.ema_alpha)
                s["ema"] = (1.0 - a) * s["ema"] + a * p_med

            # broadcast cube TF
            self._broadcast_tf_point(self.cube_tf_prefix + color, s["ema"])
            any_cube = True

        if not any_cube:
            self.get_logger().warn(
                f"No cube TF published. (camera_mode={mode}, cam_frame={cam_frame}) "
                f"Check: cubes_position arriving? bbox_timeout={self.bbox_timeout}s? optical frame mismatch?",
                throttle_duration_sec=2.0
            )


def main():
    rclpy.init()
    node = GrabCubeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

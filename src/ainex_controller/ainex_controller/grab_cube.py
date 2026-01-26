#!/usr/bin/env python3

from pathlib import Path
import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from tf2_ros import TransformListener, Buffer


class GrabCubeNode(Node):
    def __init__(self) -> None:
        super().__init__("grab_cube")
        self.cwd = Path.cwd() # current working directory

        self.run_once_flag = True # flag to run init only once

        # Camera calibration parameters
        self.camera_k = None # camera intrinsic matrix
        self.camera_d = None # camera distortion coefficients
        self.camera_width = None # camera image width
        self.camera_height = None # camera image height

        # Load calibration file parameter
        self.vision_pkg_path = Path.joinpath(self.cwd, "src/vision")
        self.camera_calibration_yaml_path = Path.joinpath(self.vision_pkg_path, "config/calibration.yaml")
        self._load_calibration()

        # Get detected aruco markers as TF-Transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.parent_frame = "camera_link"
        self.marker_frames = [f"aruco_marker_{i}" for i in range(4)]

        # Cache: last Transform per Marker + last Timestamp (sec,nsec)
        self.markers = {
            name: {
                "tf": None,              # geometry_msgs/TransformStamped
                "stamp_ns": -1,          # int nanoseconds
                "seen": False,           # ob jemals gesehen
                "last_update_wall_ns": None,  # optional: wann wir es zuletzt aktualisiert haben (now())
            }
            for name in self.marker_frames
        }

        # refreshrate for checking the TFs (Hz)
        self.timer = self.create_timer(1 / 10, self.timer_callback)


    def _load_calibration(self) -> None:
        calib_path = self.camera_calibration_yaml_path
        if not calib_path.is_file():
            self.get_logger().error(f"Calibration file not found: {calib_path}")
            return

        with calib_path.open("r") as f:
            calib = yaml.safe_load(f)

        self.camera_k = np.array(calib["K_eff"], dtype=np.float32).reshape(3, 3)
        self.camera_d = np.array(calib["D_eff"], dtype=np.float32).ravel()
        self.camera_width = int(calib["image_width"])
        self.camera_height = int(calib["image_height"])

        self.get_logger().info("Loaded camera calibration.")
    

    @staticmethod
    def _stamp_to_ns(stamp) -> int:
        # stamp: builtin_interfaces/msg/Time
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def timer_callback(self) -> None:
        updated_any = False

        for child_frame in self.marker_frames:
            # Optional: can_transform avoid Spam-Exceptions
            if not self.tf_buffer.can_transform(self.parent_frame, child_frame, rclpy.time.Time()):
                continue

            try:
                tf = self.tf_buffer.lookup_transform(
                    self.parent_frame,
                    child_frame,
                    rclpy.time.Time()  # latest
                )
            except Exception:
                continue

            stamp_ns = self._stamp_to_ns(tf.header.stamp)
            prev_ns = self.markers[child_frame]["stamp_ns"]

            # NUR updaten, wenn wirklich neuer Messwert (neuer Stamp)
            if stamp_ns > prev_ns:
                self.markers[child_frame]["tf"] = tf
                self.markers[child_frame]["stamp_ns"] = stamp_ns
                self.markers[child_frame]["seen"] = True
                self.markers[child_frame]["last_update_wall_ns"] = self.get_clock().now().nanoseconds
                updated_any = True

        # Debug-Ausgabe: nur wenn sich etwas geändert hat (oder alle X Sekunden)
        if updated_any:
            self._log_marker_states()

    def _log_marker_states(self):
        lines = []
        for name in self.marker_frames:
            entry = self.markers[name]
            if entry["tf"] is None:
                lines.append(f"{name}: (no data yet)")
                continue

            t = entry["tf"].transform.translation
            q = entry["tf"].transform.rotation
            age_ms = None
            if entry["last_update_wall_ns"] is not None:
                age_ms = (self.get_clock().now().nanoseconds - entry["last_update_wall_ns"]) / 1e6

            if age_ms is None:
                lines.append(f"{name}: t=({t.x:.3f},{t.y:.3f},{t.z:.3f})")
            else:
                lines.append(
                    f"{name}: t=({t.x:.3f},{t.y:.3f},{t.z:.3f}) age={age_ms:.0f}ms"
                )

        self.get_logger().info(" | ".join(lines))


    def display_loop(self) -> None:
        while rclpy.ok():
            if self.run_once_flag:
                print("K Matrix", self.camera_k)
                print("D Coefficients", self.camera_d)
                print("Image Width", self.camera_width)
                print("Image Height", self.camera_height)

                self.run_once_flag = False
            

            #if not self.process_key():
            #    break

            rclpy.spin_once(self, timeout_sec=0.01)

        cv2.destroyAllWindows()


def main() -> None:

    rclpy.init()
    node = GrabCubeNode()
    node.get_logger().info("GrabCubeNode node started")

    try:
        # NOT Visualize Display
        # rclpy.spin(node)
        # Visualize Display
        node.display_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

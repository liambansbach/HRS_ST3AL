#!/usr/bin/env python3
'''
Docstring for camera.camera.mp_pose
'''

from pathlib import Path
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from geometry_msgs.msg import Point
from pose_interfaces.msg import UpperbodyPose

import traceback


class MPPose(Node):
    def __init__(self):
        super().__init__('mp_pose')

        self.bridge = CvBridge()
        self.frame = None

        self._next_run_ns = 0
        self._run_period_ns = int(1e9 / 30)  # 30 Hz
        self._cb_count = 0

        '''
        RIG JOINT NAMES:
        - left shoulder == 11
        - right shoulder == 12
        - left elbow == 13
        - right elbow == 14
        - left wrist == 15
        - right wrist == 16
        - left pinky == 17
        - right pinky == 18
        - left index == 19
        - right index == 20
        '''

        self.POSE_CONNECTIONS = [
            (18, 20), (18, 16), (20, 16), (16, 14), (14, 12),
            (12, 11),
            (11, 13), (13, 15), (15, 19), (15, 17), (17, 19)
        ]

        # Path to Landmark model
        project_root = Path(__file__).resolve().parents[3]
        model_path = project_root / "models" / "pose_landmarker_lite.task"

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,  # CHANGED: VIDEO -> IMAGE
            num_poses=1
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

        cv2.namedWindow("POSE", cv2.WINDOW_NORMAL)
        cv2.startWindowThread()  

        if not model_path.exists():
            self.get_logger().error(f"Pose model not found: {model_path}")
        else:
            self.get_logger().info(f"Using pose model: {model_path}")

        self.camera_sub = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )

        self.rig_pub = self.create_publisher(
            UpperbodyPose,
            "/mp_pose/upper_body_rig",
            10
        )

        self.get_logger().info('Pose Recognition Node started! Listening to /image_raw/')

    def image_callback(self, msg: Image):
        cv2.waitKey(1)

        now_ns = time.monotonic_ns()
        if now_ns < self._next_run_ns:
            cv2.waitKey(1)
            return
        self._next_run_ns = now_ns + self._run_period_ns

        # lightweight heartbeat so we know callback runs
        self._cb_count += 1
        if self._cb_count % 60 == 0:
            self.get_logger().info("image_callback alive")

        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = np.ascontiguousarray(frame_rgb)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            result = self.detector.detect(mp_image)

            pose_landmarks_list = getattr(result, "pose_landmarks", None)
            if pose_landmarks_list:
                lms = pose_landmarks_list[0]  # num_poses=1

                def to_point(lm):
                    return Point(x=float(lm.x), y=float(lm.y), z=float(lm.z))

                def vis(lm):
                    return float(getattr(lm, "visibility", 1.0))

                rig = UpperbodyPose()
                rig.header.stamp = msg.header.stamp
                rig.header.frame_id = msg.header.frame_id if msg.header.frame_id else "camera"

                # Left: shoulder 11, elbow 13, wrist 15, pinky 17, index 19
                rig.left_shoulder = to_point(lms[11]); rig.left_shoulder_vis = vis(lms[11])
                rig.left_elbow    = to_point(lms[13]); rig.left_elbow_vis    = vis(lms[13])
                rig.left_wrist    = to_point(lms[15]); rig.left_wrist_vis    = vis(lms[15])
                rig.left_pinky    = to_point(lms[17]); rig.left_pinky_vis    = vis(lms[17])
                rig.left_index    = to_point(lms[19]); rig.left_index_vis    = vis(lms[19])

                # Right: shoulder 12, elbow 14, wrist 16, pinky 18, index 20
                rig.right_shoulder = to_point(lms[12]); rig.right_shoulder_vis = vis(lms[12])
                rig.right_elbow    = to_point(lms[14]); rig.right_elbow_vis    = vis(lms[14])
                rig.right_wrist    = to_point(lms[16]); rig.right_wrist_vis    = vis(lms[16])
                rig.right_pinky    = to_point(lms[18]); rig.right_pinky_vis    = vis(lms[18])
                rig.right_index    = to_point(lms[20]); rig.right_index_vis    = vis(lms[20])

                # Threshold for Shoulders visibility
                if vis(lms[11]) > 0.3 and vis(lms[12]) > 0.3:
                    self.rig_pub.publish(rig)

            annotated_rgb = self.draw_landmarks_on_image(frame_rgb, result)
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

            cv2.imshow("POSE", annotated_bgr)
            cv2.waitKey(1)

        except Exception:
            self.get_logger().error("Pose Pipeline failed!\n" + traceback.format_exc())
            return

    def draw_landmarks_on_image(self, rgb_image: np.ndarray, result) -> np.ndarray:
        annotated = rgb_image.copy()
        h, w = annotated.shape[:2]

        pose_landmarks_list = getattr(result, "pose_landmarks", None)
        if not pose_landmarks_list:
            return annotated

        for pose_landmarks in pose_landmarks_list:
            pts = []
            for lm in pose_landmarks:
                x = int(np.clip(lm.x * w, 0, w - 1))
                y = int(np.clip(lm.y * h, 0, h - 1))
                v = float(getattr(lm, "visibility", 1.0))
                pts.append((x, y, v))

            for a, b in self.POSE_CONNECTIONS:
                xa, ya, va = pts[a]
                xb, yb, vb = pts[b]
                if va < 0.5 or vb < 0.5:
                    continue
                cv2.line(annotated, (xa, ya), (xb, yb), (255, 255, 255), 2)

            for x, y, v in pts:
                if v < 0.5:
                    continue
                cv2.circle(annotated, (x, y), 3, (0, 255, 0), -1)

        return annotated


def main(args=None):
    rclpy.init(args=args)
    node = MPPose()
    node.get_logger().info('Pose Estimation running')

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

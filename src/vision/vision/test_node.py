#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from pose_interfaces.msg import UpperbodyPose


class UpperBodyListener(Node):
    def __init__(self):
        super().__init__('upperbody_listener')

        self.latest = None

        self.sub = self.create_subscription(
            UpperbodyPose,
            '/mp_pose/upper_body_rig',
            self.cb,
            10
        )

        # Print every 2 seconds
        self.timer = self.create_timer(2.0, self.print_latest)

        self.get_logger().info("Listening on /mp_pose/upper_body_rig ...")

    def cb(self, msg: UpperbodyPose):
        self.latest = msg

    def print_latest(self):
        if self.latest is None:
            self.get_logger().info("No messages received yet.")
            return

        m = self.latest
        # One-line-ish readable print
        self.get_logger().info(
            f"stamp={m.header.stamp.sec}.{m.header.stamp.nanosec} frame_id='{m.header.frame_id}' | "
            f"L_sh=({m.left_shoulder.x:.3f},{m.left_shoulder.y:.3f},{m.left_shoulder.z:.3f}) v={m.left_shoulder_vis:.2f} | "
            f"L_el=({m.left_elbow.x:.3f},{m.left_elbow.y:.3f},{m.left_elbow.z:.3f}) v={m.left_elbow_vis:.2f} | "
            f"L_wr=({m.left_wrist.x:.3f},{m.left_wrist.y:.3f},{m.left_wrist.z:.3f}) v={m.left_wrist_vis:.2f} | "
            f"R_sh=({m.right_shoulder.x:.3f},{m.right_shoulder.y:.3f},{m.right_shoulder.z:.3f}) v={m.right_shoulder_vis:.2f} | "
            f"R_el=({m.right_elbow.x:.3f},{m.right_elbow.y:.3f},{m.right_elbow.z:.3f}) v={m.right_elbow_vis:.2f} | "
            f"R_wr=({m.right_wrist.x:.3f},{m.right_wrist.y:.3f},{m.right_wrist.z:.3f}) v={m.right_wrist_vis:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = UpperBodyListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

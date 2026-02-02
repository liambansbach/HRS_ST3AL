#!/usr/bin/env python3
"""
Subscribes to /workflow_cmd and logs received single-key commands.

Use this to sanity-check that the keyboard publisher is working and that other
nodes can receive the commands.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class WorkflowCmdMonitor(Node):
    def __init__(self):
        super().__init__("workflow_cmd_monitor")
        self.sub = self.create_subscription(String, "workflow_cmd", self.on_cmd, 10)
        self.get_logger().info("Listening on /workflow_cmd ...")

    def on_cmd(self, msg: String) -> None:
        """Log every received workflow command."""
        self.get_logger().info(f"Received command: '{msg.data}'")


def main(args=None):
    rclpy.init(args=args)
    node = WorkflowCmdMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

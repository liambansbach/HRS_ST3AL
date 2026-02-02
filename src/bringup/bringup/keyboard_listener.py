#!/usr/bin/env python3
"""
Keyboard command publisher for the complete workflow.

Publishes single-key commands on /workflow_cmd:
  i = go idle
  r = start recording
  z = execute recorded sequence
  t = NOT IMPLEMENTED
  p = pose estimation (future)
  q = quit this keyboard node

This node only publishes commands; it does not interpret system state.
"""

import sys
import select
import termios
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class KeyboardPublisher(Node):
    def __init__(self):
        super().__init__("keyboard_publisher")

        self.publisher = self.create_publisher(String, "workflow_cmd", 10)

        # Timer drives polling so we don't block the executor
        self.timer = self.create_timer(0.02, self.poll_keyboard)

        self.allowed = {"i", "r", "z", "t", "p", "q"}

        self.stdin_fd = sys.stdin.fileno()
        self.old_term_settings = termios.tcgetattr(self.stdin_fd)
        tty.setcbreak(self.stdin_fd)

        self.get_logger().info(
            "KeyboardPublisher ready. Keys: i r z t p (q quits). Publishing on /workflow_cmd"
        )

    def poll_keyboard(self) -> None:
        """Poll stdin for a single key press and publish it as a String command."""
        if not self.key_available():
            return

        ch = sys.stdin.read(1)
        if not ch:
            return

        ch = ch.strip().lower()
        if ch not in self.allowed:
            return

        msg = String()
        msg.data = ch
        self.publisher.publish(msg)

        if ch == "q":
            self.get_logger().info("Received 'q' -> shutting down keyboard node.")
            rclpy.shutdown()
            return

        self.get_logger().info(f"Sent command: '{ch}'")

    def key_available(self) -> bool:
        """Return True if a key is waiting on stdin (non-blocking)."""
        rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
        return bool(rlist)

    def destroy_node(self):
        """Restore terminal settings before node teardown."""
        try:
            termios.tcsetattr(self.stdin_fd, termios.TCSADRAIN, self.old_term_settings)
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardPublisher()
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

#!/usr/bin/env python3
"""
Workflow controller: subscribes to /workflow_cmd and calls RecordDemo action.

Keys:
  r: start recording (RecordDemo goal) if IDLE
  i: force IDLE (cancels recording if active)
  z: placeholder for execution stage (later)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String
from ainex_interfaces.action import RecordDemo


class State(Enum):
    IDLE = "IDLE"
    RECORDING = "RECORDING"
    WAIT_EXEC = "WAIT_EXEC"
    EXECUTING = "EXECUTING"


@dataclass
class RecordingData:
    session_id: int
    events: list
    world_states: list


class WorkflowController(Node):
    def __init__(self):
        super().__init__("workflow_controller")

        # --- Parameters you can override via launch/CLI if you want ---
        self.declare_parameter("record_action_name", "record_demo")
        self.declare_parameter("min_world_states", 3)
        self.declare_parameter("idle_timeout_sec", 8.0)

        self.record_action_name = (
            self.get_parameter("record_action_name").get_parameter_value().string_value
        )
        self.default_min_world_states = (
            self.get_parameter("min_world_states").get_parameter_value().integer_value
        )
        self.default_idle_timeout_sec = (
            self.get_parameter("idle_timeout_sec").get_parameter_value().double_value
        )

        # --- State ---
        self.state = State.IDLE
        self.last_recording: Optional[RecordingData] = None

        # --- ROS comms ---
        self.cb_group = ReentrantCallbackGroup()
        self.cmd_sub = self.create_subscription(
            String, "workflow_cmd", self.on_cmd, 10, callback_group=self.cb_group
        )

        self.record_client = ActionClient(
            self, RecordDemo, self.record_action_name, callback_group=self.cb_group
        )

        # Track current action goal
        self.record_goal_handle = None
        self.record_result_future = None
        self.cancel_future = None

        # For log throttling
        self.last_feedback_log_time = self.get_clock().now()

        self.get_logger().info(
            f"WorkflowController ready. Subscribing to /workflow_cmd. RecordDemo action='{self.record_action_name}'."
        )
        self.print_state("Startup")

    def print_state(self, reason: str) -> None:
        """Log current FSM state."""
        self.get_logger().info(f"[{reason}] State = {self.state.value}")

    def set_state(self, new_state: State, reason: str) -> None:
        """Transition FSM state with logging."""
        if new_state != self.state:
            self.state = new_state
            self.print_state(reason)

    # ----------------------- Command Handling -----------------------

    def on_cmd(self, msg: String) -> None:
        """Handle incoming keyboard commands from /workflow_cmd."""
        cmd = (msg.data or "").strip().lower()
        if not cmd:
            return

        if cmd == "i":
            self.handle_force_idle()
            return

        if cmd == "r":
            self.handle_start_recording()
            return

        if cmd == "z":
            if self.state != State.WAIT_EXEC:
                self.get_logger().warn(f"Ignoring 'z' because state is {self.state.value} (need WAIT_EXEC).")
                return
            self.get_logger().info("TODO: execution stage will be added next (Step 5).")
            self.set_state(State.EXECUTING, "Command 'z' (placeholder)")
            self.set_state(State.IDLE, "Execution placeholder done")
            return

        if cmd in ("t", "p"):
            self.get_logger().info(f"Command '{cmd}' received (currently unused placeholder).")
            return

        self.get_logger().warn(f"Unknown command '{cmd}'")

    def handle_force_idle(self) -> None:
        """Force state back to IDLE; cancel recording if currently active."""
        if self.state == State.RECORDING and self.record_goal_handle is not None:
            self.get_logger().info("Force IDLE: canceling active RecordDemo goal ...")
            self.cancel_future = self.record_goal_handle.cancel_goal_async()
            self.cancel_future.add_done_callback(self.on_cancel_done)
            # We’ll transition to IDLE in on_cancel_done
            return

        self.set_state(State.IDLE, "Command 'i' (force idle)")

    def handle_start_recording(self) -> None:
        """Send a RecordDemo goal if currently IDLE."""
        if self.state != State.IDLE:
            self.get_logger().warn(f"Ignoring 'r' because state is {self.state.value} (need IDLE).")
            return

        # Ensure server is up
        if not self.record_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error(
                f"RecordDemo action server '{self.record_action_name}' not available."
            )
            return

        goal = RecordDemo.Goal()
        goal.min_world_states = int(self.default_min_world_states)
        goal.idle_timeout_sec = float(self.default_idle_timeout_sec)

        self.get_logger().info(
            f"Sending RecordDemo goal: min_world_states={goal.min_world_states}, idle_timeout_sec={goal.idle_timeout_sec}"
        )

        send_future = self.record_client.send_goal_async(goal, feedback_callback=self.on_record_feedback)
        send_future.add_done_callback(self.on_goal_response)

        self.set_state(State.RECORDING, "Command 'r' (start recording)")

    # ----------------------- Action Callbacks -----------------------

    def on_goal_response(self, future) -> None:
        """Handle goal acceptance/rejection."""
        goal_handle = future.result()
        if goal_handle is None:
            self.get_logger().error("RecordDemo goal response: None (unexpected).")
            self.set_state(State.IDLE, "Goal response error")
            return

        if not goal_handle.accepted:
            self.get_logger().warn("RecordDemo goal was REJECTED by server.")
            self.set_state(State.IDLE, "Goal rejected")
            return

        self.record_goal_handle = goal_handle
        self.get_logger().info("RecordDemo goal ACCEPTED. Recording...")

        self.record_result_future = goal_handle.get_result_async()
        self.record_result_future.add_done_callback(self.on_record_result)

    def on_record_feedback(self, feedback_msg) -> None:
        """Print feedback during recording (throttled)."""
        fb = feedback_msg.feedback

        # Throttle to ~2 Hz
        now = self.get_clock().now()
        dt = (now - self.last_feedback_log_time).nanoseconds / 1e9
        if dt < 0.5:
            return
        self.last_feedback_log_time = now

        try:
            self.get_logger().info(
                f"[feedback] status={fb.status} events={fb.num_events} world_states={fb.num_world_states}"
            )
        except Exception:
            # If feedback fields differ, we at least avoid crashing
            self.get_logger().info("[feedback] received")

    def on_record_result(self, future) -> None:
        """Handle action completion and store events/world_states."""
        result_wrap = future.result()
        if result_wrap is None:
            self.get_logger().error("RecordDemo result: None (unexpected).")
            self.cleanup_recording()
            self.set_state(State.IDLE, "Result error")
            return

        status = result_wrap.status
        result = result_wrap.result

        # Store the data even if empty (helps debugging)
        self.last_recording = RecordingData(
            session_id=int(result.session_id),
            events=list(result.events),
            world_states=list(result.world_states),
        )

        self.get_logger().info(
            f"RecordDemo finished (status={status}). "
            f"session_id={self.last_recording.session_id}, "
            f"events={len(self.last_recording.events)}, "
            f"world_states={len(self.last_recording.world_states)}"
        )

        self.cleanup_recording()
        self.set_state(State.WAIT_EXEC, "Recording successful (press 'z' to execute)")

    def on_cancel_done(self, future) -> None:
        """Handle cancel completion."""
        try:
            cancel_response = future.result()
            # cancel_response.goals_canceling exists in rclpy, but we don't rely on exact structure
            self.get_logger().info("Cancel request sent to RecordDemo server.")
        except Exception as ex:
            self.get_logger().warn(f"Cancel handling exception: {ex}")

        self.cleanup_recording()
        self.set_state(State.IDLE, "Canceled recording (force idle)")

    def cleanup_recording(self) -> None:
        """Clear active goal tracking."""
        self.record_goal_handle = None
        self.record_result_future = None
        self.cancel_future = None


def main(args=None):
    rclpy.init(args=args)
    node = WorkflowController()

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

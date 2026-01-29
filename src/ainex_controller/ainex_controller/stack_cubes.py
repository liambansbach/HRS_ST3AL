#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

import numpy as np
import pinocchio as pin
from dataclasses import dataclass
from typing import List, Optional


from ainex_controller.ainex_model import AiNexModel
from ainex_controller.ainex_robot import AinexRobot
from ainex_controller.ainex_hand_controller import HandController


@dataclass
class HandAction:
    hand: str                      # "left" or "right"
    rel_or_abs: str                # "rel" or "abs"
    target_translation: np.ndarray  # shape (3,)
    duration: float = 3.0
    wait_after: float = 0.0        # seconds to wait after motion ends


class StackCubesNode(Node):
    def __init__(self):
        super().__init__("stack_cubes")

        self.dt = 0.05  # 20 Hz

        # --- Robot model / controller setup
        pkg = get_package_share_directory("ainex_description")
        urdf_path = pkg + "/urdf/ainex.urdf"

        self.robot_model = AiNexModel(self, urdf_path)
        self.ainex_robot = AinexRobot(self, self.robot_model, self.dt, sim=True)

        self.left_hand_controller = HandController(self, self.robot_model, arm_side="left")
        self.right_hand_controller = HandController(self, self.robot_model, arm_side="right")

        # Arm DOFs for zero commands
        self.left_dof = len(self.robot_model.get_arm_ids("left"))
        self.right_dof = len(self.robot_model.get_arm_ids("right"))
        self.zero_left = np.zeros(self.left_dof)
        self.zero_right = np.zeros(self.right_dof)

        # --- Initial pose
        self.init_robot_pose = {
            "head_tilt": 0, "head_pan": 0,
            "r_gripper": 0, "l_gripper": 0,
            "r_el_yaw": 1.58, "l_el_yaw": -1.58,
            "r_el_pitch": 0, "l_el_pitch": 0,
            "r_sho_roll": 1.4, "l_sho_roll": -1.4,
            "r_sho_pitch": 0, "l_sho_pitch": 0,
            "r_hip_yaw": 0, "l_hip_yaw": 0,
            "r_hip_roll": 0, "l_hip_roll": 0,
            "r_hip_pitch": 0, "l_hip_pitch": 0,
            "r_knee": 0, "l_knee": 0,
            "r_ank_pitch": 0, "l_ank_pitch": 0,
            "r_ank_roll": 0, "l_ank_roll": 0,
        }

        # --- Sequencer data
        self.sequence: List[HandAction] = []
        self.seq_index: int = 0
        self.action_started: bool = False
        self.action_start_time: Optional[float] = None

        # --- Mode/phase control
        # phases: "init" -> "idle" <-> "run_sequence"
        self.phase: str = "init"
        self.autorun: bool = True        # if True: start automatically when sequence becomes available
        self.clear_sequence_on_finish = True  # convenient default

        # one-time init flag
        self.did_init = False

        # timer-driven control loop
        ASSERT_DT = self.dt
        self.timer = self.create_timer(self.dt, self.control_loop)

        # Optional: create an initial test sequence (you can remove this later)
        self.enqueue_sequence(self.build_test_sequence())

    # -----------------------------
    # Time helpers
    # -----------------------------
    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def action_elapsed(self) -> float:
        if self.action_start_time is None:
            return 0.0
        return self.now_s() - self.action_start_time

    # -----------------------------
    # Robot helpers
    # -----------------------------
    def move_to_initial_position(self):
        q_init = np.zeros(self.robot_model.model.nq)
        for joint_name, val in self.init_robot_pose.items():
            q_init[self.robot_model.get_joint_id(joint_name)] = float(val)
        self.ainex_robot.move_to_initial_position(q_init)

    def make_target_se3(self, translation_xyz: np.ndarray) -> pin.SE3:
        T = pin.SE3.Identity()
        T.translation = np.array(translation_xyz, dtype=float).reshape(3)
        return T

    # -----------------------------
    # Sequence API (call these later from other code)
    # -----------------------------
    def enqueue_sequence(self, actions: List[HandAction], replace: bool = False):
        """Add a sequence of actions. If replace=True, overwrites any pending actions."""
        if replace:
            self.sequence = []
            self.seq_index = 0
            self.action_started = False
            self.action_start_time = None

        if actions:
            self.sequence.extend(actions)
            self.get_logger().info(f"Enqueued {len(actions)} actions (queue size now {len(self.sequence) - self.seq_index}).")

        # If we're idle and autorun is enabled, start automatically next tick
        # (we don't flip phase here to keep callback logic simple + deterministic)

    def clear_sequence(self):
        self.sequence = []
        self.seq_index = 0
        self.action_started = False
        self.action_start_time = None

    def has_pending_actions(self) -> bool:
        return self.seq_index < len(self.sequence)

    # -----------------------------
    # Action execution
    # -----------------------------
    def start_action(self, action: HandAction):
        T = self.make_target_se3(action.target_translation)

        if action.hand == "left":
            self.left_hand_controller.set_target_pose(T, duration=action.duration, type=action.rel_or_abs)
        elif action.hand == "right":
            self.right_hand_controller.set_target_pose(T, duration=action.duration, type=action.rel_or_abs)
        else:
            self.get_logger().error(f"Unknown hand '{action.hand}' (must be 'left' or 'right')")

        self.action_started = True
        self.action_start_time = self.now_s()

        self.get_logger().info(
            f"Start action {self.seq_index+1}/{len(self.sequence)}: "
            f"{action.hand} {action.rel_or_abs} trans={action.target_translation.tolist()} "
            f"dur={action.duration}s wait={action.wait_after}s"
        )

    def finish_sequence(self):
        self.get_logger().info("Sequence finished -> going idle.")
        if self.clear_sequence_on_finish:
            self.clear_sequence()
        else:
            # keep history, just set index to end
            self.seq_index = len(self.sequence)
            self.action_started = False
            self.action_start_time = None
        self.phase = "idle"

    # -----------------------------
    # Example sequence builder
    # -----------------------------
    def build_test_sequence(self) -> List[HandAction]:
        return [
            HandAction(hand="left",  rel_or_abs="rel", target_translation=np.array([0.0,  0.10, 0.0]), duration=3.0, wait_after=5.0),
            HandAction(hand="right", rel_or_abs="abs", target_translation=np.array([0.0, -0.15, 0.0]), duration=3.0, wait_after=5.0),
            HandAction(hand="right", rel_or_abs="rel", target_translation=np.array([0.0,  0.00, 0.1]), duration=2.0, wait_after=2.0),
        ]

    # -----------------------------
    # Main loop (timer)
    # -----------------------------
    def control_loop(self):
        # Always compute commands each tick; by default: hold position (zero velocities)
        v_left = self.zero_left
        v_right = self.zero_right

        # ---- Init phase
        if not self.did_init:
            self.move_to_initial_position()
            self.did_init = True
            self.phase = "idle"
            self.get_logger().info("Init done -> idle.")
            # Still update robot once with zeros (hold)
            self.ainex_robot.update(v_left, v_right, self.dt)
            return

        # ---- Idle: do nothing unless we have pending actions and autorun enabled
        if self.phase == "idle":
            if self.autorun and self.has_pending_actions():
                self.phase = "run_sequence"
                self.get_logger().info("Idle -> run_sequence (pending actions detected).")

        # ---- Run sequence
        if self.phase == "run_sequence":
            if not self.has_pending_actions():
                self.finish_sequence()
            else:
                current = self.sequence[self.seq_index]

                if not self.action_started:
                    self.start_action(current)

                # Only update the active controller -> sequential behavior
                if current.hand == "left":
                    v_left = self.left_hand_controller.update(self.dt)
                else:
                    v_right = self.right_hand_controller.update(self.dt)

                # Time-based completion
                t = self.action_elapsed()
                if t >= (current.duration + current.wait_after):
                    self.seq_index += 1
                    self.action_started = False
                    self.action_start_time = None

                    if not self.has_pending_actions():
                        self.finish_sequence()

        # Apply commands
        self.ainex_robot.update(v_left, v_right, self.dt)


def main():
    rclpy.init()
    node = StackCubesNode()
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

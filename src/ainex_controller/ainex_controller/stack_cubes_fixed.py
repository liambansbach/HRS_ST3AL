#!/usr/bin/env python3
"""
StackCubesNode (ROS 2)

Purpose
-------
Executes a cube stacking task on the AiNex robot using TF-based cube positions and a simple
step sequencer. The stacking plan is generated from a recorded manipulation event sequence
(PICK/PLACE events) and executed with Cartesian hand controllers.

Inputs
------
- RecordDemo.Result on `workflow/record_result`
    Contains recorded ManipulationEvent list (cube_id + type).
- std_msgs/Empty on `workflow/execute`
    Trigger to start execution of the stored recording.
- std_msgs/Empty on `workflow/abort`
    Abort trigger to stop execution immediately.
- TF frames:
    * base_frame <- hrs_cube_<color> (cube center positions in base frame)

Outputs / Effects
-----------------
- Sends velocity commands via AinexRobot.update() using left/right HandController instances.
- Opens/closes grippers through AinexRobot open_hand/close_hand (if available).

Main structure
--------------
- Timer (dt): control_loop() implements a small state machine:
    init -> idle -> run_sequence -> idle
- Another timer keeps cube positions updated from TF (20 Hz).
- build_stacking_steps_from_events() converts events into a list of Step objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import pinocchio as pin
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from scipy.spatial.transform import Rotation as R

from tf2_ros import TransformListener, Buffer

from ainex_interfaces.action import RecordDemo
from ainex_interfaces.msg import ManipulationEvent

from ainex_controller.ainex_model import AiNexModel
from ainex_controller.ainex_robot import AinexRobot
from ainex_controller.ainex_hand_controller_extended import HandController as HandControllerExtended

from std_msgs.msg import Empty


@dataclass
class Step:
    """
    Single sequencer step.

    Attributes
    ----------
    kind:
        "move" or "grip"
    duration:
        Command duration (s)
    wait_after:
        Extra wait time after duration (s)

    For move steps:
      - hand: "left" or "right"
      - rel_or_abs: controller mode ("abs" or "rel")
      - target_translation: 3D target translation (meters), in base frame

    For grip steps:
      - grip_cmd: "open" or "close"
      - grip_which: "left" / "right" / "both" (passed to AinexRobot)
    """
    kind: str
    duration: float = 2.0
    wait_after: float = 0.0

    # move
    hand: Optional[str] = None
    rel_or_abs: str = "abs"
    target_translation: Optional[np.ndarray] = None

    # grip
    grip_cmd: Optional[str] = None
    grip_which: str = "both"


class StackCubesNode(Node):
    """
    ROS2 node that stacks cubes according to recorded manipulation events.

    The node waits for a RecordDemo.Result, then for an execute trigger. Once TF cube frames
    are available, it builds and runs a step sequence consisting of:
      - Move above pick -> move to pick -> close -> lift
      - Move above stack -> place -> open -> retreat
    """

    def __init__(self):
        """Initialize robot, controllers, TF helpers, subscriptions, and sequencer state."""
        super().__init__("stack_cubes")

        self.dt = 0.05  # control rate (20 Hz)

        # -----------------------------
        # Parameters / configuration
        # -----------------------------
        self.use_test_sequence: bool = False
        self.sim: bool = False

        # Name of the base/world frame where everything is transformed to.
        self.base_frame = "base_link"

        # Cube TF prefix (frames: hrs_cube_red, hrs_cube_green, hrs_cube_blue)
        self.cube_tf_prefix: str = "hrs_cube_"

        # Velocity commands (4 DoF per arm in your setup)
        self.v_left = np.zeros(4, dtype=np.float64)
        self.v_right = np.zeros(4, dtype=np.float64)

        # Cube side length (meters)
        self.cube_size = 0.035

        # -----------------------------
        # Subscription to server results
        # -----------------------------
        self.record_sub = self.create_subscription(
            RecordDemo.Result,
            "workflow/record_result",
            self.on_record_result,
            10,
        )

        self.have_recording = False
        self.execute_requested = False

        self.exec_sub = self.create_subscription(Empty, "workflow/execute", self._on_execute, 10)
        self.abort_sub = self.create_subscription(Empty, "workflow/abort", self._on_abort, 10)

        # -----------------------------
        # Robot model / controller setup
        # -----------------------------
        pkg = get_package_share_directory("ainex_description")
        urdf_path = pkg + "/urdf/ainex.urdf"

        self.robot_model = AiNexModel(self, urdf_path)
        self.ainex_robot = AinexRobot(self, self.robot_model, self.dt, sim=self.sim)

        # Use the extended controller (nullspace + damping options)
        self.left_hand_controller = HandControllerExtended(
            self,
            self.robot_model,
            arm_side="left",
            enable_nullspace=True,
            k_null=0.7,
            adaptive_damping=True,
            hard_stop_on_singularity=False,
        )

        self.right_hand_controller = HandControllerExtended(
            self,
            self.robot_model,
            arm_side="right",
            enable_nullspace=True,
            k_null=0.7,
            adaptive_damping=True,
            hard_stop_on_singularity=False,
        )

        # -----------------------------
        # Initial pose
        # -----------------------------
        self.init_robot_pose = {
            "head_tilt": -0.78, "head_pan": 0,
            "r_gripper": 0, "l_gripper": 0,
            "r_el_yaw": 0.8, "l_el_yaw": -0.8,
            "r_el_pitch": -1.57, "l_el_pitch": -1.57,
            "r_sho_roll": 0.6, "l_sho_roll": -0.6,
            "r_sho_pitch": 1.4, "l_sho_pitch": 1.4,
            "r_hip_yaw": 0, "l_hip_yaw": 0,
            "r_hip_roll": 0, "l_hip_roll": 0,
            "r_hip_pitch": 0, "l_hip_pitch": 0,
            "r_knee": 0, "l_knee": 0,
            "r_ank_pitch": 0, "l_ank_pitch": 0,
            "r_ank_roll": 0, "l_ank_roll": 0,
        }

        # -----------------------------
        # TF
        # -----------------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Cube positions stored in base_frame
        self.cube_pos: Dict[str, Optional[np.ndarray]] = {"red": None, "green": None, "blue": None}
        self.create_timer(1.0 / 20.0, self._update_cube_positions)

        # -----------------------------
        # Sequencer state
        # -----------------------------
        self.sequence: List[Step] = []
        self.seq_index: int = 0
        self.step_started: bool = False
        self.step_start_time: Optional[float] = None

        self.phase: str = "init"
        self.autorun: bool = True
        self.clear_sequence_on_finish = True
        self.did_init = False

        self.pending_events: List[ManipulationEvent] = []

        # Main control loop timer
        self.timer = self.create_timer(self.dt, self.control_loop)

        # -----------------------------
        # Startup behavior
        # -----------------------------
        if self.use_test_sequence:
            self.pending_events = self.build_test_server_events()
            self.have_recording = True
            self.get_logger().info("TEST: stored fake recording; press 'z' to execute.")
        else:
            self.get_logger().info("Waiting for workflow/record_result and workflow/execute.")

    # -----------------------------
    # Time helpers
    # -----------------------------
    def now_s(self) -> float:
        """
        Current ROS time in seconds.

        Returns:
            Time in seconds as float.
        """
        return self.get_clock().now().nanoseconds * 1e-9

    def step_elapsed(self) -> float:
        """
        Elapsed time since the current step started.

        Returns:
            Elapsed time in seconds (0 if no step active).
        """
        if self.step_start_time is None:
            return 0.0
        return self.now_s() - self.step_start_time

    # -----------------------------
    # Robot helpers
    # -----------------------------
    def move_to_initial_position(self):
        """
        Move robot to the predefined initial joint configuration.

        Returns:
            None
        """
        q_init = np.zeros(self.robot_model.model.nq)
        for joint_name, val in self.init_robot_pose.items():
            q_init[self.robot_model.get_joint_id(joint_name)] = float(val)
        self.ainex_robot.move_to_initial_position(q_init)

    # -----------------------------
    # TF helpers
    # -----------------------------
    def _can_T(self, target: str, source: str) -> bool:
        """
        Check whether TF transform target <- source is available.

        Args:
            target: Target frame.
            source: Source frame.

        Returns:
            True if transform is available.
        """
        try:
            return self.tf_buffer.can_transform(target, source, rclpy.time.Time())
        except Exception:
            return False

    def _lookup_RT(self, target: str, source: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Lookup TF and return (R, t) mapping a point from 'source' into 'target':
            p_target = R @ p_source + t

        Args:
            target: Target frame.
            source: Source frame.

        Returns:
            (R, t) if available, else None.
        """
        if not self._can_T(target, source):
            return None
        tf = self.tf_buffer.lookup_transform(target, source, rclpy.time.Time())

        quat = [
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
            tf.transform.rotation.w,
        ]
        Rm = R.from_quat(quat).as_matrix()

        t = np.array(
            [
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z,
            ],
            dtype=np.float64,
        )
        return Rm, t

    def _update_cube_positions(self):
        """
        Periodically update cube positions from TF:
          base_frame <- hrs_cube_<color>

        Returns:
            None
        """
        for c in ["red", "green", "blue"]:
            child_frame = self.cube_tf_prefix + c
            Tbc = self._lookup_RT(self.base_frame, child_frame)  # base <- cube
            if Tbc is None:
                self.cube_pos[c] = None
                continue

            _, p_base = Tbc
            self.cube_pos[c] = p_base

    def get_cube_position(self, color: str) -> Optional[np.ndarray]:
        """
        Get last known cube position (in base frame).

        Args:
            color: Cube id ("red"|"green"|"blue").

        Returns:
            3D position as np.ndarray or None.
        """
        return self.cube_pos.get(color, None)

    # -----------------------------
    # "wait until TFs exist" helpers
    # -----------------------------
    def extract_cube_order(self, events: List[ManipulationEvent]) -> List[str]:
        """
        Extract cube order from events (first appearance of each cube_id).

        Args:
            events: List of ManipulationEvent from the server.

        Returns:
            Ordered list of cube ids (subset of ["red","green","blue"]).
        """
        order: List[str] = []
        for e in events:
            cube_id = str(e.cube_id)
            if cube_id in ["red", "green", "blue"] and cube_id not in order:
                order.append(cube_id)
        print(order)
        return order

    def cube_tfs_ready(self, cube_ids: List[str]) -> bool:
        """
        Check whether TFs for all required cubes exist.

        Args:
            cube_ids: Cube ids.

        Returns:
            True if all base_frame <- hrs_cube_<id> transforms exist.
        """
        base = self.base_frame
        for cid in cube_ids:
            frame = self.cube_tf_prefix + cid
            if not self._can_T(base, frame):
                return False
        return True

    def missing_cube_tfs(self, cube_ids: List[str]) -> List[str]:
        """
        Create a list of missing TF edges for user-facing warnings.

        Args:
            cube_ids: Cube ids.

        Returns:
            List of strings like "base_link->hrs_cube_red".
        """
        base = self.base_frame or "<no_base>"
        missing = []
        for cid in cube_ids:
            frame = self.cube_tf_prefix + cid
            if base == "<no_base>" or not self._can_T(base, frame):
                missing.append(f"{base}->{frame}")
        return missing

    # -----------------------------
    # Sequence helpers
    # -----------------------------
    def clear_sequence(self):
        """Clear current sequence state (steps + indices + timing)."""
        self.sequence = []
        self.seq_index = 0
        self.step_started = False
        self.step_start_time = None

    def has_pending_steps(self) -> bool:
        """
        Returns:
            True if there are remaining steps to execute.
        """
        return self.seq_index < len(self.sequence)

    def start_step(self, step: Step):
        """
        Start executing a single step.

        For grip steps:
          - calls AinexRobot open_hand/close_hand (if available)

        For move steps:
          - sets controller target pose based on target_translation

        Args:
            step: Step to start.

        Returns:
            None
        """
        # -------------------------
        # Gripper step
        # -------------------------
        if step.kind == "grip":
            if hasattr(self.ainex_robot, "open_hand") and hasattr(self.ainex_robot, "close_hand"):
                if step.grip_cmd == "open":
                    self.ainex_robot.open_hand(which=step.grip_which, duration=step.duration)
                elif step.grip_cmd == "close":
                    self.ainex_robot.close_hand(which=step.grip_which, duration=step.duration)
                else:
                    self.get_logger().warn(f"Unknown grip_cmd='{step.grip_cmd}'")
            else:
                self.get_logger().warn("AinexRobot has no open_hand/close_hand; skipping grip step.")

            self.step_started = True
            self.step_start_time = self.now_s()
            return

        # -------------------------
        # Move step
        # -------------------------
        assert step.target_translation is not None
        assert step.hand in ["left", "right"]

        T = pin.SE3.Identity()
        T.translation = np.array(step.target_translation)

        if step.hand == "left":
            self.left_hand_controller.set_target_pose(T, duration=step.duration, type=step.rel_or_abs)
        else:
            self.right_hand_controller.set_target_pose(T, duration=step.duration, type=step.rel_or_abs)

        self.step_started = True
        self.step_start_time = self.now_s()

    def finish_sequence(self):
        """
        Finish the sequence, reset state, and return robot to initial pose.

        Returns:
            None
        """
        self.get_logger().info("Stacking sequence finished -> going idle.")
        if self.clear_sequence_on_finish:
            self.clear_sequence()
        else:
            self.seq_index = len(self.sequence)
            self.step_started = False
            self.step_start_time = None

        self.phase = "idle"
        self.move_to_initial_position()

    # -----------------------------
    # TEST: fake server events
    # -----------------------------
    def build_test_server_events(self) -> List[ManipulationEvent]:
        """
        Build a simple fake event list for offline testing.

        Returns:
            List of ManipulationEvent.
        """
        def ev(ev_type: str, cube_id: str) -> ManipulationEvent:
            e = ManipulationEvent()
            e.type = ev_type
            e.cube_id = cube_id
            e.stamp = self.get_clock().now().to_msg()
            return e

        return [
            ev("PICK", "red"), ev("PLACE", "red"),
            ev("PICK", "blue"), ev("PLACE", "blue"),
            ev("PICK", "green"), ev("PLACE", "green"),
        ]

    # -----------------------------
    # Stacking planner: events -> steps
    # -----------------------------
    def choose_arm_for_cube(self, p: np.ndarray) -> str:
        """
        Choose arm based on cube position.

        Args:
            p: Cube position in base frame (x forward, y left, z up).

        Returns:
            "right" if y < 0 else "left".
        """
        return "right" if float(p[1]) < 0.0 else "left"

    def build_stacking_steps_from_events(self, events: List[ManipulationEvent]) -> List[Step]:
        """
        Convert a recorded event list into an executable stacking sequence.

        Notes:
        - The first cube in 'order' is treated as the base cube.
        - Subsequent cubes are stacked center-on-center using i * cube_size in z.

        Args:
            events: Recorded events.

        Returns:
            List of Step objects (may be empty if TFs missing or not enough cubes).
        """
        order = self.extract_cube_order(events)
        if len(order) < 2:
            self.get_logger().warn(f"Not enough cubes in events to stack: {order}")
            return []

        # Resolve cube positions from TF
        poses: Dict[str, np.ndarray] = {}
        for cube_id in order:
            p = self.get_cube_position(cube_id)
            if p is None:
                self.get_logger().warn(f"Missing TF for cube '{cube_id}'. Cannot build stacking plan.")
                return []
            poses[cube_id] = p.copy()

        base_id = order[0]
        base_p = poses[base_id]

        # Tuning values (kept as-is to preserve behavior)
        z_above_pick = 0.0152
        z_lift = 0.15
        x_offset = -0.075  # due to projection error

        # Place approach clearance (constant above the target)
        place_clear = self.cube_size + 0.017

        # If TF is at cube center: place exactly at stack_p + z_place
        z_place = self.cube_size + 0.014

        steps: List[Step] = []

        # Start stacking from the 2nd cube (index i=1)
        for i, cube_id in enumerate(order[1:], start=1):
            pick_p = poses[cube_id]
            arm = self.choose_arm_for_cube(pick_p)

            # Arm-specific offsets (kept unchanged)
            y_offset = 0.005
            y_offset = y_offset * -1.0 - 0.01 if arm == "right" else y_offset + 0.002

            z_above_pick = z_above_pick * 1.0 if arm == "right" else z_above_pick + 0.0185
            x_offset = x_offset * 1.0 if arm == "right" else x_offset + 0.0036
            z_place = z_place * 1.0 if arm == "right" else z_place - 0.015

            y_offset_placing_left = 0.0 if arm == "right" else 0.008

            # Center-on-center stacking: increase z by i * cube_size
            stack_p = base_p + np.array([0.0, 0.0, i * self.cube_size])

            # -------------------------
            # PICK
            # -------------------------
            steps.append(Step(kind="grip", grip_cmd="open", grip_which=arm, duration=0.5, wait_after=0.2))

            steps.append(
                Step(kind="move", hand=arm, rel_or_abs="abs", target_translation=pick_p + np.array([x_offset - 0.05, y_offset, z_above_pick + 0.1]),
                    duration=2.5,
                    wait_after=0.2,
                )
            )
            steps.append(
                Step(kind="move", hand=arm, rel_or_abs="abs", target_translation=pick_p + np.array([x_offset, y_offset, z_above_pick]),
                    duration=2.5,
                    wait_after=0.2,
                )
            )
            steps.append(Step(kind="grip", grip_cmd="close", grip_which=arm, duration=0.5, wait_after=0.2))
            steps.append(
                Step(kind="move", hand=arm, rel_or_abs="abs", target_translation=pick_p + np.array([x_offset, y_offset, z_lift]),
                    duration=2.0,
                    wait_after=0.2,
                )
            )

            # -------------------------
            # PLACE
            # -------------------------
            # 1) Approach: constant clearance above target
            steps.append(
                Step(kind="move", hand=arm, rel_or_abs="abs", target_translation=stack_p + np.array([x_offset, y_offset, place_clear]),
                    duration=3.0,
                    wait_after=0.2,
                )
            )

            # 2) Down to stack position (cube center)
            steps.append(
                Step(kind="move", hand=arm, rel_or_abs="abs", target_translation=stack_p + np.array([x_offset + 0.0015, y_offset + y_offset_placing_left, z_place]),
                    duration=1.5,
                    wait_after=0.1,
                )
            )

            # 3) Release
            steps.append(Step(kind="grip", grip_cmd="open", grip_which=arm, duration=0.5, wait_after=0.2))

            # 4) Retreat: move out of the stacking area (large y offset)
            steps.append(
                Step(kind="move", hand=arm, rel_or_abs="abs", target_translation=stack_p + np.array([x_offset, y_offset * 7, place_clear * 3]),
                    duration=2.0,
                    wait_after=0.2,
                )
            )

        return steps

    # -----------------------------
    # Server callbacks
    # -----------------------------
    def on_record_result(self, msg: RecordDemo.Result):
        """
        Store events from the recorder result.

        Args:
            msg: RecordDemo.Result containing events.

        Returns:
            None
        """
        self.pending_events = list(msg.events)
        self.have_recording = True
        self.get_logger().info(
            f"Stored recording: session_id={int(msg.session_id)} "
            f"events={len(self.pending_events)} world_states={len(msg.world_states)}"
        )

    def _on_execute(self, _msg: Empty):
        """
        Execute trigger callback.

        Sets a flag; the actual planning/execution happens in control_loop().

        Returns:
            None
        """
        self.execute_requested = True
        self.get_logger().info("Execute trigger received.")

    def _on_abort(self, _msg: Empty):
        """
        Abort callback: stop sequence immediately and go idle.

        Main effects:
          - clears triggers and stored recording state
          - clears current sequence
          - zeroes velocity commands (safety)
          - optionally returns to initial pose (kept enabled)

        Returns:
            None
        """
        self.get_logger().warn("Abort received: stopping sequence and going idle.")

        # Stop planning/execution triggers
        self.execute_requested = False
        self.have_recording = False
        self.pending_events = []

        # Stop the running sequence immediately
        self.clear_sequence()
        self.phase = "idle"

        # Ensure no stale velocity commands are applied
        self.v_left[:] = 0.0
        self.v_right[:] = 0.0

        # Return to initial pose (kept as in your original behavior)
        self.move_to_initial_position()

    # -----------------------------
    # Main loop
    # -----------------------------
    def control_loop(self):
        """
        Main control loop called at dt.

        Steps:
          1) Reset command vectors to zero (prevents drift).
          2) One-time init: move robot to initial pose.
          3) In idle: if execute requested and recording available, build plan when TFs are ready.
          4) In run_sequence: start/update steps, advance when time elapsed.
          5) Send commands to the robot.
        """
        # Always start with zero commands (no drift)
        self.v_left[:] = 0.0
        self.v_right[:] = 0.0

        # Init once
        if not self.did_init:
            self.move_to_initial_position()
            self.did_init = True
            self.phase = "idle"
            self.get_logger().info("Init done -> idle.")
            self.ainex_robot.update(self.v_left, self.v_right, self.dt)
            return

        # -------------------------
        # Idle: build plan if ready
        # -------------------------
        if self.phase == "idle":
            if (
                self.execute_requested
                and self.have_recording
                and self.pending_events
                and not self.has_pending_steps()
            ):
                required = self.extract_cube_order(self.pending_events)

                if not required:
                    self.get_logger().warn("Execute requested but no cubes in events. Staying idle.")
                    # Consume the execute request because there's nothing to do
                    self.execute_requested = False
                    self.have_recording = False

                else:
                    if not self.cube_tfs_ready(required):
                        self.get_logger().warn(
                            f"Waiting for cube TFs... missing: {self.missing_cube_tfs(required)}",
                            throttle_duration_sec=2.0,
                        )
                        # IMPORTANT: do not clear execute_requested here; retry once TFs exist
                    else:
                        steps = self.build_stacking_steps_from_events(self.pending_events)
                        if steps:
                            self.sequence = steps
                            self.seq_index = 0
                            self.step_started = False
                            self.step_start_time = None

                            # Consume execute only when we actually have a plan
                            self.execute_requested = False

                            self.get_logger().info(
                                f"Built stacking plan with {len(self.sequence)} steps. Starting."
                            )
                        else:
                            self.get_logger().warn("TFs ready but could not build plan. Staying idle.")
                            # Consume to avoid retry loop with an empty plan
                            self.execute_requested = False

            if self.autorun and self.has_pending_steps():
                self.phase = "run_sequence"
                self.get_logger().info("Idle -> run_sequence.")

        # -------------------------
        # Run sequence
        # -------------------------
        if self.phase == "run_sequence":
            if not self.has_pending_steps():
                self.finish_sequence()
            else:
                current = self.sequence[self.seq_index]

                if not self.step_started:
                    self.start_step(current)

                # Update controller only for move steps
                if current.kind == "move":
                    if current.hand == "left":
                        self.v_left = self.left_hand_controller.update(self.dt)
                    else:
                        self.v_right = self.right_hand_controller.update(self.dt)

                # Step timing
                t = self.step_elapsed()
                if t >= (current.duration + current.wait_after):
                    self.seq_index += 1
                    self.step_started = False
                    self.step_start_time = None

                    if not self.has_pending_steps():
                        self.finish_sequence()

        # Apply commands
        self.ainex_robot.update(self.v_left, self.v_right, self.dt)


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

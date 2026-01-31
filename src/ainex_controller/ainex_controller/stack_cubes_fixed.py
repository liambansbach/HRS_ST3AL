#!/usr/bin/env python3
from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from ament_index_python.packages import get_package_share_directory

from scipy.spatial.transform import Rotation as R
import numpy as np
import pinocchio as pin
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

from tf2_ros import TransformListener, Buffer

from ainex_interfaces.action import RecordDemo
from ainex_interfaces.msg import ManipulationEvent

from ainex_controller.ainex_model import AiNexModel
from ainex_controller.ainex_robot import AinexRobot
from ainex_controller.ainex_hand_controller import HandController


@dataclass
class Step:
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
    def __init__(self):
        super().__init__("stack_cubes")

        self.dt = 0.05  # 20 Hz

        # -----------------------------
        # Params
        # -----------------------------
        self.use_test_sequence: bool = True
        
        self.sim: bool = False

        # Controllers expect a base frame (your test node implicitly uses this convention)
        self.base_frame = "base_link"

        # cube tf prefix
        self.cube_tf_prefix: str = str("hrs_cube_")

        self.v_left = None
        self.v_right = None

        # Action goal params -> same as on server
        self.min_world_states = 3
        self.idle_timeout_sec = 10.0

        # cube size (meters)
        self.cube_size = 0.04        

        # -----------------------------
        # Robot model / controller setup
        # -----------------------------
        pkg = get_package_share_directory("ainex_description")
        urdf_path = pkg + "/urdf/ainex.urdf"

        self.robot_model = AiNexModel(self, urdf_path)
        self.ainex_robot = AinexRobot(self, self.robot_model, self.dt, sim=self.sim)

        self.left_hand_controller = HandController(self, self.robot_model, arm_side="left")
        self.right_hand_controller = HandController(self, self.robot_model, arm_side="right")

        # -----------------------------
        # Initial pose
        # -----------------------------
        self.init_robot_pose = {
            "head_tilt": -0.65, "head_pan": 0,
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

        # cube positions ALWAYS stored in control_base_active frame
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

        self.ready_to_stack: bool = False
        self.pending_events: List[ManipulationEvent] = []

        self.timer = self.create_timer(self.dt, self.control_loop)

        # -----------------------------
        # ActionClient to server
        # -----------------------------
        self.client = ActionClient(self, RecordDemo, "record_demo")
        self.waiting_for_server_result = False

        # -----------------------------
        # Startup behavior
        # -----------------------------
        if self.use_test_sequence:
            self.pending_events = self.build_test_server_events()
            self.ready_to_stack = True
            self.get_logger().info("TEST: ready_to_stack=True (fake server says: can start stacking).")
        else:
            self.start_recording_and_wait_result()

    # -----------------------------
    # Time helpers
    # -----------------------------
    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def step_elapsed(self) -> float:
        if self.step_start_time is None:
            return 0.0
        return self.now_s() - self.step_start_time

    # -----------------------------
    # Robot helpers
    # -----------------------------
    def move_to_initial_position(self):
        q_init = np.zeros(self.robot_model.model.nq)
        for joint_name, val in self.init_robot_pose.items():
            q_init[self.robot_model.get_joint_id(joint_name)] = float(val)
        self.ainex_robot.move_to_initial_position(q_init)

    # -----------------------------
    # TF helpers
    # -----------------------------
    def _can_T(self, target: str, source: str) -> bool:
        try:
            return self.tf_buffer.can_transform(target, source, rclpy.time.Time())
        except Exception:
            return False

    def _lookup_RT(self, target: str, source: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns (R, t) mapping a point from 'source' into 'target':
            p_target = R @ p_source + t
        """
        if not self._can_T(target, source):
            return None
        tf = self.tf_buffer.lookup_transform(target, source, rclpy.time.Time())

        # Quaternion in [x, y, z, w] format
        quat = [tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w]
        
        Rm = R.from_quat(quat).as_matrix()

        t = np.array(
            [tf.transform.translation.x,
             tf.transform.translation.y,
             tf.transform.translation.z],
            dtype=np.float64,
        )
        return Rm, t

    def _update_cube_positions(self):
        # Directly lookup base <- cube (no intermediate world frame!)
        for c in ["red", "green", "blue"]:
            child_frame = self.cube_tf_prefix + c
            Tbc = self._lookup_RT(self.base_frame, child_frame)  # base <- cube
            if Tbc is None:
                self.cube_pos[c] = None
                continue

            _, p_base = Tbc

            # Optional grasp-point tweak
            self.cube_pos[c] = p_base

    def get_cube_position(self, color: str) -> Optional[np.ndarray]:
        return self.cube_pos.get(color, None)

    # -----------------------------
    # "wait until TFs exist" helpers
    # -----------------------------
    def extract_cube_order(self, events: List[ManipulationEvent]) -> List[str]:
        order: List[str] = []
        for e in events:
            cube_id = str(e.cube_id)
            if cube_id in ["red", "green", "blue"] and cube_id not in order:
                order.append(cube_id)
        return order

    def cube_tfs_ready(self, cube_ids: List[str]) -> bool:
        base = self.base_frame
        for cid in cube_ids:
            frame = self.cube_tf_prefix + cid
            if not self._can_T(base, frame):
                return False
        return True

    def missing_cube_tfs(self, cube_ids: List[str]) -> List[str]:
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
        self.sequence = []
        self.seq_index = 0
        self.step_started = False
        self.step_start_time = None

    def has_pending_steps(self) -> bool:
        return self.seq_index < len(self.sequence)

    def start_step(self, step: Step):
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

        assert step.target_translation is not None
        assert step.hand in ["left", "right"]

        #T = self.make_target_se3(step.target_translation)
        T = pin.SE3.Identity()
        T.translation = np.array(step.target_translation)

        if step.hand == "left":
            self.left_hand_controller.set_target_pose(T, duration=step.duration, type=step.rel_or_abs)
        else:
            self.right_hand_controller.set_target_pose(T, duration=step.duration, type=step.rel_or_abs)

        self.step_started = True
        self.step_start_time = self.now_s()

    def finish_sequence(self):
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
        def ev(ev_type: str, cube_id: str) -> ManipulationEvent:
            e = ManipulationEvent()
            e.type = ev_type
            e.cube_id = cube_id
            e.stamp = self.get_clock().now().to_msg()
            return e

        return [
            ev("PICK", "red"),   ev("PLACE", "red"),
            ev("PICK", "blue"),  ev("PLACE", "blue"),
            ev("PICK", "green"), ev("PLACE", "green"),
        ]

    # -----------------------------
    # Stacking planner: events -> steps
    # -----------------------------
    def choose_arm_for_cube(self, p: np.ndarray) -> str:
        # y<0 => right else left (x forward, y left, z up)
        return "right" if float(p[1]) < 0.0 else "left"

    def build_stacking_steps_from_events(self, events: List[ManipulationEvent]) -> List[Step]:
        order = self.extract_cube_order(events)
        if len(order) < 2:
            self.get_logger().warn(f"Not enough cubes in events to stack: {order}")
            return []

        poses: Dict[str, np.ndarray] = {}
        for cube_id in order:
            p = self.get_cube_position(cube_id)
            if p is None:
                self.get_logger().warn(f"Missing TF for cube '{cube_id}'. Cannot build stacking plan.")
                return []
            poses[cube_id] = p.copy()

        base_id = order[0]
        base_p = poses[base_id]

        z_above = 0.0
        z_grasp = 0.0
        z_lift  = 0.1
        place_clear = self.cube_size + 0.05

        steps: List[Step] = []

        for i, cube_id in enumerate(order[1:], start=1):
            pick_p = poses[cube_id]
            arm = self.choose_arm_for_cube(pick_p)

            # stack on base, height increases by cube_size
            stack_p = base_p + np.array([0.0, 0.0, i * self.cube_size])

            # PICK
            steps.append(Step(kind="grip", grip_cmd="open", grip_which=arm, duration=0.5, wait_after=0.2))
            steps.append(Step(kind="move", hand=arm, rel_or_abs="abs",
                              target_translation=pick_p + np.array([0.0, 0.0, z_above]),
                              duration=2.5, wait_after=0.2))
            #steps.append(Step(kind="move", hand=arm, rel_or_abs="abs",
            #                  target_translation=pick_p + np.array([0.0, 0.0, z_grasp]),
            #                  duration=1.5, wait_after=0.1))
            steps.append(Step(kind="grip", grip_cmd="close", grip_which=arm, duration=0.5, wait_after=0.2))
            steps.append(Step(kind="move", hand=arm, rel_or_abs="abs",
                              target_translation=pick_p + np.array([0.0, 0.0, z_lift]),
                              duration=2.0, wait_after=0.2))

            # PLACE
            steps.append(Step(kind="move", hand=arm, rel_or_abs="abs",
                              target_translation=stack_p + np.array([0.0, 0.0, place_clear]),
                              duration=3.0, wait_after=0.2))
            #steps.append(Step(kind="move", hand=arm, rel_or_abs="abs",
            #                  target_translation=stack_p + np.array([0.0, 0.0, z_grasp]),
            #                  duration=1.5, wait_after=0.1))
            steps.append(Step(kind="grip", grip_cmd="open", grip_which=arm, duration=0.5, wait_after=0.2))
            steps.append(Step(kind="move", hand=arm, rel_or_abs="abs",
                              target_translation=stack_p + np.array([0.0, 0.0, z_above]),
                              duration=2.0, wait_after=0.2))

        return steps

    # -----------------------------
    # SERVER: start recording, wait for result
    # -----------------------------
    def start_recording_and_wait_result(self):
        if self.waiting_for_server_result:
            self.get_logger().warn("Already waiting for a server result; ignoring.")
            return

        self.get_logger().info("Waiting for action server 'record_demo'...")
        if not self.client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Action server 'record_demo' not available.")
            return

        goal = RecordDemo.Goal()
        goal.min_world_states = int(self.min_world_states)
        goal.idle_timeout_sec = float(self.idle_timeout_sec)

        self.waiting_for_server_result = True
        future = self.client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected by server.")
            self.waiting_for_server_result = False
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _result_cb(self, future):
        res = future.result().result
        self.pending_events = list(res.events)
        self.ready_to_stack = True
        self.waiting_for_server_result = False
        self.get_logger().info(f"SERVER: ready_to_stack=True (events={len(self.pending_events)})")

    # -----------------------------
    # Main loop
    # -----------------------------
    def control_loop(self):

        # Init once
        if not self.did_init:
            self.move_to_initial_position()
            self.did_init = True
            self.phase = "idle"
            self.get_logger().info("Init done -> idle.")
            self.ainex_robot.update(self.v_left, self.v_right, self.dt)
            return

        # Idle -> build plan if ready
        if self.phase == "idle":
            if self.ready_to_stack and self.pending_events and not self.has_pending_steps():
                required = self.extract_cube_order(self.pending_events)

                if not required:
                    self.get_logger().warn("ready_to_stack but no cubes in events. Staying idle.")
                    self.ready_to_stack = False
                else:
                    if not self.cube_tfs_ready(required):
                        self.get_logger().warn(
                            f"Waiting for cube TFs... missing: {self.missing_cube_tfs(required)}",
                            throttle_duration_sec=2.0,
                        )
                    else:
                        steps = self.build_stacking_steps_from_events(self.pending_events)
                        if steps:
                            self.sequence = steps
                            self.seq_index = 0
                            self.step_started = False
                            self.step_start_time = None
                            self.ready_to_stack = False
                            self.get_logger().info(f"Built stacking plan with {len(self.sequence)} steps. Starting.")
                        else:
                            self.get_logger().warn("TFs ready but could not build plan. Staying idle.")
                            self.ready_to_stack = False

            if self.autorun and self.has_pending_steps():
                self.phase = "run_sequence"
                self.get_logger().info("Idle -> run_sequence.")

        # Run sequence
        if self.phase == "run_sequence":
            if not self.has_pending_steps():
                self.finish_sequence()
            else:
                current = self.sequence[self.seq_index]

                if not self.step_started:
                    self.start_step(current)

                if current.kind == "move":
                    if current.hand == "left":
                        self.v_left = self.left_hand_controller.update(self.dt)
                    else:
                        self.v_right = self.right_hand_controller.update(self.dt)

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

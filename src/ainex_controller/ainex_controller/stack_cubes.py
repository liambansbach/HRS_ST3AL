#!/usr/bin/env python3
from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from ament_index_python.packages import get_package_share_directory

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


# -----------------------------
# TF / Math helpers
# -----------------------------
def quat_to_rotmat(x, y, z, w):
    n = x*x + y*y + z*z + w*w
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    return np.array(
        [
            [1 - (y*y + z*z) * s, (x*y - z*w) * s,     (x*z + y*w) * s],
            [(x*y + z*w) * s,     1 - (x*x + z*z) * s, (y*z - x*w) * s],
            [(x*z - y*w) * s,     (y*z + x*w) * s,     1 - (x*x + y*y) * s],
        ],
        dtype=np.float64,
    )


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
        self.declare_parameter("use_test_sequence", True)
        self.use_test_sequence: bool = bool(self.get_parameter("use_test_sequence").value)

        self.declare_parameter("sim", False)
        self.sim: bool = bool(self.get_parameter("sim").value)

        # Controllers expect a base frame (your test node implicitly uses this convention)
        self.declare_parameter("control_base_frame", "base_link")
        self.declare_parameter("control_base_frame_alt", "body_link")
        self.control_base_pref = str(self.get_parameter("control_base_frame").value)
        self.control_base_alt = str(self.get_parameter("control_base_frame_alt").value)
        self.control_base_active: Optional[str] = None

        # cube tf prefix
        self.declare_parameter("cube_tf_prefix", "hrs_cube_")
        self.cube_tf_prefix: str = str(self.get_parameter("cube_tf_prefix").value)

        # Action goal params
        self.declare_parameter("min_world_states", 3)
        self.declare_parameter("idle_timeout_sec", 10.0)
        self.min_world_states = int(self.get_parameter("min_world_states").value)
        self.idle_timeout_sec = float(self.get_parameter("idle_timeout_sec").value)

        # cube size (meters)
        self.declare_parameter("cube_size", 0.05)
        self.cube_size = float(self.get_parameter("cube_size").value)

        # Tunables for stack planner (all in CONTROL BASE FRAME)
        self.declare_parameter("z_above", 0.10)
        self.declare_parameter("z_grasp", 0.02)
        self.declare_parameter("z_lift", 0.12)
        self.declare_parameter("place_clear", 0.10)

        # Optional: grasp point offset added to cube TF position (in control base frame)
        # Example: if cube TF is at center and you want to grab slightly above center:
        #   cube_grasp_offset = [0, 0, 0.01]
        self.declare_parameter("cube_grasp_offset", [0.0, 0.0, 0.0])
        self.cube_grasp_offset = np.array(self.get_parameter("cube_grasp_offset").value, dtype=np.float64).reshape(3)

        # -----------------------------
        # Robot model / controller setup
        # -----------------------------
        pkg = get_package_share_directory("ainex_description")
        urdf_path = pkg + "/urdf/ainex.urdf"

        self.robot_model = AiNexModel(self, urdf_path)
        self.ainex_robot = AinexRobot(self, self.robot_model, self.dt, sim=self.sim)

        self.left_hand_controller = HandController(self, self.robot_model, arm_side="left")
        self.right_hand_controller = HandController(self, self.robot_model, arm_side="right")

        # Arm DOFs for hold/zero commands
        self.left_dof = len(self.robot_model.get_arm_ids("left"))
        self.right_dof = len(self.robot_model.get_arm_ids("right"))
        self.zero_left = np.zeros(self.left_dof)
        self.zero_right = np.zeros(self.right_dof)

        # -----------------------------
        # Initial pose
        # -----------------------------
        self.init_robot_pose = {
            "head_tilt": -0.65, "head_pan": 0,
            "r_gripper": 0, "l_gripper": 0,
            "r_el_yaw": 0.8, "l_el_yaw": -0.8,
            "r_el_pitch": -1.4, "l_el_pitch": -1.4,
            "r_sho_roll": 0.0, "l_sho_roll": 0.0,
            "r_sho_pitch": 1.57, "l_sho_pitch": 1.57,
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

    def make_target_se3(self, translation_xyz: np.ndarray) -> pin.SE3:
        T = pin.SE3.Identity()
        T.translation = np.array(translation_xyz, dtype=float).reshape(3)
        return T

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
        Rm = quat_to_rotmat(
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
            tf.transform.rotation.w,
        )
        t = np.array(
            [tf.transform.translation.x,
             tf.transform.translation.y,
             tf.transform.translation.z],
            dtype=np.float64,
        )
        return Rm, t

    def _resolve_control_base_frame(self) -> Optional[str]:
        """
        Pick a base frame that is actually connected to cube TFs.
        Prefer base_link, fallback to body_link.
        """
        test_child = self.cube_tf_prefix + "red"

        # Prefer pref if it can transform to cube
        if self._can_T(self.control_base_pref, test_child):
            return self.control_base_pref

        # Else try alt
        if self._can_T(self.control_base_alt, test_child):
            if self.control_base_active != self.control_base_alt:
                self.get_logger().warn(f"Resolved control base frame -> '{self.control_base_alt}'")
            return self.control_base_alt

        return None

    def _update_cube_positions(self):
        # resolve base frame once cubes exist
        if self.control_base_active is None:
            base = self._resolve_control_base_frame()
            if base is None:
                self.cube_pos["red"] = None
                self.cube_pos["green"] = None
                self.cube_pos["blue"] = None
                return
            self.control_base_active = base
            self.get_logger().info(f"Using control base frame for cubes + targets: '{self.control_base_active}'")

        base = self.control_base_active

        # Directly lookup base <- cube (no intermediate world frame!)
        for c in ["red", "green", "blue"]:
            frame = self.cube_tf_prefix + c
            Tbc = self._lookup_RT(base, frame)  # base <- cube
            if Tbc is None:
                self.cube_pos[c] = None
                continue

            _, p_base = Tbc

            # Optional grasp-point tweak
            self.cube_pos[c] = p_base + self.cube_grasp_offset

    def get_cube_position(self, color: str) -> Optional[np.ndarray]:
        return self.cube_pos.get(color, None)

    # -----------------------------
    # "wait until TFs exist" helpers
    # -----------------------------
    def extract_cube_order(self, events: List[ManipulationEvent]) -> List[str]:
        order: List[str] = []
        for e in events:
            cid = str(e.cube_id)
            if cid in ["red", "green", "blue"] and cid not in order:
                order.append(cid)
        return order

    def cube_tfs_ready(self, cube_ids: List[str]) -> bool:
        if self.control_base_active is None:
            base = self._resolve_control_base_frame()
            if base is None:
                return False
            self.control_base_active = base

        base = self.control_base_active
        for cid in cube_ids:
            frame = self.cube_tf_prefix + cid
            if not self._can_T(base, frame):
                return False
        return True

    def missing_cube_tfs(self, cube_ids: List[str]) -> List[str]:
        base = self.control_base_active or "<no_base>"
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

        T = self.make_target_se3(step.target_translation)

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
        for cid in order:
            p = self.get_cube_position(cid)
            if p is None:
                self.get_logger().warn(f"Missing TF for cube '{cid}'. Cannot build stacking plan.")
                return []
            poses[cid] = p.copy()

        base_id = order[0]
        base_p = poses[base_id]

        z_above = float(self.get_parameter("z_above").value)
        z_grasp = float(self.get_parameter("z_grasp").value)
        z_lift  = float(self.get_parameter("z_lift").value)
        place_clear = float(self.get_parameter("place_clear").value)

        steps: List[Step] = []

        for i, cid in enumerate(order[1:], start=1):
            pick_p = poses[cid]
            arm = self.choose_arm_for_cube(pick_p)

            # stack on base, height increases by cube_size
            stack_p = base_p + np.array([0.0, 0.0, i * self.cube_size])

            # PICK
            steps.append(Step(kind="grip", grip_cmd="open", grip_which=arm, duration=0.8, wait_after=0.2))
            steps.append(Step(kind="move", hand=arm, rel_or_abs="abs",
                              target_translation=pick_p + np.array([0.0, 0.0, z_above]),
                              duration=2.5, wait_after=0.2))
            steps.append(Step(kind="move", hand=arm, rel_or_abs="abs",
                              target_translation=pick_p + np.array([0.0, 0.0, z_grasp]),
                              duration=1.5, wait_after=0.1))
            steps.append(Step(kind="grip", grip_cmd="close", grip_which=arm, duration=0.8, wait_after=0.2))
            steps.append(Step(kind="move", hand=arm, rel_or_abs="abs",
                              target_translation=pick_p + np.array([0.0, 0.0, z_lift]),
                              duration=2.0, wait_after=0.2))

            # PLACE
            steps.append(Step(kind="move", hand=arm, rel_or_abs="abs",
                              target_translation=stack_p + np.array([0.0, 0.0, place_clear]),
                              duration=3.0, wait_after=0.2))
            steps.append(Step(kind="move", hand=arm, rel_or_abs="abs",
                              target_translation=stack_p + np.array([0.0, 0.0, z_grasp]),
                              duration=1.5, wait_after=0.1))
            steps.append(Step(kind="grip", grip_cmd="open", grip_which=arm, duration=0.8, wait_after=0.2))
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
        v_left = self.zero_left
        v_right = self.zero_right

        # Init once
        if not self.did_init:
            self.move_to_initial_position()
            self.did_init = True
            self.phase = "idle"
            self.get_logger().info("Init done -> idle.")
            self.ainex_robot.update(v_left, v_right, self.dt)
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
                        v_left = self.left_hand_controller.update(self.dt)
                    else:
                        v_right = self.right_hand_controller.update(self.dt)

                t = self.step_elapsed()
                if t >= (current.duration + current.wait_after):
                    self.seq_index += 1
                    self.step_started = False
                    self.step_start_time = None

                    if not self.has_pending_steps():
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

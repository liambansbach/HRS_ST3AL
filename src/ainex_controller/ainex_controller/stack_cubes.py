#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from ament_index_python.packages import get_package_share_directory

import numpy as np
import pinocchio as pin
from dataclasses import dataclass
from typing import List, Optional, Dict

from tf2_ros import TransformListener, Buffer

from ainex_interfaces.action import RecordDemo
from ainex_interfaces.msg import ManipulationEvent  # for test events like server returns

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
    """
    A single sequential step.

    kind="move":
      - hand: "left"/"right"
      - rel_or_abs: "rel"/"abs"
      - target_translation: np.ndarray (3,)
      - duration, wait_after

    kind="grip":
      - grip_cmd: "open"/"close"
      - grip_which: "left"/"right"/"both"
      - duration, wait_after
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
    def __init__(self):
        super().__init__("stack_cubes")

        self.dt = 0.05  # 20 Hz

        # -----------------------------
        # Params
        # -----------------------------
        self.declare_parameter("use_test_sequence", True) 
        self.use_test_sequence: bool = bool(self.get_parameter("use_test_sequence").value)

        # Let you switch RViz-only vs real robot
        self.declare_parameter("sim", True)
        self.sim: bool = bool(self.get_parameter("sim").value)

        # Base frame naming mess: URDF uses body_link; some nodes publish base_link. But its the same...
        self.declare_parameter("base_frame", "body_link")      # preferred
        self.declare_parameter("alt_base_frame", "base_link")  # fallback
        self.base_frame_pref: str = str(self.get_parameter("base_frame").value)
        self.base_frame_alt: str = str(self.get_parameter("alt_base_frame").value)
        self.base_frame_active: str = self.base_frame_pref  # will auto-resolve at runtime

        # Action goal params (same as record_demo client)
        self.declare_parameter("min_world_states", 3)
        self.declare_parameter("idle_timeout_sec", 10.0)
        self.min_world_states = int(self.get_parameter("min_world_states").value)
        self.idle_timeout_sec = float(self.get_parameter("idle_timeout_sec").value)

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
            "head_tilt": -0.5, "head_pan": 0,
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

        # -----------------------------
        # TF (cube poses)
        # -----------------------------
        self.cube_tf_prefix = "hrs_cube_"  # hrs_cube_red/green/blue

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cube_pos: Dict[str, Optional[np.ndarray]] = {"red": None, "green": None, "blue": None}
        self.create_timer(1.0 / 20.0, self._update_cube_positions)

        # -----------------------------
        # Sequencer state
        # -----------------------------
        self.sequence: List[Step] = []
        self.seq_index: int = 0
        self.step_started: bool = False
        self.step_start_time: Optional[float] = None

        self.phase: str = "init"      # "init" -> "idle" <-> "run_sequence"
        self.autorun: bool = True
        self.clear_sequence_on_finish = True
        self.did_init = False

        # "ready to stack" latch: set when server result arrives (or immediately in test mode)
        self.ready_to_stack: bool = False
        self.pending_events: List[ManipulationEvent] = []

        # cube size
        self.cube_size = 0.05  # 5cm

        # main loop
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
    def _can_T(self, parent, child) -> bool:
        return self.tf_buffer.can_transform(parent, child, rclpy.time.Time())

    def _get_T(self, parent, child):
        if not self._can_T(parent, child):
            return None
        tf = self.tf_buffer.lookup_transform(parent, child, rclpy.time.Time())
        R = quat_to_rotmat(
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
        return R, t

    def _resolve_base_frame(self) -> str:
        """
        Robustly resolve whether cubes live under body_link or base_link.
        We pick the first one that can transform to any cube frame.
        """
        # If already resolved and still valid, keep it.
        test_child = self.cube_tf_prefix + "red"
        if self._can_T(self.base_frame_active, test_child):
            return self.base_frame_active

        # Try preferred
        if self._can_T(self.base_frame_pref, test_child):
            if self.base_frame_active != self.base_frame_pref:
                self.get_logger().warn(f"Resolved base frame -> '{self.base_frame_pref}'")
            self.base_frame_active = self.base_frame_pref
            return self.base_frame_active

        # Try alternative
        if self._can_T(self.base_frame_alt, test_child):
            if self.base_frame_active != self.base_frame_alt:
                self.get_logger().warn(f"Resolved base frame -> '{self.base_frame_alt}'")
            self.base_frame_active = self.base_frame_alt
            return self.base_frame_active

        # No cube TF yet; keep current, but don't spam.
        return self.base_frame_active

    def _update_cube_positions(self):
        base = self._resolve_base_frame()
        for c in ["red", "green", "blue"]:
            frame = self.cube_tf_prefix + c
            T = self._get_T(base, frame)
            self.cube_pos[c] = None if T is None else T[1]

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
        base = self._resolve_base_frame()
        for cid in cube_ids:
            frame = self.cube_tf_prefix + cid
            if not self._can_T(base, frame):
                return False
        return True

    def missing_cube_tfs(self, cube_ids: List[str]) -> List[str]:
        base = self._resolve_base_frame()
        missing = []
        for cid in cube_ids:
            frame = self.cube_tf_prefix + cid
            if not self._can_T(base, frame):
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
        # GRIP step
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
            self.get_logger().info(
                f"Start GRIP step {self.seq_index+1}/{len(self.sequence)}: "
                f"{step.grip_cmd} {step.grip_which} dur={step.duration}s wait={step.wait_after}s"
            )
            return

        # MOVE step
        assert step.target_translation is not None, "move step needs target_translation"
        assert step.hand in ["left", "right"], "move step needs hand left/right"

        T = self.make_target_se3(step.target_translation)

        if step.hand == "left":
            self.left_hand_controller.set_target_pose(T, duration=step.duration, type=step.rel_or_abs)
        else:
            self.right_hand_controller.set_target_pose(T, duration=step.duration, type=step.rel_or_abs)

        self.step_started = True
        self.step_start_time = self.now_s()

        self.get_logger().info(
            f"Start MOVE step {self.seq_index+1}/{len(self.sequence)}: "
            f"{step.hand} {step.rel_or_abs} trans={step.target_translation.tolist()} "
            f"dur={step.duration}s wait={step.wait_after}s"
        )

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
    # TEST: fake server events (same type as server returns)
    # -----------------------------
    def build_test_server_events(self) -> List[ManipulationEvent]:
        def ev(ev_type: str, cube_id: str) -> ManipulationEvent:
            e = ManipulationEvent()
            e.type = ev_type
            e.cube_id = cube_id
            e.stamp = self.get_clock().now().to_msg()
            return e

        # order matters: first unique cube becomes "base"
        return [
            ev("PICK", "red"),   ev("PLACE", "red"),
            ev("PICK", "blue"),  ev("PLACE", "blue"),
            ev("PICK", "green"), ev("PLACE", "green"),
        ]

    # -----------------------------
    # Stacking planner: events -> steps
    # -----------------------------
    def choose_arm_for_cube(self, p: np.ndarray) -> str:
        # y<0 => right else left
        return "right" if float(p[1]) < 0.0 else "left"

    def build_stacking_steps_from_events(self, events: List[ManipulationEvent]) -> List[Step]:
        # Extract cube order (unique appearance order)
        order = self.extract_cube_order(events)

        if len(order) < 2:
            self.get_logger().warn(f"Not enough cubes in events to stack: {order}")
            return []

        # Read TFs now
        poses: Dict[str, np.ndarray] = {}
        for cid in order:
            p = self.get_cube_position(cid)
            if p is None:
                self.get_logger().warn(f"Missing TF for cube '{cid}'. Cannot build stacking plan.")
                return []
            poses[cid] = p.copy()

        base_id = order[0]
        base_p = poses[base_id]

        # Tunable offsets (start conservative)
        z_above = 0.10     # approach above cube/target
        z_grasp = 0.02     # near contact height above plane
        z_lift  = 0.12     # lift after grasp
        place_clear = 0.10

        steps: List[Step] = []

        # Stack remaining cubes onto base, increasing height each time.
        for i, cid in enumerate(order[1:], start=1):
            pick_p = poses[cid]
            arm = self.choose_arm_for_cube(pick_p)

            # Target stack position: above base + i*cube_size
            stack_p = base_p + np.array([0.0, 0.0, i * self.cube_size])

            # --- PICK ---
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

            # --- PLACE ---
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

        self.get_logger().info(
            f"Sending RecordDemo goal (min_world_states={goal.min_world_states}, idle_timeout_sec={goal.idle_timeout_sec})"
        )

        self.waiting_for_server_result = True
        future = self.client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected by server.")
            self.waiting_for_server_result = False
            return

        self.get_logger().info("Server accepted goal. Waiting for result...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _result_cb(self, future):
        res = future.result().result
        self.get_logger().info(f"Events received: {len(res.events)}")

        self.pending_events = list(res.events)
        self.ready_to_stack = True
        self.waiting_for_server_result = False
        self.get_logger().info("SERVER: ready_to_stack=True (received result; can start stacking).")

    # -----------------------------
    # Main loop (timer)
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

        # Idle: if ready_to_stack, WAIT until TFs exist, then build plan
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
                self.get_logger().info("Idle -> run_sequence (pending steps detected).")

        # Run sequence (sequential)
        if self.phase == "run_sequence":
            if not self.has_pending_steps():
                self.finish_sequence()
            else:
                current = self.sequence[self.seq_index]

                if not self.step_started:
                    self.start_step(current)

                # MOVE: update active controller
                if current.kind == "move":
                    if current.hand == "left":
                        v_left = self.left_hand_controller.update(self.dt)
                    else:
                        v_right = self.right_hand_controller.update(self.dt)

                # Time-based completion (GRIP also just waits)
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

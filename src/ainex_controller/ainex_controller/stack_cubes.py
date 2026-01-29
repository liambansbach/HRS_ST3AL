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

from ainex_interfaces.action import RecordDemo  # server action :contentReference[oaicite:2]{index=2}

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

        # -----------------------------
        # Params
        # -----------------------------
        # True: run built-in test motions at startup
        # False: fetch events from server and then execute
        self.declare_parameter("use_test_sequence", False)
        self.use_test_sequence: bool = bool(self.get_parameter("use_test_sequence").value)

        # Action goal params (same as your client) :contentReference[oaicite:3]{index=3}
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
        self.ainex_robot = AinexRobot(self, self.robot_model, self.dt, sim=True)

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

        # -----------------------------
        # TF (cube poses)
        # -----------------------------
        self.base_frame = "base_link"
        self.cube_tf_prefix = "hrs_cube_"  # your projector TF naming :contentReference[oaicite:4]{index=4}

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cube_pos: Dict[str, Optional[np.ndarray]] = {"red": None, "green": None, "blue": None}
        self.create_timer(1.0 / 20.0, self._update_cube_positions)

        # -----------------------------
        # Sequencer
        # -----------------------------
        self.sequence: List[HandAction] = []
        self.seq_index: int = 0
        self.action_started: bool = False
        self.action_start_time: Optional[float] = None

        self.phase: str = "init"      # "init" -> "idle" <-> "run_sequence"
        self.autorun: bool = True
        self.clear_sequence_on_finish = True
        self.did_init = False

        self.timer = self.create_timer(self.dt, self.control_loop)

        # -----------------------------
        # ActionClient to server
        # -----------------------------
        self.client = ActionClient(self, RecordDemo, "record_demo")
        self.waiting_for_server_result = False

        # Startup behavior:
        if self.use_test_sequence:
            self.enqueue_sequence(self.build_test_sequence(), replace=True)
            self.get_logger().info("Loaded TEST sequence (use_test_sequence:=True).")
        else:
            # Immediately start recording on the server; when result comes, execute it.
            self.start_recording_and_wait_result()

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

    def _update_cube_positions(self):
        for c in ["red", "green", "blue"]:
            frame = self.cube_tf_prefix + c
            T = self._get_T(self.base_frame, frame)
            self.cube_pos[c] = None if T is None else T[1]

    def get_cube_position(self, color: str) -> Optional[np.ndarray]:
        return self.cube_pos.get(color, None)

    # -----------------------------
    # Sequence API
    # -----------------------------
    def enqueue_sequence(self, actions: List[HandAction], replace: bool = False):
        if replace:
            self.sequence = []
            self.seq_index = 0
            self.action_started = False
            self.action_start_time = None

        if actions:
            self.sequence.extend(actions)
            self.get_logger().info(
                f"Enqueued {len(actions)} actions (pending now {len(self.sequence) - self.seq_index})."
            )

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
            self.seq_index = len(self.sequence)
            self.action_started = False
            self.action_start_time = None
        self.phase = "idle"
        self.move_to_initial_position()

    # -----------------------------
    # Test sequence
    # -----------------------------
    def build_test_sequence(self) -> List[HandAction]:
        return [
            HandAction(hand="left",  rel_or_abs="rel", target_translation=np.array([0.00,  0.10, 0.00]), duration=3.0, wait_after=2.0),
            HandAction(hand="right", rel_or_abs="rel", target_translation=np.array([0.00, -0.10, 0.00]), duration=3.0, wait_after=2.0),
            HandAction(hand="left",  rel_or_abs="rel", target_translation=np.array([0.00, -0.10, 0.00]), duration=3.0, wait_after=2.0),
        ]

    # -----------------------------
    # SERVER: start recording, wait for result, translate to HandActions
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
        status = future.result().status
        res = future.result().result

        self.get_logger().info(f"=== SERVER RESULT status={status} session_id={res.session_id} ===")
        self.get_logger().info(f"Events received: {len(res.events)}")

        # Build a simple action sequence from events:
        actions = self._events_to_left_arm_actions(res.events)

        if not actions:
            self.get_logger().warn("No executable actions built from server events -> staying idle.")
        else:
            self.enqueue_sequence(actions, replace=True)
            self.get_logger().info("Loaded actions from server -> will execute sequence.")

        self.waiting_for_server_result = False

    def _events_to_left_arm_actions(self, events) -> List[HandAction]:
        """
        VERY SIMPLE TEST:
        For each event in order:
          - move LEFT arm to current 3D TF position of that cube (abs)
        """
        out: List[HandAction] = []

        for e in events:
            cube_id = str(e.cube_id)  # expected: "red"/"green"/"blue" :contentReference[oaicite:5]{index=5}
            p = self.get_cube_position(cube_id)

            if p is None:
                self.get_logger().warn(
                    f"Event {e.type} for cube '{cube_id}', but TF '{self.cube_tf_prefix + cube_id}' not available. Skipping."
                )
                continue

            # tiny offset in z so it doesn't collide immediately (tune later)
            target = p + np.array([0.0, 0.0, 0.05])

            out.append(
                HandAction(
                    hand="left",
                    rel_or_abs="abs",
                    target_translation=target,
                    duration=3.0,
                    wait_after=1.0,
                )
            )

            self.get_logger().info(f"Built action: left -> {cube_id} @ {target.tolist()} (from event {e.type})")

        return out

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

        # Idle
        if self.phase == "idle":
            if self.autorun and self.has_pending_actions():
                self.phase = "run_sequence"
                self.get_logger().info("Idle -> run_sequence (pending actions detected).")

        # Run sequence (sequential)
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

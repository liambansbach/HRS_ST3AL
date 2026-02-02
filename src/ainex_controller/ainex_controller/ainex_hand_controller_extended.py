import pinocchio as pin
from ainex_controller.ainex_model import AiNexModel
from ainex_controller.spline_trajectory import LinearSplineTrajectory
import numpy as np
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped


class HandController:
    def __init__(
        self,
        node: Node,
        model: AiNexModel,
        arm_side: str,
        Kp: np.ndarray = None,
        Kd: np.ndarray = None,
        *,
        enable_nullspace: bool = False,
        enable_feedforward: bool = True,
        adaptive_damping: bool = True,
        hard_stop_on_singularity: bool = False,
        k_null: float = 1.0,
        q_nominal: np.ndarray | None = None,
        lam_base: float = 1e-3, # higher values mean more damping, lower values mean less damping
        lam_gain: float = 5e-5, # higher values mean more damping near singularity
    ):
        """
        enable_nullspace:
            False -> legacy: u = J# xdot_des
            True  -> u = J# xdot_des + (I - J#J) qdot0
        """

        self.node = node
        self.br = TransformBroadcaster(node)

        self.robot_model = model
        self.arm_side = arm_side

        self.x_cur = pin.SE3.Identity()
        self.x_des = np.zeros(3)
        self.x_init = pin.SE3.Identity()
        self.x_target = pin.SE3.Identity()
        self.v_cur = pin.Motion.Zero()

        self.spline = None
        self.spline_duration = 0.0
        self.start_time = None

        self.w_threshold = 5e-4  # manipulability threshold

        self.Kp = Kp if Kp is not None else np.array([4.0, 4.0, 4.0])
        self.Kd = Kd if Kd is not None else np.array([0.9, 0.9, 0.9])

        # toggles / params
        self.enable_nullspace = enable_nullspace
        self.enable_feedforward = enable_feedforward
        self.adaptive_damping = adaptive_damping
        self.hard_stop_on_singularity = hard_stop_on_singularity

        self.k_null = float(k_null)
        self.lam_base = float(lam_base)
        self.lam_gain = float(lam_gain)

        # nominal posture: if not given, we'll capture it at first set_target_pose()
        self.q_nominal = None if q_nominal is None else np.asarray(q_nominal, dtype=float).copy()

    def _arm_q(self) -> np.ndarray:
        """Current arm joint positions (4,) in the same order as get_arm_v_ids()."""
        q_idx = self.robot_model.get_arm_ids(self.arm_side)  # idx_q for the 4 joints
        return np.asarray(self.robot_model.q[q_idx], dtype=float).copy()

    def set_target_pose(self, pose: pin.SE3, duration: float, type: str = "abs"):
        self.x_cur = self.robot_model.left_hand_pose() if self.arm_side == "left" else self.robot_model.right_hand_pose()
        self.x_init = self.x_cur.copy()

        if type == "abs":
            self.x_target = pose.copy()
        elif type == "rel":
            self.x_target = self.x_cur.copy()
            self.x_target.translation += pose.translation
            self.x_target.rotation = self.x_target.rotation @ pose.rotation
        else:
            raise ValueError("type must be 'abs' or 'rel'")

        self.spline = LinearSplineTrajectory(
            x_init=self.x_init.translation,
            x_final=self.x_target.translation,
            duration=duration,
            v_init=np.zeros(3),
            v_final=np.zeros(3),
        )
        self.spline_duration = duration
        self.start_time = self.node.get_clock().now().nanoseconds

        # capture nominal posture once, if not provided
        if self.q_nominal is None:
            self.q_nominal = self._arm_q()

    def update(self, dt: float) -> np.ndarray:
        # ---- TF target for visualization ----
        t_target = TransformStamped()
        t_target.header.stamp = self.node.get_clock().now().to_msg()
        t_target.header.frame_id = "base_link"
        t_target.child_frame_id = f"{self.arm_side}_target"
        t_target.transform.translation.x = float(self.x_target.translation[0])
        t_target.transform.translation.y = float(self.x_target.translation[1])
        t_target.transform.translation.z = float(self.x_target.translation[2])

        q = pin.Quaternion(self.x_target.rotation)
        t_target.transform.rotation.x = float(q.x)
        t_target.transform.rotation.y = float(q.y)
        t_target.transform.rotation.z = float(q.z)
        t_target.transform.rotation.w = float(q.w)
        self.br.sendTransform(t_target)

        # ---- Jacobian (position part) ----
        J6 = self.robot_model.left_hand_jacobian() if self.arm_side == "left" else self.robot_model.right_hand_jacobian()
        v_ids = self.robot_model.get_arm_v_ids(self.arm_side)  # idx_v order for the 4 joints
        J = J6[:3, :][:, v_ids]  # (3,4)

        # manipulability
        A = J @ J.T
        detA = np.linalg.det(A)
        w = float(np.sqrt(max(detA, 0.0)))

        # current pose/vel
        self.x_cur = self.robot_model.left_hand_pose() if self.arm_side == "left" else self.robot_model.right_hand_pose()
        self.v_cur = self.robot_model.left_hand_velocity() if self.arm_side == "left" else self.robot_model.right_hand_velocity()
        v_lin = np.asarray(self.v_cur.linear, dtype=float)  # (3,)

        # trajectory sample
        elapsed_time_s = (self.node.get_clock().now().nanoseconds - self.start_time) * 1e-9
        t_eval = min(max(elapsed_time_s, 0.0), self.spline_duration)

        x_traj, v_traj = self.spline.update(t_eval)  # both (3,)
        x_traj = np.asarray(x_traj, dtype=float)
        v_traj = np.asarray(v_traj, dtype=float)

        # cartesian command: PD (+ optional feedforward)
        x_err = x_traj - np.asarray(self.x_cur.translation, dtype=float)
        xdot_cmd = self.Kp * x_err - self.Kd * v_lin
        if self.enable_feedforward:
            xdot_cmd = xdot_cmd + v_traj

        # damping strategy
        if self.adaptive_damping:
            # increase damping near singularity smoothly
            lam = self.lam_base + self.lam_gain / (w + 1e-6)
        else:
            lam = self.lam_base

        # optional legacy behavior: hard stop
        if self.hard_stop_on_singularity and (w < self.w_threshold):
            return np.zeros(4)

        # DLS pseudoinverse (for 3x4 J): J# = J^T (J J^T + lam^2 I)^-1
        inv_term = np.linalg.inv(J @ J.T + (lam ** 2) * np.eye(3))
        Jsharp = J.T @ inv_term  # (4,3)

        u_task = Jsharp @ xdot_cmd  # (4,)

        if not self.enable_nullspace:
            return u_task

        # ---- Nullspace posture term ----
        q_arm = self._arm_q()  # (4,)
        q_nom = self.q_nominal if self.q_nominal is not None else q_arm
        qdot0 = -self.k_null * (q_arm - q_nom)  # (4,)

        N = np.eye(4) - (Jsharp @ J)  # (4,4)
        u = u_task + N @ qdot0

        return u

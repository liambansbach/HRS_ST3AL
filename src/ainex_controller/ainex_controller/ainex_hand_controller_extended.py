#!/usr/bin/env python3
"""
Ainex HandController (Extended)

Purpose
-------
Cartesian (position-only) hand controller for the AiNex arms using Pinocchio kinematics.
It tracks a target end-effector position via a DLS (damped least-squares) Jacobian
pseudoinverse controller. Optionally, it adds a nullspace posture term to keep the arm
close to a nominal configuration.

Key features
------------
- Position-only Cartesian tracking (3D translation), no orientation control.
- LinearSplineTrajectory for smooth target interpolation.
- DLS pseudoinverse with optional adaptive damping near singularities.
- Optional feedforward (trajectory velocity).
- Optional nullspace posture control:
    u = J# xdot_cmd + (I - J#J) qdot0

Inputs / Outputs
----------------
Inputs:
  - set_target_pose(): desired pose (abs or relative) + duration
  - update(dt): uses current robot state from AiNexModel (q, poses, Jacobians)

Output:
  - update(dt) returns joint velocity command u for the 4 arm joints (np.ndarray shape (4,))
"""

import numpy as np
import pinocchio as pin
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

from ainex_controller.ainex_model import AiNexModel
from ainex_controller.spline_trajectory import LinearSplineTrajectory


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
        lam_base: float = 1e-3,  # higher -> more damping, lower -> less damping
        lam_gain: float = 8e-5,  # higher -> more damping near singularity
    ):
        """
        Initialize the Cartesian hand controller.

        Args:
            node: ROS2 node (clock + TF publishing).
            model: AiNexModel wrapper around the Pinocchio model/state.
            arm_side: 'left' or 'right' arm.
            Kp: Cartesian proportional gains (3,). Defaults to [4,4,4].
            Kd: Cartesian derivative gains (3,). Defaults to [0.4,0.4,0.4].

        Keyword Args:
            enable_nullspace: Add a nullspace posture term (keeps arm near q_nominal).
            enable_feedforward: Add trajectory velocity v_traj to xdot_cmd.
            adaptive_damping: Increase damping as manipulability decreases.
            hard_stop_on_singularity: If True and w < w_threshold -> return zeros.
            k_null: Gain for nullspace posture control.
            q_nominal: Nominal arm posture (4,). If None, captured on first set_target_pose().
            lam_base: Base damping (lambda).
            lam_gain: Damping gain used for adaptive damping.

        Notes:
            - This controller tracks translation only. Orientation is not controlled.
            - Arm joint ordering follows model.get_arm_v_ids() / model.get_arm_ids().
        """
        self.node = node
        self.br = TransformBroadcaster(node)

        self.robot_model = model
        self.arm_side = arm_side

        # Current/desired pose state
        self.x_cur = pin.SE3.Identity()
        self.x_des = np.zeros(3)
        self.x_init = pin.SE3.Identity()
        self.x_target = pin.SE3.Identity()
        self.v_cur = pin.Motion.Zero()

        # Trajectory object (translation only)
        self.spline: LinearSplineTrajectory | None = None
        self.spline_duration = 0.0
        self.start_time: int | None = None  # nanoseconds

        # Manipulability threshold for optional hard stop
        self.w_threshold = 5e-4

        # Gains
        self.Kp = Kp if Kp is not None else np.array([4.0, 4.0, 4.0])
        self.Kd = Kd if Kd is not None else np.array([0.4, 0.4, 0.4])

        # Toggles / params
        self.enable_nullspace = enable_nullspace
        self.enable_feedforward = enable_feedforward
        self.adaptive_damping = adaptive_damping
        self.hard_stop_on_singularity = hard_stop_on_singularity

        self.k_null = float(k_null)
        self.lam_base = float(lam_base)
        self.lam_gain = float(lam_gain)

        # Nominal posture: captured at first set_target_pose() if not provided
        self.q_nominal = None if q_nominal is None else np.asarray(q_nominal, dtype=float).copy()

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _arm_q(self) -> np.ndarray:
        """
        Current arm joint positions (4,) in the same order as get_arm_v_ids().

        Returns:
            Arm joint positions as np.ndarray (shape (4,)).
        """
        q_idx = self.robot_model.get_arm_ids(self.arm_side)  # idx_q for the 4 joints
        return np.asarray(self.robot_model.q[q_idx], dtype=float).copy()

    # ----------------------------
    # Public API
    # ----------------------------
    def set_target_pose(self, pose: pin.SE3, duration: float, type: str = "abs"):
        """
        Set a new target pose and create a translation spline trajectory.

        The controller tracks only translation. Orientation is stored for visualization
        TF but not controlled in the Jacobian control law.

        Args:
            pose: Target pose (pin.SE3). For 'rel', translation/rotation are applied on top of x_cur.
            duration: Trajectory duration in seconds.
            type: 'abs' for absolute target, 'rel' for relative target.

        Raises:
            ValueError: If type is not 'abs' or 'rel'.

        Returns:
            None
        """
        # Read current end-effector pose
        self.x_cur = (
            self.robot_model.left_hand_pose()
            if self.arm_side == "left"
            else self.robot_model.right_hand_pose()
        )
        self.x_init = self.x_cur.copy()

        # Resolve absolute/relative target
        if type == "abs":
            self.x_target = pose.copy()
        elif type == "rel":
            self.x_target = self.x_cur.copy()
            self.x_target.translation += pose.translation
            self.x_target.rotation = self.x_target.rotation @ pose.rotation
        else:
            raise ValueError("type must be 'abs' or 'rel'")

        # Build translation spline (smooth start/stop)
        self.spline = LinearSplineTrajectory(
            x_init=self.x_init.translation,
            x_final=self.x_target.translation,
            duration=duration,
            v_init=np.zeros(3),
            v_final=np.zeros(3),
        )
        self.spline_duration = float(duration)
        self.start_time = self.node.get_clock().now().nanoseconds

        # Capture nominal posture once (if not provided)
        if self.q_nominal is None:
            self.q_nominal = self._arm_q()

    def update(self, dt: float) -> np.ndarray:
        """
        Compute joint velocity command for the current time step.

        Processing steps:
          1) Publish TF for the current target pose (debug visualization).
          2) Compute end-effector Jacobian (position rows only).
          3) Sample desired trajectory (x_traj, v_traj).
          4) Compute Cartesian PD command (+ optional feedforward).
          5) Compute DLS pseudoinverse J# with chosen damping (adaptive or fixed).
          6) Compute task-space joint velocity u_task.
          7) Optionally add nullspace posture correction.
          8) Return 4D joint velocity command.

        Args:
            dt: Control time step in seconds (currently not used directly; kept for API consistency).

        Returns:
            Joint velocity command u (shape (4,)).
        """
        # ---- TF target for visualization (RViz) ----
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

        # ---- Jacobian (position part only) ----
        J6 = (
            self.robot_model.left_hand_jacobian()
            if self.arm_side == "left"
            else self.robot_model.right_hand_jacobian()
        )
        v_ids = self.robot_model.get_arm_v_ids(self.arm_side)  # idx_v order for the 4 joints
        J = J6[:3, :][:, v_ids]  # (3,4)

        # Manipulability measure: w = sqrt(det(J J^T))
        A = J @ J.T
        detA = np.linalg.det(A)
        w = float(np.sqrt(max(detA, 0.0)))

        # Current pose/velocity
        self.x_cur = (
            self.robot_model.left_hand_pose()
            if self.arm_side == "left"
            else self.robot_model.right_hand_pose()
        )
        self.v_cur = (
            self.robot_model.left_hand_velocity()
            if self.arm_side == "left"
            else self.robot_model.right_hand_velocity()
        )
        v_lin = np.asarray(self.v_cur.linear, dtype=float)  # (3,)

        # Trajectory sample (translation)
        if self.start_time is None or self.spline is None:
            # Not initialized yet: no motion
            return np.zeros(4)

        elapsed_time_s = (self.node.get_clock().now().nanoseconds - self.start_time) * 1e-9
        t_eval = min(max(elapsed_time_s, 0.0), self.spline_duration)

        x_traj, v_traj = self.spline.update(t_eval)  # both (3,)
        x_traj = np.asarray(x_traj, dtype=float)
        v_traj = np.asarray(v_traj, dtype=float)

        # Cartesian command: PD (+ optional feedforward)
        x_err = x_traj - np.asarray(self.x_cur.translation, dtype=float)
        xdot_cmd = self.Kp * x_err - self.Kd * v_lin
        if self.enable_feedforward:
            xdot_cmd = xdot_cmd + v_traj

        # Damping strategy (fixed or adaptive near singularity)
        if self.adaptive_damping:
            lam = self.lam_base + self.lam_gain / (w + 1e-6)
        else:
            lam = self.lam_base

        # Optional legacy behavior: hard stop if too close to singularity
        if self.hard_stop_on_singularity and (w < self.w_threshold):
            return np.zeros(4)

        # DLS pseudoinverse (for 3x4 J): J# = J^T (J J^T + lam^2 I)^-1
        inv_term = np.linalg.inv(J @ J.T + (lam ** 2) * np.eye(3))
        Jsharp = J.T @ inv_term  # (4,3)

        u_task = Jsharp @ xdot_cmd  # (4,)

        # No nullspace requested
        if not self.enable_nullspace:
            return u_task

        # ---- Nullspace posture term ----
        q_arm = self._arm_q()  # (4,)
        q_nom = self.q_nominal if self.q_nominal is not None else q_arm
        qdot0 = -self.k_null * (q_arm - q_nom)  # (4,)

        N = np.eye(4) - (Jsharp @ J)  # (4,4)
        u = u_task + N @ qdot0

        return u
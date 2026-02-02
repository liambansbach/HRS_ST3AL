import pinocchio as pin
from ainex_controller.ainex_model import AiNexModel
from ainex_controller.spline_trajectory import LinearSplineTrajectory
import numpy as np  
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import scipy as sci

class HandController():
    def __init__(self, node: Node, model: AiNexModel, arm_side: str,
                 Kp: np.ndarray = None,
                 Kd: np.ndarray = None):
        """Initialize the Ainex Arm Controller.
        
        Args:
            model: Pinocchio robot model.
            arm_side: 'left' or 'right' arm.
        """
        self.node = node
        self.br = TransformBroadcaster(node)

        self.robot_model = model
        self.arm_side = arm_side

        self.x_cur = pin.SE3.Identity()
        self.x_des = pin.SE3.Identity()
        
        self.x_init = pin.SE3.Identity()
        self.x_target = pin.SE3.Identity()

        self.v_cur = pin.Motion.Zero()

        self.spline = None
        self.spline_duration = 0.0
        self.start_time = None

        self.w_threshold = 5e-4  # manipulability threshold
        
        if Kp is not None:
            self.Kp = Kp
        else:
            self.Kp = np.array([5.0, 5.0, 5.0])
        
        if Kd is not None:
            self.Kd = Kd
        else:
            self.Kd = np.array([0.5, 0.5, 0.5])


    def set_target_pose(self, pose: pin.SE3, duration: float, type: str = 'abs'):
        """Set the desired target pose for the specified arm.
        
        Args:
            pose (pin.SE3): Desired end-effector pose.
            type (str): 'abs' or 'rel' pose setting.
        """

        # TODO: get current pose from the robot model
        self.x_cur = self.robot_model.left_hand_pose() if self.arm_side == 'left' else self.robot_model.right_hand_pose()
        # not sure if this part is correct:
        self.x_init = self.x_cur.copy()

        # TODO: set target pose based on type 
        # abs: absolute pose w.r.t. base_link
        # rel: relative pose w.r.t. current pose
        if type == 'abs':
            self.x_target = pose.copy()
        elif type == 'rel':
            self.x_target = self.x_cur.copy()
            self.x_target.translation += pose.translation #Offset in translation
            self.x_target.rotation = self.x_target.rotation @ pose.rotation #Offset in rotation
            
        # TODO: Plan a spline trajectory from current to target pose using the
        # class LinearSplineTrajectory defined in spline_trajectory.py
        self.spline = LinearSplineTrajectory(
            x_init=self.x_init.translation,
            x_final=self.x_target.translation,
            duration=duration,
            v_init=np.zeros(3), 
            v_final=np.zeros(3)
        )
        self.spline_duration = duration

        # TODO: save start time
        self.start_time = self.node.get_clock().now().nanoseconds
        
    def update(self, dt: float) -> np.ndarray:
        """Update the arm controller with new joint states.
        
        Args:
            joint_states (np.ndarray): Current joint positions.
            dt (float): Time step for the update.

        Returns:
            np.ndarray: Desired joint velocities for the arm.(4,)
        """

        # TODO: Broadcast target pose as TF for visualization
        # not sure if correct:
        #self.br = TransformBroadcaster(self.node)
        t_target = TransformStamped()
        t_target.header.stamp = self.node.get_clock().now().to_msg()
        t_target.header.frame_id = "base_link"
        
        t_target.child_frame_id = self.arm_side
        
        t_target.transform.translation.x = self.x_target.translation[0]
        t_target.transform.translation.y = self.x_target.translation[1]
        t_target.transform.translation.z = self.x_target.translation[2]

        q = pin.Quaternion(self.x_target.rotation)
        t_target.transform.rotation.x = q.x
        t_target.transform.rotation.y = q.y
        t_target.transform.rotation.z = q.z
        t_target.transform.rotation.w = q.w

        self.br.sendTransform(t_target)

        # TODO: Retrieve the end-effector Jacobian that relates 
        # the end-effector LINEAR velocity and the ARM JOINTS.
        # Hint: Extract the linear part of the full Jacobian by taking its first three rows, 
        # and keep only the columns corresponding to the arm joints.
        # You can obtain the arm joint indices using AinexModel.get_arm_ids(). 
        J = self.robot_model.left_hand_jacobian() if self.arm_side == 'left' else self.robot_model.right_hand_jacobian()
        J_pos = J[:3, :][:, self.robot_model.get_arm_v_ids(self.arm_side)]

        ## Check manipulability to prevent singularities
        # TODO: calculate the manipulability index with the task Jacobian J_pos.
        # Hint: w = sqrt(det(J * J^T))
        # If the manipulability is below the threshold self.w_threshold,
        # stop the robot by setting u to zero.
        
        #w = np.sqrt(np.linalg.det(J_pos @ J_pos.T))

        A = J_pos @ J_pos.T
        detA = np.linalg.det(A)
        w = np.sqrt(max(detA, 0.0))


        if w >= self.w_threshold:
            # TODO: get current pose and velocity from the robot model
            # We assume the model is already updated with the latest joint states
            self.x_cur = self.robot_model.left_hand_pose() if self.arm_side == 'left' else self.robot_model.right_hand_pose()
            self.v_cur = self.robot_model.left_hand_velocity() if self.arm_side == 'left' else self.robot_model.right_hand_velocity()

            v_lin = self.v_cur.linear  # shape (3,)


            # TODO: Calculate the time elapsed since the start of the trajectory
            # Then update the spline to get desired position at current time
            elapsed_time_s = (self.node.get_clock().now().nanoseconds - self.start_time)*1e-9 # convert ns to s
            t_eval = min(elapsed_time_s, self.spline_duration) # Model goes flippin crazy when the time ends
            self.x_des, _ = self.spline.update(t_eval)  

            # TODO: Implement the Cartesian PD control law for end-effector POSITION only (no orientation part)
            # compute desired end-effector velocity
            #xdot_des = self.Kp * (self.x_des - self.x_cur.translation) - self.Kd * self.v_cur[:3]
            xdot_des = self.Kp * (self.x_des - self.x_cur.translation) - self.Kd * v_lin

            # TODO: compute the control command (velocities for the arm joints)
            # by mapping the desired end-effector velocity to arm joint velocities 
            # using the Jacobian pseudoinverse

            # J_pinv = np.linalg.pinv(J_pos)
            # u = J_pinv @ xdot_des
             
            lam = 1e-2  # z.B. 1e-3 bis 1e-1 testen
            J = J_pos
            u = J.T @ np.linalg.inv(J @ J.T + (lam**2) * np.eye(3)) @ xdot_des
  

        else:
            u = np.zeros(4)
    
        return u
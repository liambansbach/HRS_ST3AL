import pinocchio as pin
from ainex_controller.ainex_model import AiNexModel
from ainex_controller.spline_trajectory import LinearSplineTrajectory
import numpy as np  
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

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
        self.x_cur = pin.SE3.Identity()
        self.x_init = self.x_cur.copy()

        # TODO: set target pose based on type 
        # abs: absolute pose w.r.t. base_link
        # rel: relative pose w.r.t. current pose

        if type == 'abs':
            self.x_target = pin.SE3.Identity()
        elif type == 'rel':
            self.x_target = pin.SE3.Identity()

        
        # TODO: Plan a spline trajectory from current to target pose using the
        # class LinearSplineTrajectory defined in spline_trajectory.py
        self.spline = None
        self.spline_duration = duration
        # TODO: save start time
        self.start_time = None

    def update(self, dt: float) -> np.ndarray:
        """Update the arm controller with new joint states.
        
        Args:
            joint_states (np.ndarray): Current joint positions.
            dt (float): Time step for the update.

        Returns:
            np.ndarray: Desired joint velocities for the arm.(4,)
        """


        # TODO: Broadcast target pose as TF for visualization
        self.br

        # TODO: get current pose and velocity from the robot model
        # We assume the model is already updated with the latest joint states
        self.x_cur

        # TODO: Calculate the time elapsed since the start of the trajectory
        # Then update the spline to get desired position at current time
        self.x_des
        

        # TODO: Implement the Cartesian PD control law for end-effector POSITION only (no orientation part)
        # compute desired end-effector velocity
        xdot_des
        
        # TODO: Retrieve the end-effector Jacibian that relates 
        # the end-effector LINEAR velocity and the ARM JOINTS.
        # Hint: Extract the linear part of the full Jacobian by taking its first three rows, 
        # and keep only the columns corresponding to the arm joints.
        # You can obtain the arm joint indices using AinexModel.get_arm_ids().
        J
        J_pos

        # TODO: compute the control command (velocities for the arm joints)
        # by mapping the desired end-effector velocity to arm joint velocities 
        # using the Jacobian pseudoinverse
        u
       
        ## Check manipulability to prevent singularities
        # TODO: calculate the manipulability index with the task Jacobian J_pos.
        # Hint: w = sqrt(det(J * J^T))
        # If the manipulability is below the threshold self.w_threshold,
        # stop the robot by setting u to zero.
        w
        if w < self.w_threshold:
            pass

        return u
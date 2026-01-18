import numpy as np

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from ainex_controller.ainex_model import AiNexModel
from ainex_controller.ainex_robot_pose_imitation import AinexRobot
from ainex_controller.nmpc_controller import NMPC

from ainex_interfaces.msg import RobotImitationTargets

""" 
    Main control node for performing Upperbody Imitation on the Ainex Humanoid Robot.
    The ImitationControlNode aims to:
        - Subscribe to a stream publishing upperbody pose estimation of a human, made through the Ainex's camera
        - Utilize a "non-linear model predictive controller" (NMPC, defined in a separate module) to perform:
            + Wrist imitation of the human in x and y coordinates (primary target -> to the highest degree)
            + Optimization over the null-space and do elbow imitation (secondary target -> best as possible)

    The node completes its goals by:
        1. Using the (x,y) coordinates of the wrist and the angle of the elbow, provided through the subscription,
            as tracking reference for the NMPC.
        2. Solving a sequential quadratic program, optimizing over the forward kinematics and a set of constraints
            to find the optimal inverse kinematic solution over N steps in a time horizon, 
            giving the joint positions and velocities necessary to reach the reference as output.
        3. Using the second (next step) joint position as input to the Ainex.

    Node parameters:
        T_HORIZON_s: Time horizon in second which the nmpc should optimize over (default: 1s)
        N: Number of steps for perform in the time horizon (default: 20)
        n_joints: Number of joints in the robot arm (default: 4)
        n_ref_vals: Number of reference values to optimize over [x_wrist, y_wrist, theta_elbow] (default: 3) 
        Q_diag: Optimization weights for reference error [x_wrist, y_wrist, theta_elbow] (default: [100, 100, 10]) 
        R_diag: Optimization weights for forward kinematic input [sho_pitch_dot, sho_roll_dot, el_pitch_dot, el_yaw_dot] (default: [0.1, 0.1, 0.1, 0.1])
        theta_dot_max: Maximum speed allowed for a joint (default: 2rad/s)        
        theta_min_left: Minimum angles allowed in left arm (default: [-pi, -pi, -pi, -pi])
        theta_max_left: Maximum angles allowed in left arm (default: [pi, pi, pi, pi])
        theta_min_right: Minimum angles allowed in right arm (default: [-pi, -pi, -pi, -pi])
        theta_max_right: Maximum angles allowed in right arm (default: [pi, pi, pi, pi])

"""

class ImitationControlNode(Node):
    def __init__(self):
        super().__init__('imitation_control_node')

        self.T_HORIZON_s = self.declare_parameter('T_HORIZON_s', 1).value
        self.N = self.declare_parameter('N', 20).value
        self.n_joints = self.declare_parameter('n_joints', 4).value
        self.n_ref_vals = self.declare_parameter('n_ref_vals', 3).value
        self.Q_diag = self.declare_parameter('Q_diag', [100, 100, 10]).value
        self.R_diag = self.declare_parameter('R_diag', [0.1, 0.1, 0.1, 0.1]).value
        self.theta_dot_max = self.declare_parameter('theta_dot_max', 10).value
        
        self.theta_min_left = self.declare_parameter('theta_min_left', [-np.pi, -np.pi, -np.pi, -np.pi]).value
        self.theta_max_left = self.declare_parameter('theta_max_left', [np.pi, np.pi, np.pi, np.pi]).value
        self.theta_min_right = self.declare_parameter('theta_min_right', [-np.pi, -np.pi, -np.pi, -np.pi]).value
        self.theta_max_right = self.declare_parameter('theta_max_right', [np.pi, np.pi, np.pi, np.pi]).value
        
        self.homogeneous_transform_params_left = {
            'T_0_1': ([0, 0, 0], '-y'),
            'T_1_2': ([0.02, 0.02151, 0], 'x'),
            'T_2_3': ([-0.02, 0.07411, 0], '-y'),
            'T_3_4': ([0.0004, 0.01702, 0.01907], 'z'),
            'T_4_wrist': ([0.01989, 0.0892, -0.019], 'non')}

        self.homogeneous_transform_params_right = {
            'T_0_1': ([0, 0, 0], '-y'),
            'T_1_2': ([0.02, -0.02151, 0], 'x'),
            'T_2_3': ([-0.02, -0.07411, 0], '-y'),
            'T_3_4': ([0.0004, -0.01702, 0.01907], 'z'),
            'T_4_wrist': ([0.01989, -0.0892, -0.019], 'non')}

        self.dt = self.T_HORIZON_s / self.N

        pkg = get_package_share_directory('ainex_description') # get package path
        urdf_path = pkg + '/urdf/ainex.urdf'
        self.robot_model = AiNexModel(self, urdf_path)

        # Create AinexRobot instance
        self.ainex_robot = AinexRobot(self, self.robot_model, self.dt, sim=True)

        # Create hand controllers for left and right hands
        self.left_hand_controller = NMPC(
            self.T_HORIZON_s,
            self.N,
            self.n_joints,
            self.n_ref_vals,
            self.Q_diag,
            self.R_diag,
            self.theta_dot_max,
            self.theta_min_left,
            self.theta_max_left,
            None, #self.robot_model,
            self.ainex_robot.left_arm_ids, 
            "l_gripper",
            self.homogeneous_transform_params_left
        )

        self.right_hand_controller = NMPC(
            self.T_HORIZON_s,
            self.N,
            self.n_joints,
            self.n_ref_vals,
            self.Q_diag,
            self.R_diag,
            self.theta_dot_max,
            self.theta_min_right,
            self.theta_max_right,
            None, #self.robot_model,
            self.ainex_robot.right_arm_ids,
            "r_gripper",
            self.homogeneous_transform_params_right
        )

        self.robot_targets_sub = self.create_subscription(
            RobotImitationTargets,
            "/robot_imitation_targets",
            self.target_cb,
            10
        )

    def target_cb(self, msg: RobotImitationTargets):
        x_left = msg.wrist_target_left.x
        y_left = msg.wrist_target_left.y
        angle_left_elbow = msg.angle_left_elbow

        x_right = msg.wrist_target_right.x
        y_right = msg.wrist_target_right.y
        angle_right_elbow = msg.angle_right_elbow

        q = self.ainex_robot.read_joint_positions_from_robot()

        optimal_solution_left = self.left_hand_controller.solve_nmpc(
            q[self.ainex_robot.left_arm_ids],
            [x_left, y_left, angle_left_elbow]
        )
        
        optimal_solution_right = self.right_hand_controller.solve_nmpc(
            q[self.ainex_robot.right_arm_ids],
            [x_right, y_right, angle_right_elbow]
        )

        self.ainex_robot.update(
            optimal_solution_left['theta'], 
            optimal_solution_right['theta'],
            optimal_solution_left['theta_dot'], 
            optimal_solution_right['theta_dot'],
            self.dt
        )

    def move_to_inital_position(self):
        # Home position defined in urdf/pinocchio model
        q_init = np.zeros(self.robot_model.model.nq)
        q_init[self.robot_model.get_joint_id('r_sho_roll')] = 1.4
        q_init[self.robot_model.get_joint_id('l_sho_roll')] = -1.4
        q_init[self.robot_model.get_joint_id('r_el_yaw')] = 1.58
        q_init[self.robot_model.get_joint_id('l_el_yaw')] = -1.58

        # Move robot to initial position
        self.ainex_robot.move_to_initial_position(q_init)

    def publish_joint_states(self):
        """Publish current joint states."""
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.node.get_clock().now().to_msg()
        joint_state_msg.name = self.joint_names
        joint_state_msg.position = self.q.tolist()
        joint_state_msg.velocity = self.v.tolist()
        self.joint_states_pub.publish(joint_state_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ImitationControlNode()
    node.get_logger().info('ImitationControlNode running')

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
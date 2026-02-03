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
        theta_min_right: Minimum angles allowed in right arm (default: [-2.09, -2.09, -2.09, -2.09])
        theta_max_right: Maximum angles allowed in right arm (default: [2.09, 2.09, 2.09, 2.09])
        theta_min_left: Minimum angles allowed in left arm (default: [-2.09, -2.09, -2.09, -2.09])
        theta_max_left: Maximum angles allowed in left arm (default: [2.09, 2.09, 2.09, 2.09])
"""
import time

import numpy as np
import pinocchio as pin
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from ainex_controller.ainex_model_imitation import AiNexModel
from ainex_controller.ainex_robot_pose_imitation import AinexRobot
from ainex_controller.nmpc_controller import NMPC

from ainex_interfaces.msg import RobotImitationTargets
from visualization_msgs.msg import Marker

class ImitationControlNode(Node):
    """Node for performing upperbody imitation on the Ainex humanoid robot using NMPC."""
    def __init__(self):
        super().__init__('imitation_control_node')

        self.T_HORIZON_s = self.declare_parameter('T_HORIZON_s', 3).value 
        self.N = self.declare_parameter('N', 15).value 
        self.n_joints = self.declare_parameter('n_joints', 4).value
        # 7 reference values: [x_wrist, y_wrist, z_wrist, sho_pitch, sho_roll, el_pitch, el_yaw]
        self.n_ref_vals = self.declare_parameter('n_ref_vals', 7).value
        # Q weights: [x, y, z, sho_pitch, sho_roll, el_pitch, el_yaw]
        # Position tracking (high weight), joint angle tracking (lower weight as "loose target")
        self.Q_diag = self.declare_parameter('Q_diag', [100, 100, 100, 5, 5, 5, 5]).value
        self.R_diag = self.declare_parameter('R_diag', [0.1, 0.1, 0.1, 0.1]).value
        self.theta_dot_max = self.declare_parameter('theta_dot_max', 5).value

        # Set simulation mode (True: RViz, False: real robot)!
        self.sim = self.declare_parameter('sim', False).value

        # set joint limits
        self.theta_min_right = self.declare_parameter('theta_min_right', [-2.09, -2.09, -2.09, -1.9]).value
        self.theta_max_right = self.declare_parameter('theta_max_right', [2.09, 2.09, 2.09, 0.3]).value
        self.theta_min_left = self.declare_parameter('theta_min_left', [-2.09, -2.09, -2.09, -1.9]).value  
        self.theta_max_left = self.declare_parameter('theta_max_left', [2.09, 2.09, 2.09, 0.3]).value
        self.dt = self.T_HORIZON_s / self.N

        pkg = get_package_share_directory('ainex_description') # get package path
        self.urdf_path = pkg + '/urdf/ainex.urdf'
        self.robot_model = AiNexModel(self, self.urdf_path)
        self.ainex_robot = AinexRobot(self, self.robot_model, self.dt, sim=self.sim)
        left_params = self.extract_fk_params_from_urdf('left')
        right_params = self.extract_fk_params_from_urdf('right')

        self.homogeneous_transform_params_left = left_params
        self.homogeneous_transform_params_right = right_params
        self._perf_counter_ns = time.perf_counter_ns
        self._nmpc_solve_time_ns = 0
        self._nmpc_solve_time_accum_ns = 0
        self._nmpc_solve_time_samples = 0

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
            self.homogeneous_transform_params_right
        )
        self.get_logger().info("TESTING: NMPC controllers initialized.")
        self.robot_targets_sub = self.create_subscription(
            RobotImitationTargets,
            "/robot_imitation_targets",
            self.target_cb,
            10
        )

        # Publishers for visualizing NMPC reference targets in RViz
        self.left_ref_marker_pub = self.create_publisher(Marker, '/nmpc_ref_left', 10)
        self.right_ref_marker_pub = self.create_publisher(Marker, '/nmpc_ref_right', 10)
        
        self.q_left = np.zeros(4)
        self.q_right = np.zeros(4)
        self.v_left = np.zeros(4)
        self.v_right = np.zeros(4)


    def extract_fk_params_from_urdf(self, arm_side: str) -> dict:
        """
        Extract the transform parameters (translation, rotation axis) from URDF
        to build a symbolic CasADi FK function.
        
        Args:
            arm_side: 'left' or 'right'
        
        Returns:
            dict: Transform parameters for each joint in the arm chain
        """
        model = self.robot_model.model
        if arm_side == 'left':
            joint_names = ['l_sho_pitch', 'l_sho_roll', 'l_el_pitch', 'l_el_yaw']
            gripper_frame = 'l_gripper_link'
        elif arm_side == 'right':
            joint_names = ['r_sho_pitch', 'r_sho_roll', 'r_el_pitch', 'r_el_yaw']
            gripper_frame = 'r_gripper_link'
        else:
            raise ValueError("arm_side must be 'left' or 'right'")
        
        def get_axis_string(model, joint_id):
            """
            Extract rotation axis from Pinocchio joint.
            Uses the joint's motion subspace to determine the axis.

            Args:
                model: Pinocchio model
                joint_id: ID of the joint to analyze
            
            Returns:
                str: 'x', 'y', 'z', '-x', '-y', '-z', or 'non' if not found
            """        

            joint = model.joints[joint_id]
            joint_type = joint.shortname()
            # Method 1: Check shortname for common patterns
            if 'RX' in joint_type or 'RevoluteX' in joint_type:
                return 'x'
            elif 'RY' in joint_type or 'RevoluteY' in joint_type:
                return 'y'
            elif 'RZ' in joint_type or 'RevoluteZ' in joint_type:
                return 'z'
            # Method 2: Analyze motion subspace 
            try:
                # Get the joint's motion subspace at q=0
                q = np.zeros(model.nq)
                data = model.createData()
                pin.computeJointJacobians(model, data, q)
                # Check the joint axis directly if available
                if hasattr(joint, 'axis'):
                    axis = joint.axis
                    if abs(axis[0]) > 0.9:
                        return '-x' if axis[0] < 0 else 'x'
                    elif abs(axis[1]) > 0.9:
                        return '-y' if axis[1] < 0 else 'y'
                    elif abs(axis[2]) > 0.9:
                        return '-z' if axis[2] < 0 else 'z'
            except:
                pass
            return 'non'

        # Define known axes from URDF 
        # From URDF: sho_pitch = Y axis, sho_roll = X axis, el_pitch = Y axis, el_yaw = Z axis
        known_axes = {
            'l_sho_pitch': 'y',   
            'l_sho_roll': 'x',    
            'l_el_pitch': '-y',    
            'l_el_yaw': '-z',     
            'r_sho_pitch': '-y',  
            'r_sho_roll': 'x',    
            'r_el_pitch': 'y',   
            'r_el_yaw': 'z',      
        }

        # Extract parameters for each joint       
        params = {}
        for i, joint_name in enumerate(joint_names):
            joint_id = model.getJointId(joint_name)
            placement = model.jointPlacements[joint_id]
            translation = placement.translation.tolist()
            axis = known_axes.get(joint_name, get_axis_string(model, joint_id))
            params[f'T_{i}_{i+1}'] = (translation, axis)
        
        # Get gripper frame placement relative to last joint
        if model.existFrame(gripper_frame):
            gripper_frame_id = model.getFrameId(gripper_frame)
            frame = model.frames[gripper_frame_id]
            gripper_translation = frame.placement.translation.tolist()
            params[f'T_{len(joint_names)}_wrist'] = (gripper_translation, 'non')
        
        return params
    
    def move_to_inital_position(self):
        """
        Move the Ainex robot to a predefined home position. (if needed)
        
        Args: 
            None

        Returns: 
            None
        """
        # Home position defined in urdf/pinocchio model
        q_init = np.zeros(self.robot_model.model.nq)
        q_init[self.robot_model.get_joint_id('r_sho_roll')] = 1.4
        q_init[self.robot_model.get_joint_id('l_sho_roll')] = -1.4
        q_init[self.robot_model.get_joint_id('r_el_yaw')] = 1.58
        q_init[self.robot_model.get_joint_id('l_el_yaw')] = -1.58

        # Move robot to initial position
        self.ainex_robot.move_to_initial_position(q_init)

    def publish_nmpc_reference(self, s_ref_left: list, s_ref_right: list):
        """
        Publish NMPC reference targets as markers for visualization in RViz. 
        
        Args:
            param s_ref_left: [x, y, z] wrist target position for left arm (relative to shoulder)
            param s_ref_right: [x, y, z] wrist target position for right arm (relative to shoulder)

        Returns: 
            None
        """
        now = self.get_clock().now().to_msg()

        # Left arm reference marker (relative to left shoulder pitch frame)
        left_marker = Marker()
        left_marker.header.stamp = now
        left_marker.header.frame_id = "l_sho_pitch_link"  # Reference frame for left arm FK
        left_marker.ns = "nmpc_reference"
        left_marker.id = 0
        left_marker.type = Marker.SPHERE
        left_marker.action = Marker.ADD
        left_marker.pose.position.x = float(s_ref_left[0])
        left_marker.pose.position.y = float(s_ref_left[1])
        left_marker.pose.position.z = float(s_ref_left[2])
        left_marker.pose.orientation.w = 1.0
        left_marker.scale.x = 0.03  # 3cm sphere
        left_marker.scale.y = 0.03
        left_marker.scale.z = 0.03
        left_marker.color.r = 0.0
        left_marker.color.g = 1.0  # Green for left
        left_marker.color.b = 0.0
        left_marker.color.a = 0.8
        left_marker.lifetime.sec = 0  # Persistent until updated
        self.left_ref_marker_pub.publish(left_marker)

        # Right arm reference marker (relative to right shoulder pitch frame)
        right_marker = Marker()
        right_marker.header.stamp = now
        right_marker.header.frame_id = "r_sho_pitch_link"  # Reference frame for right arm FK
        right_marker.ns = "nmpc_reference"
        right_marker.id = 1
        right_marker.type = Marker.SPHERE
        right_marker.action = Marker.ADD
        right_marker.pose.position.x = float(s_ref_right[0])
        right_marker.pose.position.y = float(s_ref_right[1])
        right_marker.pose.position.z = float(s_ref_right[2])
        right_marker.pose.orientation.w = 1.0
        right_marker.scale.x = 0.03  # 3cm sphere
        right_marker.scale.y = 0.03
        right_marker.scale.z = 0.03
        right_marker.color.r = 1.0  # Red for right
        right_marker.color.g = 0.0
        right_marker.color.b = 0.0
        right_marker.color.a = 0.8
        right_marker.lifetime.sec = 0  # Persistent until updated
        self.right_ref_marker_pub.publish(right_marker)

    def target_cb(self, msg: RobotImitationTargets):
        """
        Callback which receives robot imitation targets and current joint states. 
        Then solving NMPC for joint states (q) and joint velocity (v) and updating the Ainex robot.
        Also publishes reference targets for visualization in RViz and logs timing information (for profiling).

        Args: 
            param msg: RobotImitationTargets message containing wrist and elbow targets                

        Returns: 
            None
        """
        perf_counter_ns = self._perf_counter_ns
        t_start = perf_counter_ns()

        # Wrist position targets
        x_left = msg.wrist_target_left.x
        y_left = msg.wrist_target_left.y
        z_left = msg.wrist_target_left.z

        x_right = msg.wrist_target_right.x
        y_right = msg.wrist_target_right.y
        z_right = msg.wrist_target_right.z
            
        # Elbow and shoulder angle targets
        sho_pitch_target_left = msg.shoulder_pitch_target_left
        sho_roll_target_left = msg.shoulder_roll_target_left
        el_pitch_target_left = msg.elbow_pitch_target_left
        el_yaw_target_left = msg.elbow_yaw_target_left
    
        sho_pitch_target_right = msg.shoulder_pitch_target_right
        sho_roll_target_right = msg.shoulder_roll_target_right
        el_pitch_target_right = msg.elbow_pitch_target_right
        el_yaw_target_right = msg.elbow_yaw_target_right

        t_before_read_q = perf_counter_ns()
        # NOTE for reducing inference time: read only necessary joints
        q = self.ainex_robot.read_joint_positions_from_robot()
        t_read_q = perf_counter_ns()

        # Reference: [x, y, z, sho_pitch, sho_roll, el_pitch, el_yaw]
        refs_left = np.array(
            [x_left, y_left, z_left, sho_pitch_target_left, sho_roll_target_left, el_pitch_target_left, el_yaw_target_left],
            dtype=float
        )
        refs_right = np.array(
            [x_right, y_right, z_right, sho_pitch_target_right, sho_roll_target_right, el_pitch_target_right, el_yaw_target_right],
            dtype=float
        )

        # test if any of the reference targets is inf or nan (safty check):
        if not np.all(np.isfinite(refs_left)) or not np.all(np.isfinite(refs_right)):
            self.get_logger().warn("Skipping NMPC step due to non-finite reference targets.")
            return

        # Solve NMPC for left arm
        t_before_left_nmpc = perf_counter_ns()
        try:
            optimal_solution_left = self.left_hand_controller.solve_nmpc(
                q[self.ainex_robot.left_arm_ids],
                refs_left.tolist()
            )
            self.q_left = optimal_solution_left['theta']
            self.v_left = optimal_solution_left['theta_dot']
        except ValueError:
            self.get_logger().warning("not able to solve QP for left arm")

        # Solve NMPC for right arm
        t_left_nmpc = perf_counter_ns()
        try:
            optimal_solution_right = self.right_hand_controller.solve_nmpc(
            q[self.ainex_robot.right_arm_ids],
            refs_right.tolist()
        )
            self.q_right = optimal_solution_right['theta']
            self.v_right = optimal_solution_right['theta_dot']
        except ValueError:
            self.get_logger().warning("not able to solve QP for right arm")

        # Timing measurements
        t_right_nmpc = perf_counter_ns()
        solve_time_ns = t_right_nmpc - t_before_left_nmpc
        self._nmpc_solve_time_ns = solve_time_ns
        self._nmpc_solve_time_accum_ns += solve_time_ns
        self._nmpc_solve_time_samples += 1

        # Update Ainex robot with new joint commands
        self.ainex_robot.update(
            self.q_left, 
            self.q_right,
            self.v_left, 
            self.v_right,
            self.dt
        )
        t_update = perf_counter_ns()

        # Publish reference targets for visualization in RViz
        self.publish_nmpc_reference(
            [x_left, y_left, z_left],
            [x_right, y_right, z_right]
        )
        t_publish = perf_counter_ns()

        self.get_logger().info(
            f"Timing (ms): read_q={(t_read_q - t_before_read_q) / 1e6:.2f} left_nmpc={(t_left_nmpc - t_before_left_nmpc) / 1e6:.2f} right_nmpc={(t_right_nmpc - t_left_nmpc) / 1e6:.2f} update={(t_update - t_right_nmpc) / 1e6:.2f} publish={(t_publish - t_update) / 1e6:.2f} total={(t_publish - t_start) / 1e6:.2f}",
        )



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

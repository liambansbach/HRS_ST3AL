import time

import numpy as np
import pinocchio as pin
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from ainex_controller.ainex_model import AiNexModel
from ainex_controller.ainex_robot_pose_imitation import AinexRobot
from ainex_controller.nmpc_controller import NMPC

from ainex_interfaces.msg import RobotImitationTargets
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


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
        theta_min_right: Minimum angles allowed in right arm (default: [-2.09, -np.pi/2, -2.09, -1.9])
        theta_max_right: Maximum angles allowed in right arm (default: [2.09, np.pi/2, 2.09, 0.3])
        theta_min_left: Minimum angles allowed in left arm (default: [-2.09, -np.pi/2, -2.09, -1.9])
        theta_max_left: Maximum angles allowed in left arm (default: [2.09, np.pi/2, 2.09, 0.3])
"""
class ImitationControlNode(Node):
    def __init__(self):
        super().__init__('imitation_control_node')

        self.T_HORIZON_s = self.declare_parameter('T_HORIZON_s', 1).value #was 1 before testing around
        self.N = self.declare_parameter('N', 5).value #was 10 before testing around
        self.n_joints = self.declare_parameter('n_joints', 4).value
        # 7 reference values: [x_wrist, y_wrist, z_wrist, sho_pitch, sho_roll, el_pitch, el_yaw]
        self.n_ref_vals = self.declare_parameter('n_ref_vals', 7).value
        # Q weights: [x, y, z, sho_pitch, sho_roll, el_pitch, el_yaw]
        # Position tracking (high weight), joint angle tracking (lower weight as "loose target")
        self.Q_diag = self.declare_parameter('Q_diag', [100, 100, 100, 5, 5, 5, 5]).value
        self.R_diag = self.declare_parameter('R_diag', [0.1, 0.1, 0.1, 0.1]).value
        self.theta_dot_max = self.declare_parameter('theta_dot_max', 10).value

        """
        Testing Protocol of NMPC parameters for 3D version and N_targets = 4 (with gaming pc not real robot):
        - Start with T_HORIZON_s = 1s, N = 100 -> didnt work, robot didnt move 
        - T_HORIZON_s = 2s, N = 30 -> makes pc crash after some time? -> robot didnt move at the start?
        - T_HORIZON_s = 0.5s, N = 5 -> robot moved fast, but with some offsets? -> but seemed kind of ok
        - T_HORIZON_s = 0.2s, N = 5 -> seems to work aswell
        - T_HORIZON_s = 0.2s, N = 10 -> seems to work aswell, maybe a bit smoother?
        """ 
        """
        Testing Protocol of NMPC parameters for 3D version and N_targets = 7 (with gaming pc not real robot):
        - Start with T_HORIZON_s = 0.2s, N = 10 -> didnt work, robots arms are extended 
        - T_HORIZON_s = 2s, N = 0 -> robot jiggles a lot and arms dont really follow targets -> seems like transformations are weird or initial positions are off??
        - T_HORIZON_s = 1s, N = 20 -> destroyed PC after a few seconds (: -> too many steps?
        - T_HORIZON_s = 1s, N = 10 -> didnt crash and follows target but very weirdly with a big offset -> seems like transformations are weird are off?? -> this shouldnt be? -> seems like their is an offset of pi/2 for reachable points?
         """

        # Joint direction conventions (from URDF):
        # l_sho_roll is inverse to r_sho_roll due to mirroring -> +1 in r_sho_roll = -1 in l_sho_roll (both move arm downwards)
        # l_sho_pitch +1 = backwards, r_sho_pitch +1 = backwards -> same direction
        # l_el_pitch +1 = backwards, r_el_pitch +1 = backwards -> same direction  
        # l_el_yaw +1 = outward, r_el_yaw +1 = inward -> mirrored
        #
        # IMPORTANT: CasADi requires min <= max for all bounds!
        # The mirroring is handled by:
        #   1. Different homogeneous transform parameters for left/right (y-components have opposite signs)
        #   2. Negating the y-component of reference targets for mirrored behavior
        #
        # Joint order: [sho_pitch, sho_roll, el_pitch, el_yaw]
        # Using URDF limits directly (symmetric for both arms)
        self.theta_min_right = self.declare_parameter('theta_min_right', [-2.09, -2.09, -2.09, -1.9]).value
        self.theta_max_right = self.declare_parameter('theta_max_right', [2.09, 2.09, 2.09, 0.3]).value
        self.theta_min_left = self.declare_parameter('theta_min_left', [-2.09, -2.09, -2.09, -1.9]).value  # el_yaw mirrored: [-0.3, 1.9]
        self.theta_max_left = self.declare_parameter('theta_max_left', [2.09, 2.09, 2.09, 0.3]).value

        self.sim = self.declare_parameter('sim', True).value

        self.dt = self.T_HORIZON_s / self.N

        pkg = get_package_share_directory('ainex_description') # get package path
        self.urdf_path = pkg + '/urdf/ainex.urdf'
        self.robot_model = AiNexModel(self, self.urdf_path)
        
        self.get_logger().info("TESTING: Ainex Model loaded.")
        # Create AinexRobot instance
        self.ainex_robot = AinexRobot(self, self.robot_model, self.dt, sim=self.sim)
        self.get_logger().info("TESTING: Ainex Robot interface initialized.")
        # manual homogeneous transform parameters for FK (to be replaced by URDF-extracted version)
        # important NOTE: this seems to fit for 3D, but not 100% sure (got it by testing different values)
        # self.homogeneous_transform_params_left = {
        #     'T_0_1': ([0, 0, 0], 'y'),
        #     'T_1_2': ([0.02, 0.02151, 0], '-x'),
        #     'T_2_3': ([-0.02, 0.07411, 0], 'y'),
        #     'T_3_4': ([0.0004, 0.01702, 0.01907], '-z'),
        #     'T_4_wrist': ([0.01989, 0.0892, -0.019], 'non')}

        # self.homogeneous_transform_params_right = {
        #     'T_0_1': ([0, 0, 0], 'y'),
        #     'T_1_2': ([0.02, -0.02151, 0], '-x'),
        #     'T_2_3': ([-0.02, -0.07411, 0], '-y'),
        #     'T_3_4': ([0.0004, -0.01702, 0.01907], 'z'),
        #     'T_4_wrist': ([0.01989, -0.0892, -0.019], 'non')}

        # Print FK comparison: manual params vs URDF-extracted
        print("=== FK from URDF (Pinocchio numerical) ===")
        print("Left gripper at q=0:", self.get_gripper_position_in_shoulder_frame('left'))
        
        print("\n=== FK params extracted from URDF ===")
        left_params = self.extract_fk_params_from_urdf('left')
        print("Left arm parameters:")
        for key, val in left_params.items():
            print(f"  {key}: {val}")
        right_params = self.extract_fk_params_from_urdf('right')
        print("Right arm parameters:")
        for key, val in right_params.items():
            print(f"  {key}: {val}")

        """
        [ainex_imitation_control_node-4] === FK params extracted from URDF ===
        [ainex_imitation_control_node-4] Left arm parameters:
        [ainex_imitation_control_node-4]   T_0_1: ([-5.4627e-05, 0.052491, 0.087448], 'non')
        [ainex_imitation_control_node-4]   T_1_2: ([0.020003, 0.021507, 0.0], 'x')
        [ainex_imitation_control_node-4]   T_2_3: ([-0.020003, 0.074109, 0.0], 'non')
        [ainex_imitation_control_node-4]   T_3_4: ([0.00039506, 0.017019, 0.019072], 'z')
        [ainex_imitation_control_node-4]   T_4_wrist: ([0.0, 0.0, 0.0], 'non')
        [ainex_imitation_control_node-4] Right arm parameters:
        [ainex_imitation_control_node-4]   T_0_1: ([-5.46273713767254e-05, -0.0524909514103703, 0.0874480463620076], 'non')
        [ainex_imitation_control_node-4]   T_1_2: ([0.020003499818653, -0.021505961295104, 0.0], 'x')
        [ainex_imitation_control_node-4]   T_2_3: ([-0.0200034998200733, -0.0741099367526448, 0.0], 'non')
        [ainex_imitation_control_node-4]   T_3_4: ([0.000397813054181526, -0.0170183162715326, 0.0190701823981362], 'z')
        [ainex_imitation_control_node-4]   T_4_wrist: ([0.0, 0.0, 0.0], 'non')
        """

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
            None, #self.robot_model,
            self.ainex_robot.left_arm_ids, 
            "l_gripper_link",
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
            "r_gripper_link",
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
        


    def extract_fk_params_from_urdf(self, arm_side: str) -> dict:
        """
        Extract the transform parameters (translation, rotation axis) from URDF
        to build a symbolic CasADi FK function.
        
        Args:
            arm_side: 'left' or 'right'
        
        Returns:
            dict: Transform parameters for each joint in the arm chain
        """
        import casadi as ca
        
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
            """
            import pinocchio as pin
            
            joint = model.joints[joint_id]
            joint_type = joint.shortname()
            
            # Debug: print the actual shortname
            # print(f"Joint {model.names[joint_id]}: shortname = '{joint_type}'")
            
            # Method 1: Check shortname for common patterns
            if 'RX' in joint_type or 'RevoluteX' in joint_type:
                return 'x'
            elif 'RY' in joint_type or 'RevoluteY' in joint_type:
                return 'y'
            elif 'RZ' in joint_type or 'RevoluteZ' in joint_type:
                return 'z'
            
            # Method 2: For RevoluteUnaligned or other types, check the joint axis
            # by looking at the motion subspace (S matrix)
            try:
                # Get the joint's motion subspace at q=0
                q = np.zeros(model.nq)
                data = model.createData()
                pin.computeJointJacobians(model, data, q)
                
                # The joint's local motion subspace
                # For revolute joints, we look at which axis has rotation
                S = joint.jointConfigSelector(q[joint.idx_q:joint.idx_q + joint.nq])
                
                # Alternatively, check the joint axis directly if available
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
        
        # Define known axes from URDF (fallback based on joint naming convention)
        # From URDF: sho_pitch = Y axis, sho_roll = X axis, el_pitch = Y axis, el_yaw = Z axis
        known_axes = {
            'l_sho_pitch': 'y',   # axis="0 1 0" in URDF (left side)
            'l_sho_roll': 'x',    # axis="1 0 0" but rotates arm inward on left
            'l_el_pitch': '-y',    # axis="0 1 0" (left arm bends)
            'l_el_yaw': '-z',     # axis="0 0 -1" (left side)
            'r_sho_pitch': '-y',  # axis="0 -1 0" in URDF (right side)
            'r_sho_roll': 'x',    # axis="1 0 0"
            'r_el_pitch': 'y',   # axis="0 -1 0"
            'r_el_yaw': 'z',      # axis="0 0 1"
        }
        if self.sim:
            # In simulation, the left arm's sho_roll axis is inverted
            known_axes = {
                'l_sho_pitch': 'y',   # axis="0 1 0" in URDF (left side)
                'l_sho_roll': 'x',    # axis="1 0 0" but rotates arm inward on left
                'l_el_pitch': '-y',    # axis="0 1 0" (left arm bends)
                'l_el_yaw': '-z',     # axis="0 0 -1" (left side)
                'r_sho_pitch': '-y',  # axis="0 -1 0" in URDF (right side)
                'r_sho_roll': 'x',    # axis="1 0 0"
                'r_el_pitch': 'y',   # axis="0 -1 0"
                'r_el_yaw': 'z',      # axis="0 0 1"
            }        
        else: 
            # Real Robot:
            known_axes = {
            'l_sho_pitch': 'y',   # axis="0 1 0" in URDF (left side)
            'l_sho_roll': 'x',    # axis="1 0 0" but rotates arm inward on left
            'l_el_pitch': '-y',    # axis="0 1 0" (left arm bends)
            'l_el_yaw': '-z',     # axis="0 0 -1" (left side)
            'r_sho_pitch': '-y',  # axis="0 -1 0" in URDF (right side)
            'r_sho_roll': 'x',    # axis="1 0 0"
            'r_el_pitch': 'y',   # axis="0 -1 0"
            'r_el_yaw': 'z',      # axis="0 0 1"
            }
        params = {}
        for i, joint_name in enumerate(joint_names):
            joint_id = model.getJointId(joint_name)
            placement = model.jointPlacements[joint_id]
            translation = placement.translation.tolist()
            
            # Use known axes from URDF analysis
            axis = known_axes.get(joint_name, get_axis_string(model, joint_id))
            
            params[f'T_{i}_{i+1}'] = (translation, axis)
        
        # Get gripper frame placement relative to last joint
        if model.existFrame(gripper_frame):
            gripper_frame_id = model.getFrameId(gripper_frame)
            frame = model.frames[gripper_frame_id]
            gripper_translation = frame.placement.translation.tolist()
            params[f'T_{len(joint_names)}_wrist'] = (gripper_translation, 'non')
        
        return params
    
    def build_symbolic_fk_from_urdf(self, arm_side: str):
        """
        Build a symbolic CasADi forward kinematics function by extracting
        parameters from the URDF model.
        
        Args:
            arm_side: 'left' or 'right'
        
        Returns:
            ca.Function: Symbolic FK function theta -> [x, y, z]
        """
        import casadi as ca
        
        params = self.extract_fk_params_from_urdf(arm_side)
        
        def _get_homogeneous_transform(xyz, axis, angle):
            T = ca.SX.eye(4)
            T[0:3, 3] = xyz
            c = ca.cos(angle); s = ca.sin(angle)
            match axis:
                case 'x':  T[1,1]=c; T[1,2]=-s; T[2,1]= s; T[2,2]=c
                case '-x': T[1,1]=c; T[1,2]= s; T[2,1]=-s; T[2,2]=c
                case 'y':  T[0,0]=c; T[0,2]= s; T[2,0]=-s; T[2,2]=c
                case '-y': T[0,0]=c; T[0,2]=-s; T[2,0]= s; T[2,2]=c
                case 'z':  T[0,0]=c; T[0,1]=-s; T[1,0]= s; T[1,1]=c
                case '-z': T[0,0]=c; T[0,1]= s; T[1,0]=-s; T[1,1]=c
                case 'non': pass
            return T
        
        n_joints = 4
        theta = ca.SX.sym('theta', n_joints)
        
        T_total = ca.SX.eye(4)
        for i in range(n_joints):
            translation, axis = params[f'T_{i}_{i+1}']
            T_i = _get_homogeneous_transform(translation, axis, theta[i])
            T_total = ca.mtimes(T_total, T_i)
        
        # Add gripper offset (no rotation)
        translation, _ = params[f'T_{n_joints}_wrist']
        T_gripper = _get_homogeneous_transform(translation, 'non', 0)
        T_total = ca.mtimes(T_total, T_gripper)
        
        p_gripper = T_total[0:3, 3]
        
        return ca.Function(f'fk_{arm_side}', [theta], [p_gripper])

    # this should work aswell (T_world_sho_pitch)^-1 * T_world_wrist = T_sho_wrist 
    # pinocchio can calculate the Transformation to the world frame form the urdf file 
    def get_fk_shoulder_to_gripper(self, arm_side: str, q: np.ndarray = None) -> pin.SE3:
        """
        Compute forward kinematics from shoulder pitch frame to gripper frame.
        
        Args:
            arm_side: 'left' or 'right'
            q: Joint configuration (optional, uses current robot_model.q if None)
        
        Returns:
            pin.SE3: Transform from shoulder pitch to gripper (T_sho_gripper)
        """
        model = self.robot_model.model
        data = self.robot_model.data
        
        if q is None:
            q = self.robot_model.q
        
        # Update forward kinematics with current/given configuration
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        
        # Get frame names based on arm side
        if arm_side == 'left':
            sho_pitch_frame = 'l_sho_pitch_link'
            gripper_frame = 'l_gripper_link'
        elif arm_side == 'right':
            sho_pitch_frame = 'r_sho_pitch_link'
            gripper_frame = 'r_gripper_link'
        else:
            raise ValueError("arm_side must be 'left' or 'right'")
        
        # Get frame IDs
        sho_pitch_id = model.getFrameId(sho_pitch_frame)
        gripper_id = model.getFrameId(gripper_frame)
        
        # Get world-frame transforms
        T_world_sho = data.oMf[sho_pitch_id]      # World -> Shoulder Pitch
        T_world_gripper = data.oMf[gripper_id]    # World -> Gripper
        
        # Compute relative transform: T_sho_gripper = T_world_sho^-1 * T_world_gripper
        T_sho_gripper = T_world_sho.actInv(T_world_gripper)
        
        return T_sho_gripper
    
    def get_gripper_position_in_shoulder_frame(self, arm_side: str, q: np.ndarray = None) -> np.ndarray:
        """
        Get the gripper position (x, y, z) relative to shoulder pitch frame.
        
        Args:
            arm_side: 'left' or 'right'
            q: Joint configuration (optional)
        
        Returns:
            np.ndarray: [x, y, z] position of gripper in shoulder frame
        """
        T_sho_gripper = self.get_fk_shoulder_to_gripper(arm_side, q)
        return T_sho_gripper.translation.copy()
    
    def get_gripper_homogeneous_matrix(self, arm_side: str, q: np.ndarray = None) -> np.ndarray:
        """
        Get the 4x4 homogeneous transformation matrix from shoulder to gripper.
        
        Args:
            arm_side: 'left' or 'right'
            q: Joint configuration (optional)
        
        Returns:
            np.ndarray: 4x4 homogeneous transformation matrix
        """
        T_sho_gripper = self.get_fk_shoulder_to_gripper(arm_side, q)
        return T_sho_gripper.homogeneous

    def target_cb(self, msg: RobotImitationTargets):
        perf_counter_ns = self._perf_counter_ns
        t_start = perf_counter_ns()
        # Wrist position targets
        x_left = msg.wrist_target_left.x
        y_left = msg.wrist_target_left.y
        z_left = msg.wrist_target_left.z

        x_right = msg.wrist_target_right.x
        y_right = msg.wrist_target_right.y
        z_right = msg.wrist_target_right.z
        
        # Joint angle targets from arm vector angles:
        # sho_pitch ~ vertical angle of shoulder-elbow (elevation)
        # sho_roll ~ horizontal angle of shoulder-elbow (azimuth)  
        # el_pitch ~ vertical angle of elbow-wrist (elevation)
        # el_yaw ~ horizontal angle of elbow-wrist (azimuth)
        sho_pitch_target_left = msg.shoulder_pitch_target_left
        sho_roll_target_left = msg.shoulder_roll_target_left
        el_pitch_target_left = msg.elbow_pitch_target_left
        el_yaw_target_left = msg.elbow_yaw_target_left
    
        sho_pitch_target_right = msg.shoulder_pitch_target_right
        sho_roll_target_right = msg.shoulder_roll_target_right
        el_pitch_target_right = msg.elbow_pitch_target_right
        el_yaw_target_right = msg.elbow_yaw_target_right

        t_before_read_q = perf_counter_ns()
        # NOTE for reducing inference time: read only necessary joints? -> when do we need all joints?
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

        # test if any of the reference targets is inf or nan:
        if not np.all(np.isfinite(refs_left)) or not np.all(np.isfinite(refs_right)):
            self.get_logger().warn("Skipping NMPC step due to non-finite reference targets.")
            return

        # Solve NMPC for left and right arms
        t_before_left_nmpc = perf_counter_ns()
        optimal_solution_left = self.left_hand_controller.solve_nmpc(
            q[self.ainex_robot.left_arm_ids],
            refs_left.tolist()
        )
        t_left_nmpc = perf_counter_ns()
        optimal_solution_right = self.right_hand_controller.solve_nmpc(
            q[self.ainex_robot.right_arm_ids],
            refs_right.tolist()
        )
        t_right_nmpc = perf_counter_ns()
        solve_time_ns = t_right_nmpc - t_before_left_nmpc
        self._nmpc_solve_time_ns = solve_time_ns
        self._nmpc_solve_time_accum_ns += solve_time_ns
        self._nmpc_solve_time_samples += 1
        # NOTE: time to solve both NMPC is about 2-4 ms on robot, but for the first call it's around 600ms
        # -> hence this is probably not the bottleneck for real-time control

        # maybe TODO You could constain the  optimal_solution_left['theta'] and  optimal_solution_right['theta'] 
        # here in such a way that robot isnt able to reach behind himself?

        self.ainex_robot.update(
            optimal_solution_left['theta'], 
            optimal_solution_right['theta'],
            optimal_solution_left['theta_dot'], 
            optimal_solution_right['theta_dot'],
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

     

    def move_to_inital_position(self):
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
        
        :param s_ref_left: [x, y, z] wrist target position for left arm (relative to shoulder)
        :param s_ref_right: [x, y, z] wrist target position for right arm (relative to shoulder)
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

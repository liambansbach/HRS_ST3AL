from ainex_controller.ainex_model import AiNexModel
import numpy as np
import time
from rclpy.node import Node
from ainex_motion.joint_controller import JointController
from sensor_msgs.msg import JointState

"""
Important NOTE:
Seems like in here the left and right arms are swapped
"""

class AinexRobot():
    def __init__(self, node: Node, model: AiNexModel, dt: float, sim: bool = True):
        """ Visualize simulation and interface with real robot"""
        self.node = node
        self.sim = sim

        # Pinocchio model
        self.robot_model = model
        self.dt = dt
        self.model_joint_names = self.robot_model.pin_joint_names()
        self.node.get_logger().info(f"Joint names in Pinocchio model: {self.model_joint_names}")
        """
        Print Output:
        [ainex_imitation_control_node-4] Node names in Pinocchio model: ['head_pan', 'head_tilt', 'l_hip_yaw', 'l_hip_roll', 'l_hip_pitch', 'l_knee', 'l_ank_pitch', 
        'l_ank_roll', 'l_sho_pitch', 'l_sho_roll', 'l_el_pitch', 'l_el_yaw', 'l_gripper', 
        'r_hip_yaw', 'r_hip_roll', 'r_hip_pitch', 'r_knee', 'r_ank_pitch', 'r_ank_roll', 'r_sho_pitch', 'r_sho_roll', 'r_el_pitch', 'r_el_yaw', 'r_gripper']
        -> hence he reads out every joint
        """
        # Joint subset used for real robot I/O
        self.joint_names = [
            'l_sho_pitch', 'l_sho_roll', 'l_el_pitch', 'l_el_yaw',
            'r_sho_pitch', 'r_sho_roll', 'r_el_pitch', 'r_el_yaw']
        self.joint_name_to_model_idx = {
            name: self.robot_model.get_joint_id(name) for name in self.joint_names
        }
        # Get initial joint positions from the robot and convert to Pinocchio orders
        if self.sim:
            self.q = np.zeros(self.robot_model.model.nq)
        else:
            ## Joint controller interface with the robot
            self.joint_controller = JointController(self.node)
            self.q = self.read_joint_positions_from_robot()
            self.node.get_logger().info(f"q_init = {self.q}")
        # Initialize velocities to zero
        self.v = np.zeros(self.robot_model.model.nv)
        

        # update the model with initial positions and zero velocities
        self.robot_model.update_model(self.q, self.v)

        self.joint_states_pub = self.node.create_publisher(JointState, 'ainex_joint_states', 10)
        # publish initial joint states
        self.publish_joint_states()

        self.left_arm_ids = self.robot_model.get_arm_ids("left")
        self.right_arm_ids = self.robot_model.get_arm_ids("right")

    def _subset_from_full(self, q_full: np.ndarray) -> np.ndarray:
        return np.array([q_full[self.joint_name_to_model_idx[name]] for name in self.joint_names])

    def _full_from_subset(self, q_subset: np.ndarray, q_full: np.ndarray | None = None) -> np.ndarray:
        if q_full is None:
            q_full = np.zeros(self.robot_model.model.nq)
        for i, name in enumerate(self.joint_names):
            q_full[self.joint_name_to_model_idx[name]] = q_subset[i]
        return q_full

    def _apply_real_robot_corrections_subset(self, q_subset: np.ndarray, to_robot: bool) -> np.ndarray:
        """Apply joint direction/offset corrections on the subset array."""
        q = q_subset.copy()

        def idx(name: str) -> int | None:
            try:
                return self.joint_names.index(name)
            except ValueError:
                return None

        # l/r_sho_pitch has flipped direction in the real robot
        #for name in ('l_sho_pitch', 'r_sho_roll'): #'l_sho_pitch', 'l_sho_roll', 'l_sho_roll', 'l_el_pitch'
        for name in ('l_sho_pitch'): # 'l_el_pitch' defintly not flipped
            i = idx(name)
            if i is not None:
                q[i] *= -1.0

        # l/r_sho_roll has an offset in the real robot
        i = idx('r_sho_roll')
        if i is not None:
            q[i] += (-1.4 if to_robot else 1.4)
        i = idx('l_sho_roll')
        if i is not None:
            q[i] += (1.4 if to_robot else -1.4)

        return q

    def move_to_initial_position(self, q_init: np.ndarray = None):
        """Move robot to initial position."""
        self.q = q_init

        if not self.sim:
            self.send_cmd(self.q, 3.0)
            time.sleep(30.0)
        self.publish_joint_states()
        self.robot_model.update_model(self.q, self.v)
        
        self.node.get_logger().warn(f"Moved to q_init = {self.q}")

    def joint_states_from_model(self):
        """Get current joint states in pinocchio format."""
        return self.q, self.v
    
    def update(self, 
               q_cmd_left: np.ndarray, q_cmd_right: np.ndarray, 
               v_cmd_left: np.ndarray, v_cmd_right: np.ndarray, 
               dt: float):
        
        """Update the robot model with new desired velocities."""
        if q_cmd_left is not None:
            # inverte the shoulder roll 
            #'sho_pitch', 'sho_roll', 'el_yaw', 'el_pitch']
            # q_cmd_left = q_cmd_left.copy()
            # q_cmd_left[1] *= -1.0  # invert shoulder roll
            self.q[self.left_arm_ids] = q_cmd_left

            #self.node.get_logger().info(f"Left arm command q_cmd_left: {q_cmd_left}")
            #[ainex_imitation_control_node-4] [INFO] [1769771819.663722743] [imitation_control_node]: Left arm command q_cmd_left: [ 0.18340868  1.70535598 -2.09       -0.9022001 ]
        if q_cmd_right is not None:
            # invert shoulder pitch 
            # q_cmd_right = q_cmd_right.copy()
            # q_cmd_right[0] *= -1.0  # invert shoulder pitch
            self.q[self.right_arm_ids] = q_cmd_right
            #self.node.get_logger().info(f"Right arm command q_cmd_right: {q_cmd_right}")
            #[ainex_imitation_control_node-4] [INFO] [1769771819.664268198] [imitation_control_node]: Right arm command q_cmd_right: [-0.05243257  1.65989739 -0.10565359  0.3       ]

        if v_cmd_left is not None:
            self.v[self.left_arm_ids] = v_cmd_left
        if v_cmd_right is not None:
            self.v[self.right_arm_ids] = v_cmd_right
        # clip velocities to reasonable values
        # -> doing clipping in model update function now
        #self.v = np.clip(self.v, -5.0, 5.0)
        # visualize joint states in RViz
        self.publish_joint_states()

        self.robot_model.update_model(self.q, self.v)

        # send joint commands to the robot
        if not self.sim:
            self.send_cmd(self.q, dt)
    
    def send_cmd(self, q_cmd: np.ndarray, dt: float):
        """
        Send joint position commands to the robot.
        Args:
            q_cmd (np.ndarray): Desired joint positions in Pinocchio format.
        """
        q_subset = self._subset_from_full(q_cmd)
        #q_subset = self._apply_real_robot_corrections_subset(q_subset, to_robot=True)
        #self.joint_controller.setJointPositions(self.joint_names, q_subset.tolist(), dt, unit="rad")

        def idx(name: str) -> int | None:
            try:
                return self.joint_names.index(name)
            except ValueError:
                return None

        ## Adjust for real robot differences
        q_subset[idx('l_sho_pitch')] *= -1.0
        q_subset[idx('l_sho_roll')] *= -1.0
        q_subset[idx('l_el_pitch')] *= -1.0

        q_subset[idx('r_sho_roll')] *= -1.0
        
        q_subset[idx('l_sho_roll')] -= 1.4
        q_subset[idx('r_sho_roll')] += 1.4

        self.node.get_logger().info(f"{q_subset}")

        self.joint_controller.setJointPositions(self.joint_names, q_subset.tolist(), dt, unit="rad")

    # NOTE: this takes the most time when running on the real robot -> around 400ms - 600 ms, but sometimes also even more
    def read_joint_positions_from_robot(self):
        """Read joint states from the robot"""
        if self.sim:
            return self.joint_states_from_model()[0]

        #q_subset = np.array(self.joint_controller.getJointPositions(self.joint_names))
        #q_subset = self._apply_real_robot_corrections_subset(q_subset, to_robot=False)
        #self.node.get_logger().info(f"q_subset{q_subset}")

        q_subset = np.array(self.joint_controller.getJointPositions(self.joint_names))
        def idx(name: str) -> int | None:
            try:
                return self.joint_names.index(name)
            except ValueError:
                return None
            
        q_subset[idx('l_sho_pitch')] *= -1.0
        q_subset[idx('r_sho_pitch')] *= -1.0

         # l/r_sho_roll has an offset in the real robot
        q_subset[idx('r_sho_roll')] += 1.4
        q_subset[idx('l_sho_roll')] -= 1.4

        return self._full_from_subset(q_subset)
   
    def publish_joint_states(self):
        """Publish current joint states."""
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.node.get_clock().now().to_msg()
        joint_state_msg.name = self.joint_names
        joint_state_msg.position = self._subset_from_full(self.q).tolist()
        joint_state_msg.velocity = self._subset_from_full(self.v).tolist()
        self.joint_states_pub.publish(joint_state_msg)

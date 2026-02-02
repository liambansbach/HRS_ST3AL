"""
Ainex Robot Pose Imitation Interface.

This module provides the glue between the Pinocchio model and the AiNEX hardware
interface (or simulator). It handles joint-state I/O, publishes JointState messages,
and sends position commands for imitation control.

Key Features:
    - **Pinocchio Model Bridge**: Maintains full-state (q, v) in Pinocchio format and
      updates the model after each command.
    - **Real Robot I/O**: Converts between Pinocchio joint ordering and the robot's
      joint subset and applies joint sign/offset corrections.
    - **Simulation Mode**: Runs without hardware I/O while still publishing joint states
      for visualization and downstream nodes.

Dependencies:
    - numpy: Array operations and state storage.
    - rclpy: ROS 2 node and publisher utilities.
    - sensor_msgs/JointState: Joint state message publishing.
    - ainex_motion: Joint controller interface for hardware commands.

Classes:
    AinexRobot: Manages joint-state synchronization and command I/O for AiNEX.
"""

from ainex_controller.ainex_model import AiNexModel
import numpy as np
import time
from rclpy.node import Node
from ainex_motion.joint_controller import JointController
from sensor_msgs.msg import JointState

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

        # Get arm joint indices for easier access
        self.left_arm_ids = self.robot_model.get_arm_ids("left")
        self.right_arm_ids = self.robot_model.get_arm_ids("right")

    def _subset_from_full(self, q_full: np.ndarray) -> np.ndarray:
        """Extract subset of joint positions for the real robot from full Pinocchio format.
        Args:
            q_full (np.ndarray): Full joint positions in Pinocchio format.
        Returns:
            np.ndarray: Joint positions subset for the real robot.
        """
        return np.array([q_full[self.joint_name_to_model_idx[name]] for name in self.joint_names])

    def _full_from_subset(self, q_subset: np.ndarray, q_full: np.ndarray | None = None) -> np.ndarray:
        """Convert subset of joint positions from real robot to full Pinocchio format.
        Args:
            q_subset (np.ndarray): Joint positions subset from the real robot.
            q_full (np.ndarray | None): Optional full joint positions to update. If None, a new array is created.
        Returns:
            np.ndarray: Full joint positions in Pinocchio format."""
        if q_full is None:
            q_full = np.zeros(self.robot_model.model.nq)
        for i, name in enumerate(self.joint_names):
            q_full[self.joint_name_to_model_idx[name]] = q_subset[i]
        return q_full

    def _apply_real_robot_corrections_subset(self, q_subset: np.ndarray, to_robot: bool) -> np.ndarray:
        """Apply joint direction/offset corrections on the subset array.
        Args:
            q_subset (np.ndarray): Joint positions subset.
            to_robot (bool): If True, convert from Pinocchio to real robot format. If False, convert from real robot to Pinocchio format.
        Returns:
            np.ndarray: Corrected joint positions subset.
        """
        q = q_subset.copy()

        def idx(name: str) -> int | None:
            try:
                return self.joint_names.index(name)
            except ValueError:
                return None
        for name in ('l_sho_pitch'): 
            i = idx(name)
            if i is not None:
                q[i] *= -1.0
        i = idx('r_sho_roll')
        if i is not None:
            q[i] += (-1.4 if to_robot else 1.4)
        i = idx('l_sho_roll')
        if i is not None:
            q[i] += (1.4 if to_robot else -1.4)
        return q

    def move_to_initial_position(self, q_init: np.ndarray = None):
        """Move robot to initial position.
        Args:
            q_init (np.ndarray): Initial joint positions in Pinocchio format.
        Returns:
            None
        """
        self.q = q_init
        if not self.sim:
            self.send_cmd(self.q, 3.0)
            time.sleep(30.0)
        self.publish_joint_states()
        self.robot_model.update_model(self.q, self.v)
        self.node.get_logger().warn(f"Moved to q_init = {self.q}")

    def joint_states_from_model(self):
        """Get current joint states in pinocchio format.
        Returns:
            Tuple[np.ndarray, np.ndarray]: Current joint positions and velocities.
        """
        return self.q, self.v
    
    def update(self, 
               q_cmd_left: np.ndarray, q_cmd_right: np.ndarray, 
               v_cmd_left: np.ndarray, v_cmd_right: np.ndarray, 
               dt: float):
        
        """Update the robot model with new desired velocities.
        Args:
            q_cmd_left (np.ndarray): Desired joint positions for the left arm.
            q_cmd_right (np.ndarray): Desired joint positions for the right arm.
            v_cmd_left (np.ndarray): Desired joint velocities for the left arm.
            v_cmd_right (np.ndarray): Desired joint velocities for the right arm.
            dt (float): Time step for the update.
        Returns:
            None"""
        if q_cmd_left is not None:
            self.q[self.left_arm_ids] = q_cmd_left

        if q_cmd_right is not None:
            self.q[self.right_arm_ids] = q_cmd_right

        if v_cmd_left is not None:
            self.v[self.left_arm_ids] = v_cmd_left
        if v_cmd_right is not None:
            self.v[self.right_arm_ids] = v_cmd_right
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
            dt (float): Time step for the command.
        Returns:
            None
        """
        q_subset = self._subset_from_full(q_cmd)

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
        """Read joint states from the robot
        Args:
            None
        Returns:
            np.ndarray: Current joint positions in full robot format.
        """
        if self.sim:
            return self.joint_states_from_model()[0]

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
        """Publish current joint states.
        Args:
            None
        Returns:
            None
        """
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.node.get_clock().now().to_msg()
        joint_state_msg.name = self.joint_names
        joint_state_msg.position = self._subset_from_full(self.q).tolist()
        joint_state_msg.velocity = self._subset_from_full(self.v).tolist()
        self.joint_states_pub.publish(joint_state_msg)

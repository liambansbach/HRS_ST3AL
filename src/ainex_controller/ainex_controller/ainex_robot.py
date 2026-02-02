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
        self.joint_names = self.robot_model.pin_joint_names()

        # Get initial joint positions from the robot and convert to Pinocchio order
        if self.sim:
            self.q = np.zeros(self.robot_model.model.nq)
        else:
            ## Joint controller interface with the robot
            self.joint_controller = JointController(self.node)
            self.q = self.read_joint_positions_from_robot()
            self.node.get_logger().warn(f"q_init = {self.q}")
        # Initialize velocities to zero
        self.v = np.zeros(self.robot_model.model.nv)

        # update the model with initial positions and zero velocities
        self.robot_model.update_model(self.q, self.v)

       # self.joint_states_pub = self.node.create_publisher(JointState, 'ainex_joint_states', 10)
        topic = 'joint_states' if self.sim else 'ainex_joint_states'
        self.joint_states_pub = self.node.create_publisher(JointState, topic, 10)

        # publish initial joint states
        self.publish_joint_states()

        #self.left_arm_ids = self.robot_model.get_arm_ids("left")
        #self.right_arm_ids = self.robot_model.get_arm_ids("right")

        self.left_arm_q_ids  = self.robot_model.get_arm_ids("left")
        self.right_arm_q_ids = self.robot_model.get_arm_ids("right")

        self.left_arm_v_ids  = self.robot_model.get_arm_v_ids("left")
        self.right_arm_v_ids = self.robot_model.get_arm_v_ids("right")


    def move_to_initial_position(self, q_init: np.ndarray = None):
        """Move robot to initial position."""
        self.q = q_init
        self.node.get_logger().warn(f"Moved to q_init = {self.q}")

        if not self.sim:
            self.send_cmd(self.q, 5.0)
            time.sleep(5.0)
        self.publish_joint_states()
        self.robot_model.update_model(self.q, self.v)
        
    def joint_states_from_model(self):
        """Get current joint states in pinocchio format."""
        return self.q, self.v
    
    def update(self, v_cmd_left: np.ndarray, v_cmd_right: np.ndarray, dt: float):
        """Update the robot model with new desired velocities."""
        # if v_cmd_left is not None:
        #     self.v[self.left_arm_ids] = v_cmd_left
        # if v_cmd_right is not None:
        #     self.v[self.right_arm_ids] = v_cmd_right
        # self.q += self.v * dt

        if v_cmd_left is not None:
            self.v[self.left_arm_v_ids] = v_cmd_left
        if v_cmd_right is not None:
            self.v[self.right_arm_v_ids] = v_cmd_right
        self.q += self.v * dt


        self.robot_model.update_model(self.q, self.v)
        
        # visualize joint states in RViz
        self.publish_joint_states()

        # send joint commands to the robot
        if not self.sim:
            self.send_cmd(self.q, dt)
    
    def send_cmd(self, q_cmd: np.ndarray, dt: float):
        """
        Send joint position commands to the robot.
        Args:
            q_cmd (np.ndarray): Desired joint positions in Pinocchio format.
        """
        q_cmd = q_cmd.copy()

        ## Adjust for real robot differences
        # l/r_sho_pitch has flipped direction in the real robot
        
        q_cmd[self.robot_model.get_joint_id('l_sho_pitch')] *= -1.0
        q_cmd[self.robot_model.get_joint_id('r_sho_pitch')] *= 1.0

        q_cmd[self.robot_model.get_joint_id('l_el_pitch')] *= -1.0
        q_cmd[self.robot_model.get_joint_id('r_el_pitch')] *= 1.0

        # q_cmd[self.robot_model.get_joint_id('l_el_yaw')] *= 1.0
        # q_cmd[self.robot_model.get_joint_id('r_el_yaw')] *= 1.0
        
        # l/r_sho_roll has an offset in the real robot
        
        q_cmd[self.robot_model.get_joint_id('r_sho_roll')] *= -1.0
        q_cmd[self.robot_model.get_joint_id('l_sho_roll')] *= -1.0

        q_cmd[self.robot_model.get_joint_id('r_sho_roll')] += 1.45
        q_cmd[self.robot_model.get_joint_id('l_sho_roll')] -= 1.45

        
        self.joint_controller.setJointPositions(self.joint_names, q_cmd.tolist(), dt, unit="rad")

    def read_joint_positions_from_robot(self):
        """Read joint states from the robot"""
        q_real = np.array(self.joint_controller.getJointPositions(self.joint_names))

        ## Adjust for real robot differences
        # l/r_sho_pitch has flipped direction in the real robot

        # q_real[self.robot_model.get_joint_id('l_sho_pitch')] *= 1.0
        # q_real[self.robot_model.get_joint_id('r_sho_pitch')] *= -1.0

        # q_real[self.robot_model.get_joint_id('l_el_pitch')] *= 1.0
        # q_real[self.robot_model.get_joint_id('r_el_pitch')] *= -1.0      
        # q_real[self.robot_model.get_joint_id('l_el_yaw')] *= 1.0
        # q_real[self.robot_model.get_joint_id('r_el_yaw')] *= 1.0
        
        # # l/r_sho_roll has an offset in the real robot
        
        
        # q_real[self.robot_model.get_joint_id('r_sho_roll')] -= 1.45
        # q_real[self.robot_model.get_joint_id('l_sho_roll')] += 1.45

        return q_real
    
    def publish_joint_states(self):
        """Publish current joint states."""
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.node.get_clock().now().to_msg()
        joint_state_msg.name = self.joint_names
        joint_state_msg.position = self.q.tolist()
        joint_state_msg.velocity = self.v.tolist()
        self.joint_states_pub.publish(joint_state_msg)

    def set_grippers(self, l_pos: float = None, r_pos: float = None, duration: float = 1.0):
        """
        Set gripper joint positions in *model space* (URDF/pin order).
        Works in sim (updates q + publishes joint_states) and real (also sends cmd).
        """
        l_gripper_id = self.robot_model.get_joint_id("l_gripper")
        r_gripper_id = self.robot_model.get_joint_id("r_gripper")

        if l_pos is not None:
            self.q[l_gripper_id] = float(l_pos)
        if r_pos is not None:
            self.q[r_gripper_id] = float(r_pos)

        # keep model consistent
        self.robot_model.update_model(self.q, self.v)
        self.publish_joint_states()

        if not self.sim:
            # send full-body q command (includes grippers)
            self.send_cmd(self.q, duration)

    def open_hand(self, which: str = "both", duration: float = 1.0):
        # NOTE: choose values that match your robot conventions
        if which == "both":
            self.set_grippers(l_pos=-1.5, r_pos=1.5, duration=duration)
        elif which == "left":
            self.set_grippers(l_pos=-1.5, r_pos=None, duration=duration)
        elif which == "right":
            self.set_grippers(l_pos=None, r_pos=1.5, duration=duration)
        else:
            self.node.get_logger().warn(f"open_hand: unknown which='{which}'")

    def close_hand(self, which: str = "both", duration: float = 1.0):
        # NOTE: pick “closed” values; placeholders:
        if which == "both":
            self.set_grippers(l_pos=-0.4, r_pos=0.4, duration=duration)
        elif which == "left":
            self.set_grippers(l_pos=-0.4, r_pos=None, duration=duration)
        elif which == "right":
            self.set_grippers(l_pos=None, r_pos=0.4, duration=duration)
        else:
            self.node.get_logger().warn(f"close_hand: unknown which='{which}'")



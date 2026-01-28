import rclpy
from ament_index_python.packages import get_package_share_directory

import pinocchio as pin
import numpy as np
from rclpy.node import Node

from ainex_controller.ainex_model import AiNexModel
from ainex_controller.ainex_robot import AinexRobot
from ainex_controller.ainex_hand_controller import HandController

from tf2_ros import TransformListener, Buffer, TransformBroadcaster


class StackCubesNode(Node):
    def __init__(self):
        super().__init__("stack_cubes")

        self.dt = 0.05 # 50 ms == 20 Hz

        # Initialize robot model
        # get package path
        self.pkg = get_package_share_directory('ainex_description')
        self.urdf_path = self.pkg + '/urdf/ainex.urdf'
        self.robot_model = AiNexModel(self, self.urdf_path)

        # Create AinexRobot instance
        # TODO: set sim=False when interfacing with real robot
        # IMPORTANT !!!: Always test first in simulation to avoid damage to the real robot!!!
        self.ainex_robot = AinexRobot(self, self.robot_model, self.dt, sim=True)

        # Create HandController instances for left and right hands
        self.left_hand_controller = HandController(self, self.robot_model, arm_side='left')
        self.right_hand_controller = HandController(self, self.robot_model, arm_side='right')

        self.v_cmd_left = None
        self.v_cmd_right = None

        self.create_timer(self.dt, self.main_loop) # 50 Hz 

        # flags
        self.execute_once = True


    # -----------------------------
    # Main loop
    # -----------------------------
    def main_loop(self):
        while rclpy.ok():
            if self.execute_once:
                self.move_to_initial_position()
                self.execute_once = False
                
            self.v_cmd_left = self.left_hand_controller.update(self.dt)
            self.v_cmd_right = self.right_hand_controller.update(self.dt)
            self.ainex_robot.update(self.v_cmd_left, self.v_cmd_right, self.dt)
            rclpy.spin_once(self, timeout_sec=self.dt)

    def move_to_initial_position(self):
        q_init = np.zeros(self.robot_model.model.nq)
        # Home position defined in urdf/pinocchio model
        # TODO: feel free to change to other initial positions away from singularities
        q_init[self.robot_model.get_joint_id('r_sho_roll')] = 1.4
        q_init[self.robot_model.get_joint_id('l_sho_roll')] = -1.4
        q_init[self.robot_model.get_joint_id('r_el_yaw')] = 1.58
        q_init[self.robot_model.get_joint_id('l_el_yaw')] = -1.58
        # Move robot to initial position
        self.ainex_robot.move_to_initial_position(q_init)
        # TODO: Feel free to change to other target poses for testing

        # left hand target pose
        left_target = pin.SE3.Identity()
        left_target.translation = np.array([0.0, 0.03, 0.0])  # Move 3 cm forward
        self.left_hand_controller.set_target_pose(left_target, duration=3.0, type='rel')

        # right hand target pose
        right_current = self.robot_model.right_hand_pose()
        right_target = right_current.copy()
        right_target.translation[2] += 0.02  # Move up by 2 cm 
        self.right_hand_controller.set_target_pose(right_target, duration=3.0, type='abs')


def main():
    rclpy.init()
    node = StackCubesNode()
    try:
        #rclpy.spin(node)
        node.main_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

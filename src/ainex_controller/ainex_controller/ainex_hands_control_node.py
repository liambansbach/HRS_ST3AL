import rclpy
from ament_index_python.packages import get_package_share_directory

import pinocchio as pin
import numpy as np

from ainex_controller.ainex_model import AiNexModel
from ainex_controller.ainex_robot import AinexRobot
from ainex_controller.ainex_hand_controller import HandController
from ainex_controller.ainex_hand_controller_extended import HandController as HandControllerExtended


def main():
    rclpy.init()
    node = rclpy.create_node('ainex_hands_control_node')

    dt = 0.05

    # Initialize robot model
    # get package path
    pkg = get_package_share_directory('ainex_description')
    urdf_path = pkg + '/urdf/ainex.urdf'
    robot_model = AiNexModel(node, urdf_path)

    # Create AinexRobot instance
    # TODO: set sim=False when interfacing with real robot
    # IMPORTANT !!!: Always test first in simulation to avoid damage to the real robot!!!
    ainex_robot = AinexRobot(node, robot_model, dt, sim=False)

    q_init = np.zeros(robot_model.model.nq)
    # Home position defined in urdf/pinocchio model
    # TODO: feel free to change to other initial positions away from singularities
    q_init[robot_model.get_joint_id('r_sho_pitch')] = 1.15
    q_init[robot_model.get_joint_id('l_sho_pitch')] = 1.15
    q_init[robot_model.get_joint_id('r_sho_roll')] = 0.6
    q_init[robot_model.get_joint_id('l_sho_roll')] = -0.6
    q_init[robot_model.get_joint_id('r_el_yaw')] = 0.8
    q_init[robot_model.get_joint_id('l_el_yaw')] = -0.8
    q_init[robot_model.get_joint_id('r_el_pitch')] = -1.57
    q_init[robot_model.get_joint_id('l_el_pitch')] = -1.57
    # Move robot to initial position
    ainex_robot.move_to_initial_position(q_init)

    # Create HandController instances for left and right hands
    #left_hand_controller = HandController(node, robot_model, arm_side='left')
    #right_hand_controller = HandController(node, robot_model, arm_side='right')
    left_hand_controller = HandControllerExtended(
        node, 
        robot_model, 
        arm_side="left",
        enable_nullspace=True,     # true for better singularity handling // false is the old simple controller
        k_null=0.6,                # <--- nullspace strength
        adaptive_damping=True,
        hard_stop_on_singularity=False,
    )

    right_hand_controller = HandControllerExtended(
        node, 
        robot_model, 
        arm_side="right",
        enable_nullspace=True,     # true for better singularity handling // false is the old simple controller
        k_null=0.6,                # <--- nullspace strength
        adaptive_damping=True,
        hard_stop_on_singularity=False,
    )

    # TODO: Feel free to change to other target poses for testing

    # left hand target pose
    left_target = pin.SE3.Identity()
    left_target.translation = np.array([0.08, -0.04, 0.05])  # Move 3 cm forward
    left_hand_controller.set_target_pose(left_target, duration=3.0, type='abs')

    # # right hand target pose
    # right_current = robot_model.right_hand_pose()
    # right_target = right_current.copy()
    # right_target.translation[2] += 0.0  # Move up by 2 cm 
    # right_hand_controller.set_target_pose(right_target, duration=3.0, type='abs')

    # right hand target pose
    right_target = pin.SE3.Identity()
    right_target.translation = np.array([0.1, -0.15, 0.0])  # Move up by 2 cm :: z= nach oben, y= zur seite, x= nach vorne
    right_hand_controller.set_target_pose(right_target, duration=3.0, type='abs')

    v_cmd_left = None
    v_cmd_right = None
    while rclpy.ok():
        v_cmd_left = left_hand_controller.update(dt)
        v_cmd_right = right_hand_controller.update(dt)
        ainex_robot.update(v_cmd_left, v_cmd_right, dt)
        rclpy.spin_once(node, timeout_sec=dt)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
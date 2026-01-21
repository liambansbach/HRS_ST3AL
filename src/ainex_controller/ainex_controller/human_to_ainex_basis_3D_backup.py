#!/usr/bin/env python3

""" 
    Calculates wrist position target and elbow angle target for the Ainex robot
      based on human pose estimation.
    Focuses on mapping 2D pose estimation to the robot, as depth estimation is considered highly uncertain.

    HRV => Human Reach Vector, vector from humans shoulder to the wrist
    HOA => Human Over Arm, segment between shoulder and elbow
    HOA => Human Under Arm, segment between elbow and wrist
"""

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, TransformStamped
from tf2_ros import TransformBroadcaster

from ainex_interfaces.msg import UpperbodyPose, RobotImitationTargets




class HumanToAinex(Node):
    def __init__(self):
        super().__init__('human_to_ainex')

        self.br = TransformBroadcaster(self)

        self.bodypose_sub = self.create_subscription(
            UpperbodyPose,
            "/mp_pose/upper_body_rig",
            self.bodypose_cb,
            10
        )

        self.robot_targets_pub = self.create_publisher(
            RobotImitationTargets,
            "/robot_imitation_targets",
            10
        )

        self.robot_full_reach_length = 0.187 
        self.get_logger().info('Human to Robot basis tranformation node started! Listening to "/mp_pose/upper_body_rig"')

    def bodypose_cb(self, msg: UpperbodyPose):
        """ 
            Callback for upperbody pose subscriber
            Also calculates and publisher robot targets for each message
        """
        self.left_shoulder = msg.left_shoulder 
        self.left_elbow = msg.left_elbow   
        self.left_wrist = msg.left_wrist   
       
        self.right_shoulder = msg.right_shoulder 
        self.right_elbow = msg.right_elbow   
        self.right_wrist = msg.right_wrist   
        self.publish_robot_targets()

    def publish_robot_targets(self):
        """     
            Assigns values to msg for publishing based on wirst and elbow target calculations 
        """
        wrist_target_left, wrist_target_right = self.robot_wrist_target()
        angle_left_elbow, angle_right_elbow = self.robot_elbow_angle_target()
        
        msg = RobotImitationTargets()

        self.visualize_targets(wrist_target_left, "left")
        self.visualize_targets(wrist_target_right, "right")

        msg.wrist_target_left = wrist_target_left
        msg.wrist_target_right = wrist_target_right
        msg.angle_left_elbow = float(angle_left_elbow)
        msg.angle_right_elbow = float(angle_right_elbow)

        self.robot_targets_pub.publish(msg)

    def robot_wrist_target(self):
        """ 
            Calculates target position for robot wrist by mapping from humans full reach to robots full reach
            Uses human direction from shoulder to wrist and a reaching factor [0-1] to determine robot wrist target
            
            3D coordinate mapping (MediaPipe image coords -> Robot coords):
                robot.x = human.z (depth/forward)
                robot.y = human.x (left/right)
                robot.z = -human.y (up/down, inverted)
        """

        """ Left """
        HRV_left = np.array([
            self.left_wrist.x - self.left_shoulder.x,
            self.left_wrist.y - self.left_shoulder.y,
            self.left_wrist.z - self.left_shoulder.z
        ])
        HRV_left_len = np.linalg.norm(HRV_left)
        HRV_left_unit = HRV_left / HRV_left_len

        HOA_left = np.array([
            self.left_elbow.x - self.left_shoulder.x,
            self.left_elbow.y - self.left_shoulder.y,
            self.left_elbow.z - self.left_shoulder.z
        ])
        HUA_left = np.array([
            self.left_wrist.x - self.left_elbow.x,
            self.left_wrist.y - self.left_elbow.y,
            self.left_wrist.z - self.left_elbow.z
        ])

        reaching_factor_left = HRV_left_len / (np.linalg.norm(HOA_left) + np.linalg.norm(HUA_left))

        full_reach_reaching_direction_left = HRV_left_unit * self.robot_full_reach_length
        wtl = full_reach_reaching_direction_left * reaching_factor_left
        wrist_target_left = Point()
        wrist_target_left.x = -wtl[2]   # depth (MediaPipe z -> Robot x)
        wrist_target_left.y = wtl[0]   # left/right (MediaPipe x -> Robot y)
        wrist_target_left.z = -wtl[1]  # up/down (MediaPipe -y -> Robot z)

        """ Right """
        HRV_right = np.array([
            self.right_wrist.x - self.right_shoulder.x,
            self.right_wrist.y - self.right_shoulder.y,
            self.right_wrist.z - self.right_shoulder.z
        ])
        HRV_right_len = np.linalg.norm(HRV_right)
        HRV_right_unit = HRV_right / HRV_right_len

        HOA_right = np.array([
            self.right_elbow.x - self.right_shoulder.x,
            self.right_elbow.y - self.right_shoulder.y,
            self.right_elbow.z - self.right_shoulder.z
        ])
        HUA_right = np.array([
            self.right_wrist.x - self.right_elbow.x,
            self.right_wrist.y - self.right_elbow.y,
            self.right_wrist.z - self.right_elbow.z
        ])

        reaching_factor_right = HRV_right_len / (np.linalg.norm(HOA_right) + np.linalg.norm(HUA_right))

        full_reach_reaching_direction_right = HRV_right_unit * self.robot_full_reach_length
        wtr = full_reach_reaching_direction_right * reaching_factor_right
        wrist_target_right = Point()
        wrist_target_right.x = -wtr[2]   # depth (MediaPipe z -> Robot x)
        wrist_target_right.y = wtr[0]   # left/right (MediaPipe x -> Robot y)
        wrist_target_right.z = -wtr[1]  # up/down (MediaPipe -y -> Robot z)

        return wrist_target_left, wrist_target_right

    def robot_elbow_angle_target(self):
        """ 
            Calculates the angle of the human elbow
            Robot will target the same angle, but as a secondary optimization condition.
        """

        """ Left """
        HOA_left = np.array([self.left_shoulder.x - self.left_elbow.x, self.left_shoulder.y -self.left_elbow.y])
        HUA_left = np.array([self.left_wrist.x - self.left_elbow.x, self.left_wrist.y - self.left_elbow.y])
        HOA_len_left = np.linalg.norm(HOA_left)
        HUA_len_left = np.linalg.norm(HUA_left)

        angle_left_elbow = np.arccos(np.dot(HOA_left, HUA_left) / (HOA_len_left * HUA_len_left))

        """ Right """
        HOA_right = np.array([self.right_shoulder.x - self.right_elbow.x, self.right_shoulder.y -self.right_elbow.y])
        HUA_right = np.array([self.right_wrist.x - self.right_elbow.x, self.right_wrist.y - self.right_elbow.y])
        HOA_len_right = np.linalg.norm(HOA_right)
        HUA_len_right = np.linalg.norm(HUA_right)

        angle_right_elbow = np.arccos(np.dot(HOA_right, HUA_right) / (HOA_len_right * HUA_len_right))

        return angle_left_elbow, angle_right_elbow
    
    def visualize_targets(self, xyz, side):
        """ 
            Visualizes the detected wrist positions and elbow angles in TF
        """
        """ # Nimm den ersten Marker (falls du mehrere willst, musst du hier erweitern)
        tvec = self.tvecs.astype(float)   # [tx, ty, tz] im OpenCV-Kameraframe
        rvec = self.rvecs.astype(float)
        # === Low-pass Filter auf Translation ===
        if self.last_tvec is None:
            tvec_f = tvec
        else:
            tvec_f = self.alpha_pos * tvec + (1.0 - self.alpha_pos) * self.last_tvec
        self.last_tvec = tvec_f

        # === Low-pass Filter auf Rotation (auf rvec) ===
        if self.last_rvec is None:
            rvec_f = rvec
        else:
            rvec_f = self.alpha_rot * rvec + (1.0 - self.alpha_rot) * self.last_rvec
        self.last_rvec = rvec_f """

        # === Koordinatensystem-Anpassung ===
        tx = xyz.x
        ty = xyz.y
        tz = xyz.z

        t_msg = TransformStamped()
        t_msg.header.stamp = self.get_clock().now().to_msg()
        match side:
            case "left":
                t_msg.header.frame_id = "l_sho_pitch_link"
                t_msg.child_frame_id = "wrist_left_target_rviz"
            case "right": 
                t_msg.header.frame_id = "r_sho_pitch_link"
                t_msg.child_frame_id = "wrist_right_target_rviz"


        t_msg.transform.translation.x = float(tx)    # vor der Kamera
        t_msg.transform.translation.y = float(ty)   # links/rechts
        t_msg.transform.translation.z = float(tz)   # hoch/runter

        # Rotation aus gefiltertem rvec
        #R_cv, _ = cv2.Rodrigues(rvec_f_i)
        #r = R.from_matrix(R_cv)
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0 # r.as_quat()  # [x, y, z, w]

        t_msg.transform.rotation.x = float(qx)
        t_msg.transform.rotation.y = float(qy)
        t_msg.transform.rotation.z = float(qz)
        t_msg.transform.rotation.w = float(qw)

        # TF senden
        self.br.sendTransform(t_msg)
    
def main(args=None):
    rclpy.init(args=args)
    node = HumanToAinex()
    node.get_logger().info('HumanToAinex running')

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

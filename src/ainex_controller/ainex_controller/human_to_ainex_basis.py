#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, TransformStamped
from tf2_ros import TransformBroadcaster

from ainex_interfaces.msg import UpperbodyPose, RobotImitationTargets

"""
Human to AiNEX Robot Motion Retargeting Node
============================================

This ROS 2 node acts as a bridge between human pose estimation (specifically from 
MediaPipe) and the AiNEX robot's control system. It performs real-time motion 
retargeting, converting normalized human upper-body coordinates into specific 
joint targets and Cartesian wrist coordinates for the robot.

Key Features:
-------------
1.  **Coordinate Frame Transformation**: Maps MediaPipe camera coordinates to the 
    AiNEX robot frame:
    - Robot X (Forward) = -Human Z
    - Robot Y (Left)    =  Human X
    - Robot Z (Up)      = -Human Y

2.  **Reach Scaling**: Scales human arm movements to the robot's physical dimensions 
    (0.187m max reach), calculating a "reaching factor" based on how fully extended 
    the human arm is.

3.  **Analytical Inverse Kinematics**: Computes specific joint angles (Shoulder Pitch/Roll, 
    Elbow Pitch/Yaw) based on the geometric vectors between shoulder, elbow, and wrist.

ROS 2 Interface:
----------------
**Node Name:** `human_to_ainex`

**Subscribers:**
  - `/mp_pose/upper_body_rig` (`ainex_interfaces/UpperbodyPose`): 
    Real-time 3D coordinates of the human shoulders, elbows, and wrists.

**Publishers:**
  - `/robot_imitation_targets` (`ainex_interfaces/RobotImitationTargets`): 
    Calculated joint angles and Cartesian wrist targets for the robot controller.

**Broadcasters:**
  - `tf2_ros.TransformBroadcaster`: 
    Broadcasts visualization frames (`wrist_left_target_rviz`, `wrist_right_target_rviz`) 
    relative to the robot's shoulder links for debugging in Rviz.

"""

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
        self._eps = 1e-6

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
            Assigns values to msg for publishing based on 
            wirst coordinate and joint target calculations 
        """
        wrist_target_left = self.robot_wrist_target_left(self.left_shoulder, self.left_elbow, self.left_wrist)
        wrist_target_right = self.robot_wrist_target_right(self.right_shoulder, self.right_elbow, self.right_wrist)
        
        # Compute vector angles for loose joint targets
        theta_left = self.calc_theta_angles("left", self.left_shoulder, self.left_elbow, self.left_wrist)
        theta_right = self.calc_theta_angles("right", self.right_shoulder, self.right_elbow, self.right_wrist)

        self.visualize_targets(wrist_target_left, "left")
        self.visualize_targets(wrist_target_right, "right")

        msg = RobotImitationTargets()

        msg.wrist_target_left = wrist_target_left
        msg.wrist_target_right = wrist_target_right

        msg.shoulder_pitch_target_left = float(theta_left[0])
        msg.shoulder_roll_target_left = float(theta_left[1])
        msg.elbow_pitch_target_left = float(theta_left[2])
        msg.elbow_yaw_target_left = float(theta_left[3])

        msg.shoulder_pitch_target_right = float(theta_right[0])
        msg.shoulder_roll_target_right = float(theta_right[1])
        msg.elbow_pitch_target_right = float(theta_right[2])
        msg.elbow_yaw_target_right = float(theta_right[3])

        self.robot_targets_pub.publish(msg)

    def robot_wrist_target_left(self, shoulder, elbow, wrist):
        """ 
            Calculates target position for robot wrist by mapping from humans full reach to robots full reach
            Uses human direction from shoulder to wrist and a reaching factor [0-1] to determine robot wrist target
            
            3D coordinate mapping (MediaPipe image coords -> Robot coords):
                robot.x = -human.z (depth/forward, inverted)
                robot.y = human.x (left/right)
                robot.z = -human.y (up/down, inverted)
        """

        shoulder_wrist_mp = np.array([
            wrist.x - shoulder.x,
            wrist.y - shoulder.y,
            wrist.z - shoulder.z
        ])
        shoulder_elbow_mp = np.array([
            elbow.x - shoulder.x,
            elbow.y - shoulder.y,
            elbow.z - shoulder.z
        ])
        elbow_wrist_mp = np.array([
            wrist.x - elbow.x,
            wrist.y - elbow.y,
            wrist.z - elbow.z
        ])

        shoulder_wrist_mp_len = np.linalg.norm(shoulder_wrist_mp)
        # checking for zero-length vector to avoid division by zero
        shoulder_wrist_mp_unit = np.zeros(3) if shoulder_wrist_mp_len < self._eps else shoulder_wrist_mp / shoulder_wrist_mp_len

        denom = np.linalg.norm(shoulder_elbow_mp) + np.linalg.norm(elbow_wrist_mp)
        # checking for zero-length denom to avoid division by zero
        reaching_factor = 0.0 if denom < self._eps else shoulder_wrist_mp_len / denom

        full_reach_reaching_direction = shoulder_wrist_mp_unit * self.robot_full_reach_length
        wrist_target = self.mp_to_ainex_frame(full_reach_reaching_direction * reaching_factor)

        wrist_target_robot = Point()
        wrist_target_robot.x = wrist_target[0]  
        wrist_target_robot.y = wrist_target[1]  
        wrist_target_robot.z = wrist_target[2] 

        return wrist_target_robot
    
    def robot_wrist_target_right(self, shoulder, elbow, wrist):
        """ 
            Calculates target position for robot wrist by mapping from humans full reach to robots full reach
            Uses human direction from shoulder to wrist and a reaching factor [0-1] to determine robot wrist target
            
            3D coordinate mapping (MediaPipe image coords -> Robot coords):
                robot.x = -human.z (depth/forward, inverted)
                robot.y = human.x (left/right)
                robot.z = -human.y (up/down, inverted)
        """

        shoulder_wrist_mp = np.array([
            wrist.x - shoulder.x,
            wrist.y - shoulder.y,
            wrist.z - shoulder.z
        ])

        shoulder_wrist_mp_len = np.linalg.norm(shoulder_wrist_mp)
        if shoulder_wrist_mp_len < self._eps:
            shoulder_wrist_mp_unit = np.zeros(3)
        else:
            shoulder_wrist_mp_unit = shoulder_wrist_mp / shoulder_wrist_mp_len

        shoulder_elbow_mp = np.array([
            elbow.x - shoulder.x,
            elbow.y - shoulder.y,
            elbow.z - shoulder.z
        ])
        elbow_wrist_mp = np.array([
            wrist.x - elbow.x,
            wrist.y - elbow.y,
            wrist.z - elbow.z
        ])
        denom = np.linalg.norm(shoulder_elbow_mp) + np.linalg.norm(elbow_wrist_mp)
        reaching_factor = 0.0 if denom < self._eps else shoulder_wrist_mp_len / denom

        full_reach_reaching_direction = shoulder_wrist_mp_unit * self.robot_full_reach_length
        wrist_target = self.mp_to_ainex_frame(full_reach_reaching_direction * reaching_factor)

        wrist_target_robot = Point()
        wrist_target_robot.x = wrist_target[0]  
        wrist_target_robot.y = wrist_target[1]  
        wrist_target_robot.z = wrist_target[2] 

        return wrist_target_robot
    
    def calc_theta_angles(self, side, shoulder, elbow, wrist):

        # Computing vectors in AiNEX coordinates
        shoulder_elbow_robot = self.mp_to_ainex_frame(
            np.array([
                (elbow.x - shoulder.x),
                (elbow.y - shoulder.y),
                (elbow.z - shoulder.z)
            ])
        )

        elbow_wrist_robot = self.mp_to_ainex_frame(
            np.array([
                (wrist.x - elbow.x),
                (wrist.y - elbow.y),
                (wrist.z - elbow.z)
            ])
        )

        L1 = np.linalg.norm(shoulder_elbow_robot)
        if side == "left":
            def left_shoulder_to_elbow(vec):
                x, y, z = vec
                theta_1 = np.arctan2(x, -z)
                theta_2 = np.arctan2(np.sqrt(x**2 + z**2), y)

                return theta_1, theta_2
            def left_elbow_to_wrist(vec, theta_1, theta_2):
                x, y, z = vec
                s1 = np.sin(theta_1)
                c1 = np.cos(theta_1)
                s2 = np.sin(theta_2)
                c2 = np.cos(theta_2)

                A = c1 * x + s1 * z 
                B = - s1  * c2 * x + s2 * y + c1 * c2 * z
                C = s1 * s2 * x + c2 * y - c1 * s2 * z + L1
                theta_3 = np.arctan2(B, A)

                sqrt_term = np.sqrt(A**2 + B**2)
                elbow_above_shoulder = shoulder_elbow_robot[2] > 0.0
                theta_4 = np.arctan2(-sqrt_term if elbow_above_shoulder else sqrt_term, C)

                return theta_3, theta_4
            
            sho_pitch, sho_roll = left_shoulder_to_elbow(shoulder_elbow_robot)
            elbow_pitch, elbow_yaw = left_elbow_to_wrist(elbow_wrist_robot, sho_pitch, sho_roll)
            
            return sho_pitch, -sho_elbow_vert, elbow_pitch, -elbow_yaw

        elif side == "right":
            def right_shoulder_to_elbow(vec):
                x, y, z = vec
                theta_1 = np.arctan2(x, -z)
                theta_2 = np.arctan2(np.sqrt(x**2 + z**2), -y)
                return theta_1, theta_2
            
            def right_elbow_to_wrist(vec, theta_1, theta_2):
                x, y, z = vec
                s1 = np.sin(theta_1)
                c1 = np.cos(theta_1)
                s2 = np.sin(theta_2)
                c2 = np.cos(theta_2)

                A = c1 * x + s1 * z 
                B = - s1  * c2 * x - s2 * y + c1 * c2 * z
                C = -(s1 * s2 * x + c2 * y + c1 * s2 * z + L1)
                theta_3 = np.arctan2(B, A)

                sqrt_term = np.sqrt(A**2 + B**2)
                elbow_above_shoulder = shoulder_elbow_robot[2] > 0.0
                theta_4 = np.arctan2(sqrt_term if elbow_above_shoulder else sqrt_term, C)

                return theta_3, theta_4
            
            sho_pitch, sho_roll = right_shoulder_to_elbow(shoulder_elbow_robot)
            elbow_pitch, elbow_yaw = right_elbow_to_wrist(elbow_wrist_robot, sho_pitch, sho_roll)
            
            return sho_pitch, sho_roll, -elbow_pitch, elbow_yaw


    def mp_to_ainex_frame(self, mp_frame):
        ainex_frame = np.array([
            -mp_frame[2],   #x = -mp.z (forward)
            mp_frame[0],    # y = mp.x (left/right)
            -mp_frame[1],   # z = -mp.y (up/down)
        ])

        return ainex_frame

    def visualize_targets(self, xyz, side):
        """ 
            Visualizes the detected wrist positions and elbow angles in TF
        """
        
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


        t_msg.transform.translation.x = float(tx)   # vor der Kamera
        t_msg.transform.translation.y = float(ty)   # links/rechts
        t_msg.transform.translation.z = float(tz)   # hoch/runter

        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0 

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

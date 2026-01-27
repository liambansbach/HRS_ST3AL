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
        wrist_target_left = self.robot_wrist_target_left(self.left_shoulder, self.left_elbow, self.left_wrist)
        wrist_target_right = self.robot_wrist_target_right(self.right_shoulder, self.right_elbow, self.right_wrist)
        
        # Compute vector angles for loose joint targets
        theta_left = self.compute_arm_vector_angles_left(self.left_shoulder, self.left_elbow, self.left_wrist)
        theta_right = self.compute_arm_vector_angles_right(self.right_shoulder, self.right_elbow, self.right_wrist)
        theta_right = self.compute_arm_vector_angles_left(self.right_shoulder, self.right_elbow, self.right_wrist)

        theta_left = self.calc_theta_angles("left", self.left_shoulder, self.left_elbow, self.left_wrist)
        theta_right = self.calc_theta_angles("right", self.right_shoulder, self.right_elbow, self.right_wrist)

        self.visualize_targets(wrist_target_left, "left")
        self.visualize_targets(wrist_target_right, "right")

        msg = RobotImitationTargets()

        msg.wrist_target_left = wrist_target_left
        msg.wrist_target_right = wrist_target_right

        # Vector angle fields
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
                robot.x = human.z (depth/forward)
                robot.y = human.x (left/right)
                robot.z = -human.y (up/down, inverted)
        """

        """ Left """
        shoulder_wrist_mp = np.array([
            wrist.x - shoulder.x,
            wrist.y - shoulder.y,
            wrist.z - shoulder.z
        ])

        shoulder_wrist_mp_len = np.linalg.norm(shoulder_wrist_mp)
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
        reaching_factor = shoulder_wrist_mp_len / (np.linalg.norm(shoulder_elbow_mp) + np.linalg.norm(elbow_wrist_mp))

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
                robot.x = human.z (depth/forward)
                robot.y = human.x (left/right)
                robot.z = -human.y (up/down, inverted)
        """

        """ Right """
        shoulder_wrist_mp = np.array([
            wrist.x - shoulder.x,
            wrist.y - shoulder.y,
            wrist.z - shoulder.z
        ])

        shoulder_wrist_mp_len = np.linalg.norm(shoulder_wrist_mp)
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
        reaching_factor = shoulder_wrist_mp_len / (np.linalg.norm(shoulder_elbow_mp) + np.linalg.norm(elbow_wrist_mp))

        full_reach_reaching_direction = shoulder_wrist_mp_unit * self.robot_full_reach_length
        wrist_target = self.mp_to_ainex_frame(full_reach_reaching_direction * reaching_factor)

        #rotate around robot.y by -90 degrees
        # wrist_target = np.array([
        #     wrist_target[2],    #new x = old z
        #     wrist_target[1],    #new y = old y
        #     -wrist_target[0]    #new z = old x
        # ])

        wrist_target_robot = Point()
        wrist_target_robot.x = wrist_target[0]  
        wrist_target_robot.y = wrist_target[1]  
        wrist_target_robot.z = wrist_target[2] 

        return wrist_target_robot
    
    def compute_arm_vector_angles_left(self, shoulder, elbow, wrist):
        """
        Compute horizontal (azimuth) and vertical (elevation) angles for arm vectors.
        
        These angles represent the direction of:
          - Shoulder->Elbow vector (upper arm)
          - Elbow->Wrist vector (forearm)
        
        relative to a reference direction (pointing forward/down).
        
        The angles are computed in the robot coordinate frame:
          - robot.x = forward (MediaPipe -z)
          - robot.y = left/right (MediaPipe x)
          - robot.z = up/down (MediaPipe -y)
        
        Args:
            side: 'left' or 'right'
        
        Returns:
            (sho_elbow_horiz, sho_elbow_vert, elbow_wrist_horiz, elbow_wrist_vert)
            All angles in radians.
        """

        # Compute vectors in MediaPipe coordinates
        shoulder_elbow_robot = self.mp_to_ainex_frame(
            np.array([
                elbow.x - shoulder.x,
                elbow.y - shoulder.y,
                elbow.z - shoulder.z
            ])
        )
        elbow_wrist_robot = self.mp_to_ainex_frame(
            np.array([
                wrist.x - elbow.x,
                wrist.y - elbow.y,
                wrist.z - elbow.z
            ])
        )
        
        def left_shoulder_to_elbow(vec):
            x, y, z = vec
            theta_1 = np.arctan2(x, -z)
            theta_2 = np.arctan2(np.sqrt(x**2 + z**2), y)
            return theta_1, theta_2
        
        def left_elbow_to_wrist(theta_1, theta_2):
            u = np.array([
                np.sin(theta_1)*np.sin(theta_2), 
                np.cos(theta_2),
                -np.cos(theta_1)*np.sin(theta_2)
            ])
            c = np.array([
                np.cos(theta_1), 
                0,
                np.sin(theta_1)
            ])
            s = np.array([
                -np.sin(theta_1)*np.cos(theta_2), 
                np.sin(theta_2),
                np.cos(theta_1)*np.cos(theta_2)
            ])

            proj_u = np.dot(elbow_wrist_robot, u)
            proj_c = np.dot(elbow_wrist_robot, c)
            proj_s = np.dot(elbow_wrist_robot, s) 

            theta_3 = np.arctan2(proj_s, proj_c)
            theta_4 = np.arctan2(np.sqrt(proj_c**2 + proj_s**2), proj_u)

            return theta_3, theta_4
            
        sho_elbow_horiz, sho_elbow_vert = left_shoulder_to_elbow(shoulder_elbow_robot)
        elbow_wrist_horiz, elbow_wrist_vert = left_elbow_to_wrist(sho_elbow_horiz, sho_elbow_vert)
        
        return sho_elbow_horiz, -sho_elbow_vert, elbow_wrist_horiz, -elbow_wrist_vert
    
    def compute_arm_vector_angles_right(self, shoulder, elbow, wrist):
        """
        Compute horizontal (azimuth) and vertical (elevation) angles for arm vectors.
        
        These angles represent the direction of:
          - Shoulder->Elbow vector (upper arm)
          - Elbow->Wrist vector (forearm)
        
        relative to a reference direction (pointing forward/down).
        
        The angles are computed in the robot coordinate frame:
          - robot.x = forward (MediaPipe -z)
          - robot.y = left/right (MediaPipe x)
          - robot.z = up/down (MediaPipe -y)
        
        Args:
            side: 'left' or 'right'
        
        Returns:
            (sho_elbow_horiz, sho_elbow_vert, elbow_wrist_horiz, elbow_wrist_vert)
            All angles in radians.
        """

        # Compute vectors in MediaPipe coordinates
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
        
        def right_shoulder_to_elbow(vec):
            x, y, z = vec
            # flipped signs for right arm
            theta_1 = np.arctan2(-x, z)
            # not sure if signs of theta_2 aka the sqrt should be negative here (is +-)
            theta_2 = np.arctan2(np.sqrt(x**2 + z**2), y)
            return theta_1, theta_2
        
        def right_elbow_to_wrist(theta_1, theta_2):
            u = np.array([
                -np.sin(theta_1)*np.sin(theta_2), 
                np.cos(theta_2),
                np.cos(theta_1)*np.sin(theta_2)
            ])
            c = np.array([
                -np.cos(theta_1), 
                0,
                -np.sin(theta_1) 
            ])
            s = np.array([
                np.sin(theta_1)*np.cos(theta_2), 
                np.sin(theta_2),
                -np.cos(theta_1)*np.cos(theta_2)
            ])

            # with T0_1 and T2_3 inverted instead of T1_2 and T3_4
            # u = np.array([
            #     -np.sin(theta_1)*np.sin(theta_2), 
            #     np.cos(theta_2),
            #     -np.cos(theta_1)*np.sin(theta_2)
            # ])
            # c = np.array([
            #     np.cos(theta_1), 
            #     0,
            #     -np.sin(theta_1) 
            # ])
            # s = np.array([
            #     -np.sin(theta_1)*np.cos(theta_2), 
            #     np.sin(theta_2),
            #     -np.cos(theta_1)*np.cos(theta_2)
            # ])

            proj_u = np.dot(elbow_wrist_robot, u)
            proj_c = np.dot(elbow_wrist_robot, c)
            proj_s = np.dot(elbow_wrist_robot, s) 

            theta_3 = np.arctan2(proj_s, proj_c)
            theta_4 = np.arctan2(np.sqrt(proj_c**2 + proj_s**2), proj_u)

            return theta_3, theta_4
            
        sho_elbow_horiz, sho_elbow_vert = right_shoulder_to_elbow(shoulder_elbow_robot)
        elbow_wrist_horiz, elbow_wrist_vert = right_elbow_to_wrist(sho_elbow_horiz, sho_elbow_vert)
        
        return sho_elbow_horiz, -sho_elbow_vert, elbow_wrist_horiz, -elbow_wrist_vert
    
    def calc_theta_angles(self, side, shoulder, elbow, wrist):

        # Compute vectors in MediaPipe coordinates
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
        # not sure about this:
        L1 = np.linalg.norm(shoulder_elbow_robot)
        L2 = np.linalg.norm(elbow_wrist_robot)    
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

                # probably wrong, due to sign error in Pos
                #theta_3 = np.arctan2(-s1 * c2 * x + s2 * x + c1 * c2 * z, -(c1 * x + s1 * z))
                #theta_4 = np.arctan2(np.sqrt((c1 * x + s1 * z)**2 + (-s1 * c2 * x + s2 * x + c1 * c2 * z)**2),(s1 *s2 +x + c2 * y - c1 *s2 * z - L1))


                # your first try, produces nan for theta_3 :
                # theta_4 = np.arccos(1/L2 *( s1 *s2 * x + c2 * y - c1 * s2 * z - L1))
                # s4 = np.sin(theta_4)
                # theta_3 = np.arccos((c1 * x + s1 * z )/ (s4 * L2))  
                # self.get_logger().info(f"theta_3_first: {theta_3}, theta_4_first: {theta_4}")

                # gpts correction, works:
                A = c1 * x + s1 * z 
                B = - s1  * c2 * x + s2 * y + c1 * c2 * z
                D = s1 * s2 * x + c2 * y - c1 * s2 * z - L1
                theta_3 = np.arctan2(B, A)
                # maybe the +- infront of sqrt should be considered somehow to reach every position?
                theta_4 = np.arctan2(np.sqrt(A**2 + B**2), D)

                #self.get_logger().info(f"theta_3_second: {theta_3}, theta_4_second: {theta_4}")
                return theta_3, theta_4
            
            sho_elbow_horiz, sho_elbow_vert = left_shoulder_to_elbow(shoulder_elbow_robot)
            elbow_wrist_horiz, elbow_wrist_vert = left_elbow_to_wrist(elbow_wrist_robot, sho_elbow_horiz, sho_elbow_vert)
            
            return sho_elbow_horiz, -sho_elbow_vert, elbow_wrist_horiz, -elbow_wrist_vert

        elif side == "right":
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

                # not sure if marius equations are correct as -s2 * x seems wrong? 
                #theta_3 = np.arctan((c1 * x + s1 *z), -(-s1 * c2 * x - s2 * x + c1 * c2 * z))
                #theta_4 = np.arctan2(np.sqrt )

                # gpts solved equations
                A = c1 * x + s1 * z
                B = - s1 * s2 * x + c2 * y + c1 * s2 * z + L1
                C = - s1 * c2 * x - s2 * y + c1 * c2 * z
                theta_3 = np.arctan2(A, -C)
                theta_4 = np.arctan2(-B, np.sqrt( L2**2 - B**2))  # maybe the +- infront of sqrt should be considered somehow to reach every position?


                #self.get_logger().info(f"theta_3_second: {theta_3}, theta_4_second: {theta_4}")
                return theta_3, theta_4
            
            sho_elbow_horiz, sho_elbow_vert = left_shoulder_to_elbow(shoulder_elbow_robot)
            elbow_wrist_horiz, elbow_wrist_vert = left_elbow_to_wrist(elbow_wrist_robot, sho_elbow_horiz, sho_elbow_vert)
            
            return sho_elbow_horiz, -sho_elbow_vert, elbow_wrist_horiz, -elbow_wrist_vert


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

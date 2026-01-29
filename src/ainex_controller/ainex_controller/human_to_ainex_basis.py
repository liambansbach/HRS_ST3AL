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
        self._eps = 1e-6

    def bodypose_cb(self, msg: UpperbodyPose):
        """ 
            Callback for upperbody pose subscriber
            Also calculates and publisher robot targets for each message
        """

        # TODO check if we need to convert the coordinates from camera_link to base_link?
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

                # gpts correction, works:
                A = c1 * x + s1 * z 
                B = - s1  * c2 * x + s2 * y + c1 * c2 * z
                C = s1 * s2 * x + c2 * y - c1 * s2 * z + L1
                theta_3 = np.arctan2(B, A)
                # maybe the +- infront of sqrt should be considered somehow to reach every position?
                # positive sign => elbow BELOW the shoulder seems to work
                # negative sign => elbow ABOVE the shoulder seems to work
                #theta_4 = np.arctan2(np.sqrt(A**2 + B**2), C)
                sqrt_term = np.sqrt(A**2 + B**2)
                elbow_above_shoulder = shoulder_elbow_robot[2] > 0.0
                theta_4 = np.arctan2(-sqrt_term if elbow_above_shoulder else sqrt_term, C)


                return theta_3, theta_4
            
            sho_elbow_horiz, sho_elbow_vert = left_shoulder_to_elbow(shoulder_elbow_robot)
            elbow_wrist_horiz, elbow_wrist_vert = left_elbow_to_wrist(elbow_wrist_robot, sho_elbow_horiz, sho_elbow_vert)
            
            return sho_elbow_horiz, -sho_elbow_vert, elbow_wrist_horiz, -elbow_wrist_vert

        elif side == "right":
            def left_shoulder_to_elbow(vec):
                x, y, z = vec
                theta_1 = np.arctan2(x, -z)
                theta_2 = np.arctan2(np.sqrt(x**2 + z**2), -y)
                return theta_1, theta_2
            
            def left_elbow_to_wrist(vec, theta_1, theta_2):
                x, y, z = vec
                s1 = np.sin(theta_1)
                c1 = np.cos(theta_1)
                s2 = np.sin(theta_2)
                c2 = np.cos(theta_2)
                #chatgpts correction of your arctan2 version:
                # TODO change signs of FK aka in Transformation matrices to match the negation of theta_3 here;
                theta_3 = -np.arctan2(-s1*c2*x - s2*y + c1*c2*z, c1*x + s1*z)
                proj = -s1*s2*x + c2*y + c1*s2*z + L1
                # for this theta_4 following works: lift arms straighupwards, bending sideways, doing 90 degree elbow bend inwards (but only up to a point, but this also happens to left arm) -> seems to almost everything work
                # what doesnt work: turning arm inside aka when both arms are on the thighs -> arm bends outwards (to the left) -> seems like theta_4 sign is still wrong

                # positive sign => elbow BELOW the shoulder seems to work
                # negative sign => elbow ABOVE the shoulder seems to work
                if L2 < self._eps:
                    return 0.0, 0.0
                proj_clamped = np.clip(proj, -L2, L2)
                if abs(proj - proj_clamped) > self._eps:
                    self.get_logger().warn(
                        f"proj out of range (proj={proj:.6f}, L2={L2:.6f}), clamping for theta_4."
                    )
                proj = proj_clamped
                radicand = L2**2 - proj**2
                if radicand < 0.0:
                    self.get_logger().warn(
                        f"Negative radicand for theta_4 sqrt (radicand={radicand:.6f}); clamping to 0."
                    )
                    radicand = 0.0
                sqrt_term = np.sqrt(radicand)
                theta_4 = np.arctan2(sqrt_term, -proj)

                # with non inverted theta_3, this produces shit
                # with inverted theta_3, this produces arm bending inwards (to the right) -> could be correct!!! -> test tomorrow on real robot
                # what works: arm inside aka both arms on the thighs -> arm bends inwards (to the right)
                # what doesnt work: bending arm upwards with elbow bend -> arm bends outwards (to the left) -> seems like theta_4 sign is still wrong -> could this come from an unreachable position?
                theta_4 = np.arctan2(-sqrt_term, -proj)

                # checking if arm is above shoulder to choose correct sign for theta_4:
                elbow_above_shoulder = shoulder_elbow_robot[2] > 0.0
                # reuse sqrt_term from clamped radicand
                theta_4 = np.arctan2(-sqrt_term if elbow_above_shoulder else sqrt_term, -proj)

                """
                #### Mini report of theta_4 behaviour:
                - theta_4 with negative sign seems to be more correct, as the wrist is bend inwards like all the time in the videos
                - for theta_4 with positive sign, the wrist is bend outwards in most poses, which seems wrong
                - however, with negative theta_4, the wrist turns inwards instead of outwards for positions where the elbow is above the shoulder 
                """

                #self.get_logger().info(f"theta_3: {theta_3}, theta_4: {theta_4}")
                return theta_3, theta_4
            
            sho_elbow_horiz, sho_elbow_vert = left_shoulder_to_elbow(shoulder_elbow_robot)
            elbow_wrist_horiz, elbow_wrist_vert = left_elbow_to_wrist(elbow_wrist_robot, sho_elbow_horiz, sho_elbow_vert)
            
            return sho_elbow_horiz, sho_elbow_vert, elbow_wrist_horiz, elbow_wrist_vert


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

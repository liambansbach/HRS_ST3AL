import pinocchio as pin
import numpy as np
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class AiNexModel:
    """AiNex Robot Model using Pinocchio"""
    def __init__(self, node: Node, urdf_path: str, q_init=None, v_init=None):
        self.node = node
        self.br = TransformBroadcaster(node)

        # Load Pinocchio model from URDF
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        self.q = q_init if q_init is not None else np.zeros(self.model.nq)
        self.v = v_init if v_init is not None else np.zeros(self.model.nv)

        # Add additional frames for left and right hands
        self.add_additional_frames(
            name="l_hand_link",
            parent_frame="l_gripper_link",
            translation=np.array([-0.02, 0.025, 0.0]),
            rotation=np.eye(3)
        )
        self.add_additional_frames(
            name="r_hand_link",
            parent_frame="r_gripper_link",
            translation=np.array([-0.02, -0.025, 0.0]),
            rotation=np.eye(3)
        )

        # Retrieve frame IDs for hands for later use
        self.left_hand_id = self.model.getFrameId("l_hand_link")
        self.right_hand_id = self.model.getFrameId("r_hand_link")

        # Initialize end-effector poses and velocities
        self.x_left = pin.SE3.Identity()
        self.x_right = pin.SE3.Identity()

        self.v_left = pin.Motion.Zero()
        self.v_right = pin.Motion.Zero()

        # Initialize Jacobians for left and right hands
        self.J_left = np.zeros((6, self.model.nv))
        self.J_right = np.zeros((6, self.model.nv))

        # Store joint names in Pinocchio order
        self.joint_names = list(self.model.names[1:])
        print("Node names in Pinocchio model:", self.joint_names)

    def update_model(self, q, v):
        """Update the model with new joint positions and velocities."""
        self.q = q
        self.v = v

        # update pinocchio model with new q, v
        pin.forwardKinematics(self.model, self.data, self.q, self.v)
        pin.updateFramePlacements(self.model, self.data)

        # TODO: retrieve end-effector poses from updated pinocchio data
        # Hint: take a look at the init function for hand ids
        self.x_left = self.data.oMf[self.left_hand_id].copy() # ref docu: Vector of absolute operationnel frame placements (wrt the world). 
        self.x_right = self.data.oMf[self.right_hand_id].copy() # Essentially this is the end-effector frame placement with respect to the base-link (com?)

        # TODO: get end-effector Jacobians in pin.LOCAL_WORLD_ALIGNED frame
        # https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/jnrh2023/template/frame.html
        pin.computeJointJacobians(self.model, self.data, self.q) 
        self.J_left = pin.getFrameJacobian(self.model, self.data, self.left_hand_id, pin.LOCAL_WORLD_ALIGNED)
        self.J_right = pin.getFrameJacobian(self.model, self.data, self.right_hand_id, pin.LOCAL_WORLD_ALIGNED)

        # TODO: calculate end-effector velocities using the Jacobians
        # Hint: v_cartesian = J * v_joint
        # self.v_left = self.J_left @ self.v
        # self.v_right = self.J_right @ self.v

        self.v_left = pin.Motion(self.J_left @ self.v)
        self.v_right = pin.Motion(self.J_right @ self.v)

        # TODO: broadcast tf transformation of hand links w.r.t. base_link for visualization in RViz
        # Hint: take a look at the tf2_ros documentation for examples
        # https://docs.ros.org/en/jazzy/Tutorials/Intermediate/Tf2/Writing-A-Tf2-Broadcaster-Py.html
        self.broadcast_tf("left") # Broadcaster for left hand
        self.broadcast_tf("right") # Broadcaster for right hand

    def broadcast_tf(self, hand: str):
        t = TransformStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.header.frame_id = "base_link"
        t.child_frame_id = hand

        match hand:
            case "left":
                chosen_hand = self.x_left
            case "right":
                chosen_hand = self.x_right
            case _:
                print("The robot only have two hands, left or right...") 
                return 

        t.transform.translation.x = chosen_hand.translation[0]
        t.transform.translation.y = chosen_hand.translation[1]
        t.transform.translation.z = chosen_hand.translation[2]

        q = pin.Quaternion(chosen_hand.rotation)
        t.transform.rotation.x = q.x
        t.transform.rotation.y = q.y
        t.transform.rotation.z = q.z
        t.transform.rotation.w = q.w

        self.br.sendTransform(t)

    def add_additional_frames(self, name, parent_frame, translation, rotation):
        # parent_frame: a frame name in the existing model (string)
        # compute the parent joint index from the existing frame
        parent_frame_id = self.model.getFrameId(parent_frame)
        parent_joint = self.model.frames[parent_frame_id].parent

        # ensure numpy arrays and create SE3 placement
        translation = np.asarray(translation, dtype=float)
        rotation = np.asarray(rotation, dtype=float)
        placement = pin.SE3(rotation, translation)

        # create Frame using parent joint id and SE3 placement
        frame = pin.Frame(name, parent_joint, placement, pin.FrameType.OP_FRAME)

        # add to model and recreate data so sizes match
        self.model.addFrame(frame)
        self.data = self.model.createData()

    def left_hand_pose(self):
        """Return the left hand pose in base_link frame."""
        return self.x_left
    
    def right_hand_pose(self):
        """Return the right hand pose in base_link frame."""
        return self.x_right
    
    def left_hand_velocity(self):
        """Return the left hand velocity in base_link frame."""
        return self.v_left
    
    def right_hand_velocity(self):
        """Return the right hand velocity in base_link frame."""
        return self.v_right
    
    def left_hand_jacobian(self):
        """Return the left hand jacobian in base_link frame."""
        return self.J_left
    
    def right_hand_jacobian(self):
        """Return the right hand jacobian in base_link frame."""
        return self.J_right
    
    def pin_joint_names(self):
        """Return the joint names in Pinocchio model order."""
        return self.joint_names
    
    def get_arm_ids(self, arm_side: str):
        """Get joint ids for the specified arm side ('left' or 'right')."""
        # l/r_sho_pitch, l/r_sho_roll, l/r_el_yaw, l/r_el_pitch

        arm_joint_names = ['sho_pitch', 'sho_roll', 'el_yaw', 'el_pitch']
        if arm_side == 'left':
            prefix = 'l_'
        elif arm_side == 'right':
            prefix = 'r_'
        else:
            raise ValueError("arm_side must be 'left' or 'right'")
        
        arm_joint_names = [prefix + 'sho_pitch', 
                           prefix + 'sho_roll', 
                           prefix + 'el_pitch', 
                           prefix + 'el_yaw']

        arm_ids = []
        for name in arm_joint_names:
            jid = self.model.getJointId(name)
            q_idx = self.model.joints[jid].idx_q
            arm_ids.append(q_idx)

        return arm_ids
    

    def get_arm_v_ids(self, arm_side: str):
        """Velocity indices (idx_v) for arm joints, for slicing Jacobians."""
        if arm_side == 'left':
            prefix = 'l_'
        elif arm_side == 'right':
            prefix = 'r_'
        else:
            raise ValueError("arm_side must be 'left' or 'right'")

        arm_joint_names = [
            prefix + 'sho_pitch',
            prefix + 'sho_roll',
            prefix + 'el_pitch',
            prefix + 'el_yaw',
        ]

        v_ids = []
        for name in arm_joint_names:
            jid = self.model.getJointId(name)
            v_ids.append(self.model.joints[jid].idx_v)  # <-- important
        return v_ids

    def get_joint_id(self, joint_name: str) -> int:
        """Get the joint id from the pinocchio model."""
        jid = self.model.getJointId(joint_name)
        q_idx = self.model.joints[jid].idx_q
        return q_idx
    
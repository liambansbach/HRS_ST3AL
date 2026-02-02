"""
AiNEX Robot Model Wrapper.

This module wraps a Pinocchio model loaded from URDF and exposes convenience helpers
for updating forward kinematics, managing hand frames, and accessing joint metadata
used by the imitation controllers.

Key Features:
    - **URDF Loading**: Builds the Pinocchio model and data from a provided URDF path.
    - **Frame Augmentation**: Adds custom left/right hand frames for easier end-effector access.
    - **Kinematics Update**: Maintains end-effector poses and provides update utilities.
    - **Joint Metadata**: Exposes joint names and arm joint indices in Pinocchio order.

Dependencies:
    - pinocchio: Rigid-body model, kinematics, and frame handling.
    - numpy: State storage and array utilities.
    - rclpy: ROS 2 node utilities (for TF broadcaster setup).
    - tf2_ros, geometry_msgs: TF broadcasting helpers (currently unused).

Classes:
    AiNexModel: Pinocchio-backed model utilities for AiNEX.
"""

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

        self.x_left = self.data.oMf[self.left_hand_id].copy() 
        self.x_right = self.data.oMf[self.right_hand_id].copy() 

    def add_additional_frames(self, name, parent_frame, translation, rotation):
        """Add an additional frame to the Pinocchio model.
        Args:
            name (str): Name of the new frame.
            parent_frame (str): Name of the parent frame.
            translation (np.ndarray): Translation vector of the new frame.
            rotation (np.ndarray): Rotation matrix of the new frame.
        Returns:
            None
        """
        # parent_frame: a frame name in the existing model (string)
        # compute the parent joint index from the existing frame
        parent_frame_id = self.model.getFrameId(parent_frame)
        parent_joint = self.model.frames[parent_frame_id].parent

        # ensure numpy arrays and create SE3 placement
        translation = np.asarray(translation, dtype=float)
        rotation = np.asarray(rotation, dtype=float)
        placement = pin.SE3(rotation, translation)

        # create Frame using parent joint id, parent frame id, and SE3 placement
        #frame = pin.Frame(name, parent_joint, placement, pin.FrameType.OP_FRAME)
        frame = pin.Frame(name, parent_joint, parent_frame_id, placement, pin.FrameType.OP_FRAME)

        # add to model and recreate data so sizes match
        self.model.addFrame(frame)
        self.data = self.model.createData()

    def left_hand_pose(self):
        """Return the left hand pose in base_link frame.
        Args:
            None
        Returns:
            SE3: Left hand pose in base_link frame.
        """
        return self.x_left
    
    def right_hand_pose(self):
        """Return the right hand pose in base_link frame.
        Args:
            None
        Returns:
            SE3: Right hand pose in base_link frame.
        """
        return self.x_right
        
    def pin_joint_names(self):
        """Return the joint names in Pinocchio model order.
        Args:
            None
        Returns:
            List[str]: Joint names.
        """
        return self.joint_names
    
    def get_arm_ids(self, arm_side: str):
        """Get joint ids for the specified arm side ('left' or 'right').
        Args:
            arm_side (str): 'left' or 'right'
        Returns:
            List[int]: Joint ids for the specified arm.
        """
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
                           prefix + 'el_yaw', 
                           prefix + 'el_pitch']

        arm_ids = []
        for name in arm_joint_names:
            jid = self.model.getJointId(name)
            q_idx = self.model.joints[jid].idx_q
            arm_ids.append(q_idx)

        return arm_ids

    def get_joint_id(self, joint_name: str) -> int:
        """Get the joint id from the pinocchio model.
        Args:
            joint_name (str): Name of the joint.
        Returns:
            int: Joint id."""
        jid = self.model.getJointId(joint_name)
        q_idx = self.model.joints[jid].idx_q
        return q_idx

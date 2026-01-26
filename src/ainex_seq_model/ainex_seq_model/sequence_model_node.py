#!/usr/bin/env python3
"""
Under Construction
"""

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from collections import deque #double ended queue as buffer

from ainex_interfaces.msg import ManipulationSeq, CubeBBoxList


class SequenceModel(Node):
    def __init__(self):
        super().__init__('sequence_model')

        self.base_frame = 'base_link'
        self.frame_to_id = {
            "hrs_cube_red": "red",
            "hrs_cube_green": "green",
            "hrs_cube_blue": "blue",
        }

        #History length ~2 seconds at 20 Hz
        self.history_len = 40  

        self.histories = {}
        for cube in self.frame_to_id.values():
            self.histories[cube] = deque(maxlen=self.history_len)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        #20Hz timer -> equals broadcasting frequency
        self.create_timer(0.05, self.tick)
    
    def tick(self):
        now = rclpy.time.Time()
        state = {}

        for cube, color in self.frame_to_id.items():
            if not self.tf_buffer.can_transform(self.base_frame, cube, now):
                continue

            tf = self.tf_buffer.lookup_transform(self.base_frame, cube, now)

            obs = {
                "t": tf.header.stamp,
                "cube": color,
                "x": tf.transform.translation.x,
                "y": tf.transform.translation.y,
                "z": tf.transform.translation.z,
            }

            state[color] = obs
            self.histories[color].append(obs)



def main(args=None):
    rclpy.init(args=args)
    node = SequenceModel()
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
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from ainex_interfaces.msg import ManipulationSeq


class DetectionListener(Node):
    def __init__(self):
        super().__init__('detection_listener')



def main(args=None):
    rclpy.init(args=args)
    node = DetectionListener()
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

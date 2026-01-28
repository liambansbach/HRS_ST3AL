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

        # --- motion params (TF xyz) ---
        self.motion_window = 10          # frames to look back (10 @ 20Hz ≈ 0.5s)
        self.occluded_frames = 6         # last N frames missing => occluded (~0.3s @ 20Hz)
        self.min_disp_m = 0.002          # ignore tiny jitter (<2mm)

        self.v_moving_thresh = 0.03      # m/s  (tune)
        self.v_still_thresh  = 0.01      # m/s  (tune)

        for cube in self.frame_to_id.values():
            self.motion = {cube: {"moving": False, "still": False, "occluded": True, "speed": 0.0}}
        
    
    def tick(self):
        """
        Docstring for tick

        """
        now = rclpy.time.Time()
        state = {}

        #For each cube, build a message and append it to its corresponding deque 
        for cube, color in self.frame_to_id.items():
            visible = self.tf_buffer.can_transform(self.base_frame, cube, now)

            #Feed current TF information if cube is visible
            if visible:
                tf = self.tf_buffer.lookup_transform(self.base_frame, cube, now)
                obs = {
                    "t": tf.header.stamp,
                    "cube": color,
                    "visible": True,
                    "x": tf.transform.translation.x,
                    "y": tf.transform.translation.y,
                    "z": tf.transform.translation.z,
                }
            
            #Feed the last visible state when cube is occluded
            else:
                last = self.histories[color][-1] if len(self.histories[color]) else None
                obs = {
                    "t": self.get_clock().now().to_msg(),  # no TF stamp available
                    "cube": color,
                    "visible": False,
                    "x": last["x"] if last and last.get("x") is not None else None,
                    "y": last["y"] if last and last.get("y") is not None else None,
                    "z": last["z"] if last and last.get("z") is not None else None,
                }

            state[color] = obs
            self.histories[color].append(obs)

    def derive_motion(self) -> None:
        """
        Compute moving/still/occluded per cube using 3D speed over a window.
        """
        for cube in self.frame_to_id.values():
            hist = self.histories.get(cube, None)
            if not hist or len(hist) < 2:
                self.motion[cube] = {"moving": False, "still": False, "occluded": True, "speed": 0.0}
                continue
            
            #Check how many frames a cube was occluded 
            last_n = list(hist)[-self.occluded_frames:]
            occluded = False
            if len(last_n) == self.occluded_frames:
                all_invisible = True
                for occ in last_n:
                    if occ.get("visible", False):
                        all_invisible = False
                        break
                occluded = all_invisible
            

            recent = list(hist)[-self.motion_window:]
            valid_obs = []
            for obs in recent:
                if not obs.get("visible", False):
                    continue
                if obs.get("x") is None:
                    continue
                if obs.get("y") is None:
                    continue
                if obs.get("z") is None:
                    continue
                valid_obs.append(obs)

            speed = 0.0

            if (not occluded) and len(valid_obs) >= 2:
                old_obs = valid_obs[0]
                new_obs = valid_obs[-1]

            #Derive dt using TF stamps 
            dt_ns = (new_obs["t"].sec - old_obs["t"].sec) * 1_000_000_000
            dt_ns += (new_obs["t"].nanosec - old_obs["t"].nanosec)
            dt = 0.0
            if dt_ns > 0:
                dt = dt_ns / 1e9

            if dt > 0.0:
                dx = float(new_obs["x"]) - float(old_obs["x"])
                dy = float(new_obs["y"]) - float(old_obs["y"])
                dz = float(new_obs["z"]) - float(old_obs["z"])

                #Distance in 3D
                dist = (dx * dx + dy * dy + dz * dz) ** 0.5

                if dist >= self.min_disp_m:
                    speed = dist / dt

            moving = False
            still = False

            if not occluded:
                if speed > self.v_moving_thresh:
                    moving = True
                if speed < self.v_still_thresh:
                    still = True

            self.motion[cube] = {"moving": moving, "still": still, "occluded": occluded, "speed": speed}


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
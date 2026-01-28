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
        self.motion_window = 15         # frames to look back (10 @ 20Hz ≈ 0.5s)
        self.occluded_frames = 6         # last N frames missing => occluded (~0.3s @ 20Hz)
        self.min_disp_m = 0.005          # ignore tiny jitter (<2mm)

        self.v_moving_thresh = 0.05      # m/s  (tune)
        self.v_still_thresh  = 0.01      # m/s  (tune)

        self.motion = {}
        self.fsm_state = {}
        self.still_count = {}
        self.moving_count = {}
        for cube in self.frame_to_id.values():
            self.motion[cube] = {"moving": False, "still": False, "occluded": True, "speed": 0.0}
            self.fsm_state[cube] = "UNKNOWN"
            self.still_count[cube] = 0
            self.moving_count[cube] = 0

        # --- moving/still params ---
        self.still_thresh = 5       # framecount (tune)
        self.moving_thresh = 5      # framecount (tune)

        self.events = []
        
    
    def tick(self):
        """
        Docstring for tick

        """
        now = rclpy.time.Time()

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

            self.histories[color].append(obs)

        self.derive_motion()
        self.fsm_pick_place()


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


    def fsm_pick_place(self) -> None:
        event_t = self.get_clock().now().to_msg()

        for cube in self.frame_to_id.values():
            motion = self.motion.get(cube, None)
            if motion is None:
                continue

            occ_flag = bool(motion.get("occluded", True))
            cube_moving = bool(motion.get("moving", False))
            cube_still  = bool(motion.get("still", False))

            #Occluded cubes count as moving
            if occ_flag:
                cube_moving = True
                cube_still = False

            if cube_still:
                self.still_count[cube] = self.still_count[cube] +1
            else:
                self.still_count[cube] = 0

            if cube_moving:
                self.moving_count[cube] = self.moving_count[cube] +1
            else:
                self.moving_count[cube] = 0

            previous_state = self.fsm_state[cube]
            new_state = previous_state

            #State Transitions
            if previous_state == "UNKNOWN":
                if self.still_count[cube] >= self.still_thresh:
                    new_state = "STILL"
                elif self.moving_count[cube] >= self.moving_thresh:
                    new_state = "MOVING"
            
            #PICK
            elif previous_state == "STILL":
                if self.moving_count[cube] >= self.moving_thresh:
                    new_state = "MOVING"
                    
            #PLACE
            elif previous_state == "MOVING":
                if self.still_count[cube] >= self.still_thresh:
                    new_state = "STILL"
                    
            #Write the events in the sequence
            if (previous_state == "STILL") and (new_state == "MOVING"):
                self.events.append({"t": event_t, "type": "PICK", "cube": cube})
                last_event = self.events[-1]
                self.get_logger().info(f'{last_event["type"]}: {last_event["cube"]}')
            elif (previous_state == "MOVING") and (new_state == "STILL"):
                self.events.append({"t": event_t, "type": "PLACE", "cube": cube})
                last_event = self.events[-1]
                self.get_logger().info(f'{last_event["type"]}: {last_event["cube"]}')

            self.fsm_state[cube] = new_state


            
            









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
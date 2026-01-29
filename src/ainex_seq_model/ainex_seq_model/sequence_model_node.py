#!/usr/bin/env python3
"""
ROS 2 node that converts 2D cube detections into a symbolic manipulation trace.

This node subscribes to per-frame cube bounding boxes, maintains a short history
buffer per cube, estimates motion (moving/still/occluded) from windowed speeds,
runs a per-cube FSM to detect PICK/PLACE events, and infers a discrete world
state (stacking/support relationships and left/middle/right grouping) once the
scene is stable.

When a RecordDemo action goal is active, the node records:
    - a time-ordered list of ManipulationEvent (PICK/PLACE per cube)
    - a time-ordered list of WorldState snapshots (symbolic state per cube)

Threading/ROS execution:
    - Uses a ReentrantCallbackGroup and a MultiThreadedExecutor so that the
    subscriber callback and action server can run concurrently.

HRS 2025 - Group B:
    Liam Bansbach
    Marius Moe Rena
    Niklas Peter
    Tobias Töws
    Maalon Jochmann
"""
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from collections import deque 

from ainex_interfaces.msg import CubeBBoxList, WorldState, CubeSymbolicState, ManipulationEvent
from ainex_interfaces.action import RecordDemo


class SequenceModel(Node):
    def __init__(self):
        super().__init__('sequence_model')

        self.cube_ids = ["red", "green", "blue"]

        #History length ~2 seconds at 30 Hz
        self.history_len = 60
        self.histories = {}
        for cube in self.cube_ids:
            self.histories[cube] = deque(maxlen=self.history_len)

        #PARAMETERS OBSERVATIONS
        self.size_change_filter = 0.05

        #PARAMETERS MOTION
        self.motion_window = 20 
        self.occluded_frames = 8
        self.min_disp_m = 7.0          # px (ignore tiny jitter)
        self.v_moving_thresh = 25.0    # px/s 
        self.v_still_thresh = 4.0    # px/s 

        self.motion = {}
        self.fsm_state = {}
        self.still_count = {}
        self.moving_count = {}
        for cube in self.cube_ids:
            self.motion[cube] = {"moving": False, "still": False, "occluded": True, "speed": 0.0}
            self.fsm_state[cube] = "UNKNOWN"
            self.still_count[cube] = 0
            self.moving_count[cube] = 0

        # PARAMETERS FSM
        self.still_thresh = 8
        self.moving_thresh = 5

        #PARAMETERS WORLD STATE
        self.world_still_thresh = 25
        self.x_overlap_ratio_t = 0.30
        self.y_gap_px_min = 12.0

        self.world_state_locked = False

        self.events = []
        self.world_states = []

        #Action recorder
        self.cb_group = ReentrantCallbackGroup()
        self.recording = False
        self.session_id = 0
        self.last_activity_t = self.get_clock().now()
        self.action_server = ActionServer(
            self,
            RecordDemo,
            "record_demo",
            execute_callback=self.execute_record_demo,
            goal_callback=self.goal_record_demo,
            cancel_callback=self.cancel_record_demo,
            callback_group=self.cb_group,
        )

        #Subscribe to Cube Detection
        self.create_subscription(CubeBBoxList, "cubes_position", self.cb, 10, callback_group=self.cb_group)


    def cb(self, msg: CubeBBoxList) -> None:
        """
        Main subscriber callback for cube detections.

        Pipeline (per incoming CubeBBoxList):
          1) Convert message into per-cube buffered observations (with smoothing
             against sudden size changes).
          2) Derive motion state (moving/still/occluded + speed estimate) per cube.
          3) Update FSM and emit PICK/PLACE events on transitions.
          4) If the scene is stable, infer and (optionally) record a symbolic world
             state describing stacking and relative grouping.

        Args:
            msg: Incoming list of cube bounding boxes for the current frame.
        """
        
        self.convert_msg_to_buffer(msg)
        self.derive_motion()
        self.fsm_pick_place()
        self.build_world_state()


    def convert_msg_to_buffer(self, msg: CubeBBoxList) -> None:
        '''
        Append the latest detection (or a placeholder if missing) to each cube's history.

        - Stores bbox center/size/angle plus a bottom-center point (x=cx, y=cy+0.5*h).
        - If detection is missing, appends an invisible obs carrying forward last values.
        - If bbox size jumps beyond self.size_change_filter, reuse last center to reduce jitter.

        Args:
            msg: Incoming CubeBBoxList for the current frame.
        '''
        stamp = self.get_clock().now()

        for cube in self.cube_ids:
            found_bbox = None

            for bbox in msg.cubes:
                if bbox.id == cube:
                    found_bbox = bbox
                    break

            if found_bbox is not None:
                cx = float(found_bbox.cx)
                cy = float(found_bbox.cy)
                w = float(found_bbox.w)
                h = float(found_bbox.h)
                angle = float(found_bbox.angle)

                last = self.histories[cube][-1] if len(self.histories[cube]) else None
                if last and last.get("visible", False) and last.get("w") and last.get("h"):
                    dw = abs(w - float(last["w"])) / float(last["w"])
                    dh = abs(h - float(last["h"])) / float(last["h"])

                    if (dw > self.size_change_filter) or (dh > self.size_change_filter):
                        if last.get("cx") is not None and last.get("cy") is not None:
                            cx = float(last["cx"])
                            cy = float(last["cy"])

                #Shift py to the bottom center
                px = cx
                py = cy + 0.5 * h

                obs = {
                    "t": stamp,
                    "cube": cube,
                    "visible": True,
                    "cx": cx,
                    "cy": cy,
                    "w": w,
                    "h": h,
                    "angle": angle,
                    "x": px,  
                    "y": py,
                    "z": 0.0,
                }

            else:
                last = self.histories[cube][-1] if len(self.histories[cube]) else None

                obs = {
                    "t": stamp,
                    "cube": cube,
                    "visible": False,
                    "cx": last["cx"] if last else None,
                    "cy": last["cy"] if last else None,
                    "w": last["w"] if last else None,
                    "h": last["h"] if last else None,
                    "angle": last["angle"] if last else None,
                    "x": last["x"] if last else None,
                    "y": last["y"] if last else None,
                    "z": 0.0,
                }

            self.histories[cube].append(obs)


    def derive_motion(self) -> None:
        """
        Compute per-cube motion flags from recent history.

        - Occluded if last self.occluded_frames are all invisible.
        - Speed from oldest->newest visible observation within the motion window
        - moving if speed > self.v_moving_thresh, still if speed < self.v_still_thresh
        (both disabled when occluded). Small motion is ignored.
        """
        for cube in self.cube_ids:
            hist = self.histories.get(cube, None)
            if not hist or len(hist) < 2:
                self.motion[cube] = {"moving": False, "still": False, "occluded": False, "speed": 0.0}
                continue

            #Occlusion check
            last_n = list(hist)[-self.occluded_frames:]
            occluded = (len(last_n) == self.occluded_frames) and all(
                (not obs.get("visible", False)) for obs in last_n
            )

            #Build window with only valid observations
            recent = list(hist)[-self.motion_window:]
            visible = []
            for obs in recent:
                if not obs.get("visible", False):
                    continue
                if obs.get("cx") is None:
                    continue
                if obs.get("cy") is None:
                    continue
                if obs.get("t") is None:
                    continue
                if obs.get("h") is None:
                    continue
                visible.append(obs)

            speed = 0.0
            if len(visible) >= 2:
                old_obs = visible[0]
                new_obs = visible[-1]

                dt_ns = (new_obs["t"] - old_obs["t"]).nanoseconds
                dt = dt_ns / 1e9 if dt_ns > 0 else 0.0

                if dt > 0.0:
                    x_old = float(old_obs["cx"])
                    y_old = float(old_obs["cy"]) + 0.5 * float(old_obs["h"])

                    x_new = float(new_obs["cx"])
                    y_new = float(new_obs["cy"]) + 0.5 * float(new_obs["h"])

                    dx = x_new - x_old
                    dy = y_new - y_old
                    dist = (dx * dx + dy * dy) ** 0.5

                    if dist < self.min_disp_m:
                        speed = 0.0
                    else:
                        speed = dist / dt

            moving = (speed > self.v_moving_thresh) and (not occluded)
            still  = (speed < self.v_still_thresh) and (not occluded)

            self.motion[cube] = {"moving": moving, "still": still, "occluded": occluded, "speed": speed,}


    def fsm_pick_place(self) -> None:
        '''
        Per-cube finite-state machine to detect PICK and PLACE events.

        The FSM tracks three states per cube:
          - "UNKNOWN": initial state until stable motion classification is observed
          - "STILL":   cube considered stationary on the surface/stack
          - "MOVING":  cube considered manipulated (or occluded)

        Debouncing is done using counters:
          - still_count increments when motion indicates still, else resets
          - moving_count increments when motion indicates moving, else resets
          - transitions require counts to exceed:
              self.still_thresh  (to enter STILL)
              self.moving_thresh (to enter MOVING)

        Event logic:
          - STILL  -> MOVING : emit "PICK"
          - MOVING -> STILL  : emit "PLACE"

        Recording behavior:
          - If self.recording is True, events are appended to self.events with:
              stamp, type ("PICK"/"PLACE"), cube_id
          - last_activity_t is updated on recorded events.

        Notes:
          - Occluded cubes are forced to count as moving (to avoid false STILL).
        '''
        event_t = self.get_clock().now().to_msg()

        for cube in self.cube_ids:
            motion = self.motion.get(cube, None)
            if motion is None:
                continue

            occ_flag = bool(motion.get("occluded", True))
            cube_moving = bool(motion.get("moving", False))
            cube_still  = bool(motion.get("still", False))

            # Occluded cubes count as moving
            if occ_flag:
                cube_moving = True
                cube_still = False

            if cube_still:
                self.still_count[cube] = self.still_count[cube] + 1
            else:
                self.still_count[cube] = 0

            if cube_moving:
                self.moving_count[cube] = self.moving_count[cube] + 1
            else:
                self.moving_count[cube] = 0

            previous_state = self.fsm_state[cube]
            new_state = previous_state

            if previous_state == "UNKNOWN":
                if self.still_count[cube] >= self.still_thresh:
                    new_state = "STILL"
                elif self.moving_count[cube] >= self.moving_thresh:
                    new_state = "MOVING"

            elif previous_state == "STILL":
                if self.moving_count[cube] >= self.moving_thresh:
                    new_state = "MOVING"

            elif previous_state == "MOVING":
                if self.still_count[cube] >= self.still_thresh:
                    new_state = "STILL"

            if (previous_state == "STILL") and (new_state == "MOVING"):
                if self.recording:
                    e = ManipulationEvent()
                    e.stamp = event_t
                    e.type = "PICK"
                    e.cube_id = cube
                    self.events.append(e)
                    self.last_activity_t = self.get_clock().now()
                self.get_logger().info(f'PICK: {cube}')

            elif (previous_state == "MOVING") and (new_state == "STILL"):
                if self.recording:
                    e = ManipulationEvent()
                    e.stamp = event_t
                    e.type = "PLACE"
                    e.cube_id = cube
                    self.events.append(e)
                    self.last_activity_t = self.get_clock().now()
                self.get_logger().info(f'PLACE: {cube}')

            self.fsm_state[cube] = new_state


    def build_world_state(self) -> None:
        '''
        Infer and optionally record a discrete symbolic world state from stable detections.

        Locking / stability criteria (must hold for ALL cubes):
          - Not occluded
          - Classified as still
          - still_count >= self.world_still_thresh
        If criteria are met and the world state is not already locked, a snapshot is built.

        World-state inference:
          1) Gather latest visible bbox geometry per cube.
          2) Infer direct support relations (supports[cube] = cube_below or None):
             - Candidate below cube must have larger y (visually lower).
             - Must overlap sufficiently in x (overlap_ratio >= self.x_overlap_ratio_t).
             - Must have a small vertical gap between top(below) and bottom(above),
               within a gap threshold (self.y_gap_px_min or 0.25*min(h1,h2)).
             - Choose best candidate by (overlap_ratio, -gap).
          3) Compute stack height per cube by following supports links.
          4) Compute a coarse horizontal grouping label (vert):
             - If one base exists (all stacked): set all "middle"
             - If two bases exist: label bases "left"/"right" by x-order and
               propagate label up each support chain
             - If three bases exist (no stacking): label by x-order as
               "left", "middle", "right"
          5) For each cube, produce:
             - level: "table" if no support, else f"on_{below_cube}"
             - vert:  one of {"left","middle","right"} (coarse grouping)
             - height: integer stack height (0 on base)

        Recording behavior:
          - If self.recording is True, append a WorldState message to self.world_states
            containing CubeSymbolicState entries for each cube.
          - Sets self.world_state_locked = True after recording a snapshot and updates
            last_activity_t.
          - If not recording, world_state_locked is cleared.

        Notes:
          - This method assumes image coordinates with y increasing downward.
        '''
        #Locking conditions: All cubes still with still count over threshold + not occluded
        for cube in self.cube_ids:
            m = self.motion.get(cube, None)
            if m is None:
                self.world_state_locked = False
                return

            if bool(m.get("occluded", True)):
                self.world_state_locked = False
                return
            
            if not bool(m.get("still", False)):
                self.world_state_locked = False
                return
            
            if self.still_count.get(cube, 0) < self.world_still_thresh:
                self.world_state_locked = False
                return
            
        if self.world_state_locked: return

        latest = {}
        latest_times = []

        for cube in self.cube_ids:
            if len(self.histories[cube]) == 0:
                self.world_state_locked = False
                return

            obs = self.histories[cube][-1]
            if not obs.get("visible", False):
                self.world_state_locked = False
                return

            # need bbox geometry for stacking logic
            if (obs.get("cx") is None) or (obs.get("cy") is None) or (obs.get("w") is None) or (obs.get("h") is None):
                self.world_state_locked = False
                return

            latest[cube] = obs
            latest_times.append(obs["t"])

        t_ws = max(latest_times)
        t_ws_msg = t_ws.to_msg()

        supports = {}
        for cube in self.cube_ids:
            supports[cube] = None

        #Stacking Inference
        for cube1 in self.cube_ids:
            cx1, cy1, w1, h1, l1, r1, t1, b1 = self.bbox(latest[cube1])

            best_2 = None
            best_score = None

            for cube2 in self.cube_ids:
                if cube2 == cube1:
                    continue

                cx2, cy2, w2, h2, l2, r2, t2, b2 = self.bbox(latest[cube2])

                #Cube 2 is under Cube 1 -> larger y value is below
                if cy2 <= cy1:
                    continue

                #X is Overlapping
                overlap = max(0.0, min(r1, r2) - max(l1, l2))
                denom = min(w1, w2)
                if denom <= 1e-6:
                    continue
                overlap_ratio = overlap / denom
                if overlap_ratio < self.x_overlap_ratio_t:
                    continue

                #Y values are closeby
                gap = abs(t2 - b1)
                gap_thresh = max(self.y_gap_px_min, 0.25*min(h1, h2))
                if gap > gap_thresh:
                    continue

                #Evaluate Inference
                score = (overlap_ratio, -gap)
                if best_score is None or score > best_score:
                    best_score = score
                    best_2 = cube2

            #Take cube that fits description of inference best as direct support
            supports[cube1] = best_2

        heights = {}
        bases = []
        vert = {}
        for cube in self.cube_ids:
            heights[cube] = self.height_of(cube, supports)
            if supports[cube] is None:
                bases.append(cube)
            vert[cube] = "middle"

        #All 3 cubes are stacked
        if len(bases) == 1:
            for cube in self.cube_ids:
                vert[cube] = "middle"

        #There are two cubes touching the table
        elif len(bases) == 2:
            bases_sorted = sorted(bases, key=lambda cube: float(latest[cube]["cx"]))
            left_base, right_base = bases_sorted[0], bases_sorted[1]
            vert[left_base] = "left"
            vert[right_base] = "right"

            for cube in self.cube_ids:
                base = self.base_from_supports(cube, supports)
                if base == left_base:
                    vert[cube] = "left"
                elif base == right_base:
                    vert[cube] = "right"
        
        #All cubes are unstacked on the table
        else:
            order = sorted(self.cube_ids, key=lambda cube: float(latest[cube]["cx"]))
            vert[order[0]] = "left"
            vert[order[1]] = "middle"
            vert[order[2]] = "right"

        #Cube states
        state = {}
        for cube in self.cube_ids:
            below = supports[cube]
            if below is None:
                level = "table"
            else:
                level = f"on_{below}"

            state[cube] = {
                "vert": vert[cube],
                "level": level,
                "height": heights[cube],
            }
        
        if self.recording:
            ws = WorldState()
            ws.stamp = t_ws_msg

            cubes_msg = []
            for cube in self.cube_ids:
                cs = CubeSymbolicState()
                cs.cube_id = cube
                cs.vert = state[cube]["vert"]
                cs.level = state[cube]["level"]
                cs.height = int(state[cube]["height"])
                cubes_msg.append(cs)

            ws.cubes = cubes_msg

            self.world_states.append(ws)
            self.world_state_locked = True
            self.last_activity_t = self.get_clock().now()
        else:
            self.world_state_locked = False



    #HELPER FUNCTIONS --------------------------------------------------
    def bbox(self, cube):
        '''Compute bbox edges (left/right/top/bottom) from an observation dict.'''
        cx = float(cube["cx"])
        cy = float(cube["cy"])
        w = float(cube["w"])
        h = float(cube["h"])
        left = cx - 0.5 * w
        right = cx + 0.5 * w
        top = cy - 0.5 * h
        bottom = cy + 0.5 * h
        return cx, cy, w, h, left, right, top, bottom

    def height_of(self, cube: str, supports:dict) -> int:
        '''Return stack height (0..2) by following support links downward.'''
        h = 0
        cur = cube
        seen = set()
        while supports.get(cur, None) is not None:
            if cur in seen:
                break  # cycle guard
            seen.add(cur)
            cur = supports[cur]
            h += 1
            if h >= 2:
                break
        return h

    def base_from_supports(self, cube: str, supports: dict) -> str:
        '''Return the base cube of a support chain.'''
        cur = cube
        seen = set()
        while supports.get(cur, None) is not None and cur not in seen:
            seen.add(cur)
            cur = supports[cur]
        return cur


    #SERVER FUNCTIONS --------------------------------------------------
    def reset_episode(self) -> None:
        '''Reset recording buffers and per-cube tracking state for a new episode.'''
        self.events = []
        self.world_states = []
        self.world_state_locked = False

        for cube in self.cube_ids:
            self.fsm_state[cube] = "UNKNOWN"
            self.still_count[cube] = 0
            self.moving_count[cube] = 0
            self.histories[cube].clear()
            self.motion[cube] = {"moving": False, "still": False, "occluded": True, "speed": 0.0}

    def goal_record_demo(self, goal_request: RecordDemo.Goal) -> GoalResponse:
        '''Accept a goal only if not currently recording.'''
        if self.recording:
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT
    
    def cancel_record_demo(self, goal_handle) -> CancelResponse:
        '''Accept action cancellation requests.'''
        return CancelResponse.ACCEPT

    def execute_record_demo(self, goal_handle):
        '''Record events/world states until canceled or idle completion, then return results.'''
        goal = goal_handle.request

        self.session_id += 1
        session_id = self.session_id
        
        self.reset_episode()
        self.recording = True
        self.last_activity_t = self.get_clock().now()

        self.get_logger().info(
            f"RecordDemo START session={session_id} min_world_states={goal.min_world_states} idle_timeout={goal.idle_timeout_sec}s"
        )

        feedback = RecordDemo.Feedback()
        feedback.status = "recording"

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self.recording = False
                goal_handle.canceled()
                self.get_logger().info(f"RecordDemo CANCELED session={session_id}")
                break
        
            feedback.num_events = len(self.events)
            feedback.num_world_states = len(self.world_states)
            feedback.status = "recording"
            goal_handle.publish_feedback(feedback)

            now = self.get_clock().now()
            idle_dt = (now - self.last_activity_t).nanoseconds / 1e9

            if (len(self.world_states) >= int(goal.min_world_states)) and self.world_state_locked:
                if idle_dt >= float(goal.idle_timeout_sec):
                    self.recording = False
                    goal_handle.succeed()
                    self.get_logger().info(f"RecordDemo DONE session={session_id}")
                    break

            time.sleep(0.1)

        result = RecordDemo.Result()
        result.session_id = session_id
        result.events = self.events
        result.world_states = self.world_states
        
        return result





def main(args=None):
    rclpy.init(args=args)
    node = SequenceModel()

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
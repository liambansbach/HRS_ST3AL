#!/usr/bin/env python3
"""
This node listens to the /cubes_position topic, where every message contains the current bounding boxes
for the red, green, and blue cubes. Each time a message arrives, the callback runs a small processing pipeline:
<<<<<<< HEAD
it first reads the cube data (IDs + bbox center/size), then stores it in a short history so it can compare
the current frame to the last few frames. Using that history, it estimates whether each cube is moving,
still, or temporarily missing (occluded). Based on those motion signals, it runs a simple state machine per
cube to detect events like “this cube was picked up” (was still → starts moving/disappears) and “this cube
was placed” (was moving → becomes still again). When a place event happens, it then compares the placed
cube’s position to the other cubes to decide basic relations like left/right/front/behind and whether it
looks like it was put on top of another cube. Finally, it logs or publishes the detected events and the
current symbolic “stacking/placement state,” which is what you’ll later use to imitate the demonstrated
manipulation sequence.

Subscribes to CubeBBoxList messages on /cubes_position

---conversion node
TODO: Change Cube Detection (Stabelize + Use Only Center)
TODO: Project Center into 3D Workspace coordinates

---this node
TODO: Get reliable World State Updates (better bag needed)
TODO: Publishing
=======
    it first reads the cube data (IDs + bbox center/size), then stores it in a short history so it can compare
    the current frame to the last few frames. Using that history, it estimates whether each cube is moving,
    still, or temporarily missing (occluded). Based on those motion signals, it runs a simple state machine per
    cube to detect events like “this cube was picked up” (was still → starts moving/disappears) and “this cube
    was placed” (was moving → becomes still again). When a place event happens, it then compares the placed
    cube’s position to the other cubes to decide basic relations like left/right/front/behind and whether it
    looks like it was put on top of another cube. Finally, it logs or publishes the detected events and the
    current symbolic “stacking/placement state,” which is what you’ll later use to imitate the demonstrated
    manipulation sequence.

Subscribes to CubeBBoxList messages on /cubes_position

hook functions for: ingesting observations, maintaining per-cube history, detecting
pick/place events, extracting relative relations (left/right/front/behind + stacking),

Publishes the resulting symbolic sequence.
>>>>>>> 3e05762 (Sequence Model running,, no reliable world state)
"""

import rclpy
from rclpy.node import Node
from collections import deque #double ended queue as buffer

<<<<<<< HEAD
from ainex_interfaces.msg import ManipulationSeq, CubeBBoxList
=======
from ainex_interfaces.msg import ManipulationSeq, ManipulationStep, CubeBBoxList
>>>>>>> 3e05762 (Sequence Model running,, no reliable world state)


class SequenceModel(Node):
    def __init__(self):
        super().__init__('sequence_model')

        self.debug_cube_id = "green"   # set to "red"/"green"/"blue"

        self._last_motion_log_time = self.get_clock().now()
        self._motion_log_period_ns = int(0.5 * 1e9)  # twice per second
        
        self.cube_ids = ["red", "green", "blue"]
        self.frame_obs = {}
        self.latest_msg = None

        # History length (how many frames to remember per cube)
        self.history_len = 60  # e.g., ~2 seconds at 30 Hz; adjust later

        # Per-cube history buffers: each entry is the observation dict you already build in frame_obs
        self.histories = {}
        for cid in self.cube_ids:
            self.histories[cid] = deque(maxlen=self.history_len)

<<<<<<< HEAD
        # ---- Motion estimation params (TUNE HERE) ----
=======
    # ---- Motion estimation params (TUNE HERE) ----
>>>>>>> 3e05762 (Sequence Model running,, no reliable world state)
        self.motion_window = 10         # how many recent frames to consider
        self.v_moving_thresh = 15.0     # > this => moving
        self.v_still_thresh = 7.0       # < this => still
        self.occluded_frames = 10        # N consecutive invisible frames => occluded
        self.min_disp_px = 2.0  # ignore tiny drift across the window

        self.motion = {}
        for cid in self.cube_ids:
            self.motion[cid] = {"moving": False, "still": False, "occluded": False, "speed": 0.0}

        self.fsm_state = {}
        self.still_count = {}
        for cid in self.cube_ids:
            self.fsm_state[cid] = "UNKNOWN"
            self.still_count[cid] = 0
        self.events = []
        self.still_threshold = 3  # frames

        self.last_fsm_state = {}
        for cid in self.cube_ids:
            self.last_fsm_state[cid] = self.fsm_state[cid]

        self.moving_count = {}
        for cid in self.cube_ids:
            self.moving_count[cid] = 0

        self.moving_threshold = 3   # frames required to confirm MOVING
        self.still_threshold = 5    # frames required to confirm STILL (you can tune)

        # --- Sequence model storage ---
        self.sequence = []          # list of recorded PLACE steps over time
        self.support_of = {}        # current world state: cube -> "table" or other cube

        for cid in self.cube_ids:
            self.support_of[cid] = "table"

        # --- Relation / stacking thresholds (tune) ---
        self.relation_x_band = 20.0     # px: deadband for left/right
        self.relation_y_band = 20.0     # px: deadband for above/below

        self.stack_cx_align_thresh = 25.0  # px: how aligned cx must be to count as stacked
        self.stack_dy_min = 12.0           # px: placed must be at least this much above base
        self.stack_dy_max = 140.0          # px: placed must not be too far above base

        
        # --- World-state change gating ---
        self.world_changes = []          # accepted world states over time (compact timeline)
        self._pending_world_sig = None   # candidate state signature waiting to be confirmed
        self._pending_count = 0
        self._confirm_frames = 5         # require this many consecutive frames to accept a change
        self._last_committed_sig = None  # last accepted signature

        # Optional: ignore updates when not all cubes visible (prevents occlusion noise)
        self.require_all_visible_for_commit = True


        # Subscribe to Cubes
        self.sub = self.create_subscription(
            CubeBBoxList,
            '/cubes_position',
            self.cb,
            10
        )

        # publish symbolic sequence / targets
        self.seq_pub = self.create_publisher(
            ManipulationSeq,
            "/ainex_seq_model/sequence",
            10
        )

    def cb(self, msg: CubeBBoxList) -> None:
        """Receive the latest cube detections and pass them through the processing pipeline."""
        
        #Ingest and cache observations (latest message + per-cube snapshot)
        self._ingest_observations(msg)

        #Update per-cube histories (positions/sizes/visibility over time)
        self._update_histories()

        #Estimate motion/stillness/occlusion for each cube from history
        self._update_motion_estimates()

        #Run per-cube finite state machines to detect PICK/PLACE events
        self._run_event_fsm()

        self._process_new_events()
                                                                                             




    def _ingest_observations(self, msg: CubeBBoxList) -> None:    
        """Convert incoming CubeBBoxList into a normalized per-cube observation snapshot.

        Creates a dict for the current frame:
          self.frame_obs[cube_id] = {
            "t": <node time>,
            "visible": bool,
            "cx": float, "cy": float, "w": float, "h": float, "angle": float
          }

        Missing cubes are included with visible=False so downstream logic can detect occlusions.
        """
        # Timestamp this "frame" with the node clock (works even if msg.header.stamp is 0)
        now = self.get_clock().now()

        # Start with all expected cubes as "not visible" this frame
        frame = {}
        for cube_id in self.cube_ids:
            frame[cube_id] = {
                "t": now,
                "visible": False,
                "cx": None,
                "cy": None,
                "w": None,
                "h": None,
                "angle": None,
            }

        # Fill in observations from the message
        for c in msg.cubes:
            cid = c.id.strip()  # be tolerant to whitespace
            if cid not in frame:
                # Unknown ID -> ignore but log once in a while
                self.get_logger().warn(f"Unknown cube id '{cid}' received on /cubes_position; ignoring.")
                continue

            frame[cid] = {
                "t": now,
                "visible": True,
                "cx": float(c.cx),
                "cy": float(c.cy),
                "w": float(c.w),
                "h": float(c.h),
                "angle": float(c.angle),
            }

        # Store for later pipeline steps
        self.frame_obs = frame
        self.latest_msg = msg

        #self.get_logger().info(f"frame_obs keys: {list(self.frame_obs.keys())}")
        #for cid, o in self.frame_obs.items():
        #    self.get_logger().info(f"{cid}: visible={o['visible']} cx={o['cx']} cy={o['cy']}")



    def _update_histories(self) -> None:
        """Append the latest per-cube observations into fixed-length history buffers.

        After this runs:
          self.histories["red"]  -> deque of last N obs dicts for red
          self.histories["green"]-> deque of last N obs dicts for green
          self.histories["blue"] -> deque of last N obs dicts for blue

        Each stored obs is exactly what _ingest_observations() created, including visible=False frames.
        """
        if not self.frame_obs:
            return  # nothing ingested yet

        for cid in self.cube_ids:
            obs = self.frame_obs.get(cid, None)
            if obs is None:
                # If for some reason it's missing, store an explicit "not visible" placeholder
                obs = {
                    "t": self.get_clock().now(),
                    "visible": False,
                    "cx": None,
                    "cy": None,
                    "w": None,
                    "h": None,
                    "angle": None,
                }

            self.histories[cid].append(obs)

            #self.get_logger().info(
            #    "history sizes: " + ", ".join([f"{cid}={len(self.histories[cid])}" for cid in self.cube_ids])
            #)


    def _update_motion_estimates(self) -> None:
        """Compute moving/still/occluded flags per cube using a true window-based speed.

        Speed is computed between the OLDEST and NEWEST visible observations within the last
        self.motion_window frames (≈0.33s at 30 Hz if motion_window=10).
        """
        for cid in self.cube_ids:
            hist = self.histories.get(cid, None)
            if not hist or len(hist) < 2:
                self.motion[cid] = {"moving": False, "still": False, "occluded": False, "speed": 0.0}
                continue

            # --- occlusion: last N frames invisible ---
            last_n = list(hist)[-self.occluded_frames:]
            occluded = (len(last_n) == self.occluded_frames) and all(not o["visible"] for o in last_n)

            # --- window selection ---
            recent = list(hist)[-self.motion_window:]

            # Keep only valid visible observations in the window
            visible = [
                o for o in recent
                if o["visible"] and o["cx"] is not None and o["cy"] is not None
            ]

            speed = 0.0
            if len(visible) >= 2:
                o_old = visible[0]    # oldest visible in window
                o_new = visible[-1]   # newest visible in window

                dt_ns = (o_new["t"] - o_old["t"]).nanoseconds
                dt = dt_ns / 1e9 if dt_ns > 0 else 0.0

                if dt > 0.0:
                    dx = float(o_new["cx"]) - float(o_old["cx"])
                    dy = float(o_new["cy"]) - float(o_old["cy"])
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist < self.min_disp_px:
                        speed = 0.0
                    else:
                        speed = dist / dt

            # Decide states
            moving = (speed > self.v_moving_thresh) and (not occluded)
            still = (speed < self.v_still_thresh) and (not occluded)

            self.motion[cid] = {
                "moving": moving,
                "still": still,
                "occluded": occluded,
                "speed": speed,
            }

        #now = self.get_clock().now()
        #if (now - self._last_motion_log_time).nanoseconds >= self._motion_log_period_ns:
        #    self._last_motion_log_time = now
        #    self.get_logger().info(f"{self.motion}")



    def _run_event_fsm(self) -> None:
        """FSM with hysteresis using consecutive-frame confirmation.

        Fixes:
        - Detects slow motion (no dead zone): moving_candidate = speed >= v_still_thresh
        - Prevents bounce: requires N consecutive frames to switch states
        """
        for cid in self.cube_ids:
            motion = self.motion.get(cid, None)
            obs = self.frame_obs.get(cid, None)

            if motion is None:
                continue
            if obs is None:
                continue

            visible_sig = bool(obs.get("visible", False))
            occluded_sig = bool(motion.get("occluded", False))
            speed = float(motion.get("speed", 0.0))

            # Candidates based on speed (remove dead-zone)
            still_candidate = False
            moving_candidate = False

            if visible_sig and (not occluded_sig):
                if speed < self.v_still_thresh:
                    still_candidate = True
                else:
                    moving_candidate = True

            # If occluded while previously stable, treat as moving (picked/covered)
            if occluded_sig and (not visible_sig):
                moving_candidate = True
                still_candidate = False

            # Update consecutive counters
            if still_candidate:
                self.still_count[cid] = self.still_count[cid] + 1
            else:
                self.still_count[cid] = 0

            if moving_candidate:
                self.moving_count[cid] = self.moving_count[cid] + 1
            else:
                self.moving_count[cid] = 0

            prev_state = self.fsm_state[cid]
            new_state = prev_state

            # Transitions with thresholds
            if prev_state == "UNKNOWN":
                if self.still_count[cid] >= self.still_threshold:
                    new_state = "STILL"
                elif self.moving_count[cid] >= self.moving_threshold:
                    new_state = "MOVING"

            elif prev_state == "STILL":
                if self.moving_count[cid] >= self.moving_threshold:
                    new_state = "MOVING"

            elif prev_state == "MOVING":
                if self.still_count[cid] >= self.still_threshold:
                    new_state = "STILL"
                    self.events.append({"t": self.get_clock().now(), "type": "PLACE", "cube": cid}) 
                

            else:
                new_state = "UNKNOWN"
                self.still_count[cid] = 0
                self.moving_count[cid] = 0

            # Commit + log only transitions (one cube)
            if new_state != prev_state:
                self.fsm_state[cid] = new_state

                #if cid == self.debug_cube_id:
                 #   self.get_logger().info(f"FSM {cid}: {prev_state} -> {new_state}")

            else:
                self.fsm_state[cid] = new_state



    def _infer_support_from_cx_cy(self, placed_id: str) -> str:
        """Return 'table' or the cube id that placed_id is stacked on (cx/cy only)."""
        placed_obs = self.frame_obs.get(placed_id, None)
        if placed_obs is None:
            return "table"
        if not placed_obs.get("visible", False):
            return "table"
        if placed_obs.get("cx", None) is None or placed_obs.get("cy", None) is None:
            return "table"

        placed_cx = float(placed_obs["cx"])
        placed_cy = float(placed_obs["cy"])

        best_base = "table"
        best_score = 1e18

        for base_id in self.cube_ids:
            if base_id == placed_id:
                continue

            base_obs = self.frame_obs.get(base_id, None)
            if base_obs is None:
                continue
            if not base_obs.get("visible", False):
                continue
            if base_obs.get("cx", None) is None or base_obs.get("cy", None) is None:
                continue

            base_cx = float(base_obs["cx"])
            base_cy = float(base_obs["cy"])

            dx = abs(placed_cx - base_cx)
            dy = base_cy - placed_cy  # >0 means base is below placed in the image

            if dx > self.stack_cx_align_thresh:
                continue
            if dy < self.stack_dy_min:
                continue
            if dy > self.stack_dy_max:
                continue

            score = dx + 0.5 * dy
            if score < best_score:
                best_score = score
                best_base = base_id

        return best_base



    def _record_place_step(self, placed_id: str) -> dict | None:
        """Record a PLACE: support (stack/table) + left/right + above/below relations."""
        placed_obs = self.frame_obs.get(placed_id, None)
        if placed_obs is None:
            return None
        if not placed_obs.get("visible", False):
            return None
        if placed_obs.get("cx", None) is None or placed_obs.get("cy", None) is None:
            return None

        placed_cx = float(placed_obs["cx"])
        placed_cy = float(placed_obs["cy"])

        # 1) stacking support
        support = self._infer_support_from_cx_cy(placed_id)
        self.support_of[placed_id] = support

        # 2) relations to other cubes (skip the support cube to keep stacking separate)
        left_of = []
        right_of = []
        above = []
        below = []

        for other_id in self.cube_ids:
            if other_id == placed_id:
                continue
            if other_id == support:
                continue

            other_obs = self.frame_obs.get(other_id, None)
            if other_obs is None:
                continue
            if not other_obs.get("visible", False):
                continue
            if other_obs.get("cx", None) is None or other_obs.get("cy", None) is None:
                continue

            other_cx = float(other_obs["cx"])
            other_cy = float(other_obs["cy"])

            dx = placed_cx - other_cx
            dy = placed_cy - other_cy

            if dx < -self.relation_x_band:
                left_of.append(other_id)
            elif dx > self.relation_x_band:
                right_of.append(other_id)

            if dy < -self.relation_y_band:
                above.append(other_id)
            elif dy > self.relation_y_band:
                below.append(other_id)

        t_sec = self.get_clock().now().nanoseconds / 1e9

        step = {
            "t": t_sec,
            "type": "PLACE",
            "cube": placed_id,
            "support": support,  # "table" or cube id
            "relations": {
                "left_of": left_of,
                "right_of": right_of,
                "above": above,
                "below": below,
            }
        }

        self.sequence.append(step)

        # if placed_id == self.debug_cube_id:
        #     self.get_logger().info(f"SEQ PLACE {placed_id} support={support} L={left_of} R={right_of} A={above} B={below}")
        
        return step



    def _process_new_events(self) -> None:
        """Handle new PLACE events by recording sequence steps."""
        if not self.events:
            return

        for e in self.events:
            if e.get("type", "") == "PLACE":
                placed_id = e.get("cube", "")
                if placed_id in self.cube_ids:
                    last_step = self._record_place_step(placed_id)

        self.events = []

        self._evaluate_world_change(last_step)



    def _build_world_state(self) -> dict | None:
        """Build a snapshot of the current symbolic world state.
        Returns dict with support map and pairwise left/right/above/below relations.
        """

        # Optionally require all cubes visible for clean relation inference
        if self.require_all_visible_for_commit:
            for cid in self.cube_ids:
                o = self.frame_obs.get(cid, {})
                if not o.get("visible", False):
                    return None
                if o.get("cx") is None or o.get("cy") is None:
                    return None

        state = {
            "t": self.get_clock().now().nanoseconds / 1e9,
            "support_of": dict(self.support_of),
            "relations": {}
        }

        # Compute relations from current frame for each cube
        for a in self.cube_ids:
            aobs = self.frame_obs.get(a, {})
            if not aobs.get("visible", False):
                continue
            acx, acy = float(aobs["cx"]), float(aobs["cy"])

            rel = {"left_of": [], "right_of": [], "above": [], "below": []}

            for b in self.cube_ids:
                if b == a:
                    continue
                bobs = self.frame_obs.get(b, {})
                if not bobs.get("visible", False):
                    continue
                bcx, bcy = float(bobs["cx"]), float(bobs["cy"])

                dx = acx - bcx
                dy = acy - bcy

                if dx < -self.relation_x_band:
                    rel["left_of"].append(b)
                elif dx > self.relation_x_band:
                    rel["right_of"].append(b)

                if dy < -self.relation_y_band:
                    rel["above"].append(b)
                elif dy > self.relation_y_band:
                    rel["below"].append(b)

            # stable ordering
            for k in rel:
                rel[k].sort()

            state["relations"][a] = rel

        return state
    


    def _log_stack_chains(self, support_of: dict) -> None:
        """Log stacks as bottom->top chains, e.g. red>green, blue."""
        cubes = list(support_of.keys())
        supported = set(support_of.values())
        tops = [c for c in cubes if c not in supported]

        chains = []
        for top in tops:
            chain = [top]
            cur = top
            seen = {cur}
            while True:
                base = support_of.get(cur, "table")
                if base == "table":
                    break
                if base in seen:
                    break
                chain.append(base)
                seen.add(base)
                cur = base
            chain.reverse()
            chains.append(chain)

        chains.sort(key=lambda ch: (len(ch), ch))
        pretty = " | ".join(">".join(ch) for ch in chains) if chains else "(none)"
        self.get_logger().info(f"Stacks: {pretty}")



    def _world_signature(self, state: dict) -> str:
        """Create a deterministic signature for supports + relations."""
        sup = state["support_of"]
        sup_part = "|".join(f"{cid}->{sup.get(cid,'table')}" for cid in sorted(self.cube_ids))

        rel = state["relations"]
        rel_parts = []
        for cid in sorted(rel.keys()):
            r = rel[cid]
            rel_parts.append(
                f"{cid}:L[{','.join(r['left_of'])}]"
                f"R[{','.join(r['right_of'])}]"
                f"A[{','.join(r['above'])}]"
                f"B[{','.join(r['below'])}]"
            )
        rel_part = "|".join(rel_parts)

        return sup_part + "||" + rel_part



    def _evaluate_world_change(self, placed_step: dict | None) -> None:
        """Decide if current world state should be committed as a 'world change'."""
        state = self._build_world_state()
        if state is None:
            # not enough info to decide (e.g., occluded)
            self._pending_world_sig = None
            self._pending_count = 0
            return

        sig = self._world_signature(state)

        # Initialize committed signature on first valid state (no event yet)
        if self._last_committed_sig is None:
            self._last_committed_sig = sig
            self.world_changes.append(state)
            self.get_logger().info(f"[WORLD INIT] {sig}")
            return

        # If no actual difference, clear pending
        if sig == self._last_committed_sig:
            self._pending_world_sig = None
            self._pending_count = 0
            return

        # Optional consistency check: only accept a new world if it matches the placed cube support
        # This prevents unrelated relation flicker from being accepted.
        if placed_step is not None:
            placed_cube = placed_step.get("cube", "")
            placed_support = placed_step.get("support", None)
            if placed_cube in self.cube_ids and placed_support is not None:
                # require that the support map reflects that step
                if state["support_of"].get(placed_cube, "table") != placed_support:
                    return

        # Stability gate: require N consecutive frames with the same sig
        if sig != self._pending_world_sig:
            self._pending_world_sig = sig
            self._pending_count = 1
            return
        else:
            self._pending_count += 1

        if self._pending_count >= self._confirm_frames:
            # Commit!
            self._last_committed_sig = sig
            self._pending_world_sig = None
            self._pending_count = 0

            self.world_changes.append(state)

            # Print something useful: the stack chains
            self.get_logger().info(f"[WORLD CHANGE] {sig}")
            self._log_stack_chains(state["support_of"])



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

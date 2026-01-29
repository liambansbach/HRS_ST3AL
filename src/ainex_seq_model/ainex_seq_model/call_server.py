#!/usr/bin/env python3
'''
Create an action client that can start (send goal) and stop (cancel goal) demo recording.

HRS 2025 - Group B:
    Liam Bansbach
    Marius Moe Rena
    Niklas Peter
    Tobias Töws
    Maalon Jochmann
'''
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from ainex_interfaces.action import RecordDemo


class RecordDemoClient(Node):
    def __init__(self):
        super().__init__("record_demo_client")

        self.client = ActionClient(self, RecordDemo, "record_demo")

        self.goalHandle = None
        self.resultFuture = None

        #PARAMETERS FOR RECORDING
        self.minWorldStates = 3
        self.idleTimeoutSec = 10.0

        #Simple toggle state
        self.recordingActive = False

    def startRecording(self) -> None:
        '''Start recording by sending a RecordDemo goal to the server (this is the 'START' condition).'''
        if self.recordingActive:
            self.get_logger().warn("Already recording; ignore start.")
            return

        self.get_logger().info("Waiting for action server...")
        if not self.client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Action server 'record_demo' not available.")
            return

        goal = RecordDemo.Goal()
        goal.min_world_states = int(self.minWorldStates)
        goal.idle_timeout_sec = float(self.idleTimeoutSec)

        self.get_logger().info(
            f"START: sending goal (min_world_states={goal.min_world_states}, idle_timeout_sec={goal.idle_timeout_sec})"
        )

        future = self.client.send_goal_async(goal, feedback_callback=self.feedbackCb)
        future.add_done_callback(self.goalResponseCb)

    def stopRecording(self) -> None:
        '''Stop recording by canceling the active goal (this is the explicit 'STOP' condition).'''
        if not self.recordingActive or self.goalHandle is None:
            self.get_logger().warn("Not recording; nothing to stop.")
            return

        self.get_logger().warn("STOP: canceling active goal...")
        cancelFuture = self.goalHandle.cancel_goal_async()
        cancelFuture.add_done_callback(self.cancelDoneCb)

    def toggleRecording(self) -> None:
        '''Toggle between START (send goal) and STOP (cancel goal), so a user/robot can decide boundaries.'''
        if self.recordingActive:
            self.stopRecording()
        else:
            self.startRecording()

    def goalResponseCb(self, future) -> None:
        '''Handle server accept/reject; if accepted, recording is considered started until result/cancel returns.'''
        goalHandle = future.result()
        if not goalHandle.accepted:
            self.get_logger().error("Goal rejected by server.")
            self.recordingActive = False
            self.goalHandle = None
            return

        self.goalHandle = goalHandle
        self.recordingActive = True
        self.get_logger().info("Server accepted goal: recording ACTIVE.")

        self.resultFuture = goalHandle.get_result_async()
        self.resultFuture.add_done_callback(self.resultCb)

    def feedbackCb(self, feedbackMsg) -> None:
        '''Read server feedback while recording so you know progress before it stops (auto-finish or cancel).'''
        fb = feedbackMsg.feedback
        self.get_logger().info(
            f"[feedback] status={fb.status} events={fb.num_events} world_states={fb.num_world_states}"
        )

    def cancelDoneCb(self, future) -> None:
        '''Confirm whether STOP (cancel) was accepted; either way the server will soon return a result.'''
        cancelResponse = future.result()
        if len(cancelResponse.goals_canceling) > 0:
            self.get_logger().warn("Cancel accepted; waiting for result...")
        else:
            self.get_logger().warn("Cancel rejected or nothing to cancel; still waiting for result if running.")

    def resultCb(self, future) -> None:
        '''Receive the final sequence (events + world_states) when recording stops (auto-finish or cancel).'''
        status = future.result().status
        res = future.result().result

        self.get_logger().info(f"=== RESULT status={status} session_id={res.session_id} ===")

        self.get_logger().info(f"Events: {len(res.events)}")
        for e in res.events:
            self.get_logger().info(
                f"  - {e.type} cube={e.cube_id} t={e.stamp.sec}.{e.stamp.nanosec:09d}"
            )

        self.get_logger().info(f"World states: {len(res.world_states)}")
        for i, ws in enumerate(res.world_states):
            self.get_logger().info(
                f"  - WS[{i}] t={ws.stamp.sec}.{ws.stamp.nanosec:09d} cubes={len(ws.cubes)}"
            )
            for cs in ws.cubes:
                self.get_logger().info(
                    f"      {cs.cube_id}: vert={cs.vert} level={cs.level} height={cs.height}"
                )

        #Reset local state so user can START again
        self.recordingActive = False
        self.goalHandle = None
        self.resultFuture = None
        self.get_logger().info("Recording finished; toggle START is available again.")


def runToggleLoop(node: RecordDemoClient) -> None:
    '''Blocking terminal loop: press Enter to toggle START/STOP, or type 'q' to quit.'''
    print("\nToggle control:")
    print("  Press Enter to START/STOP recording")
    print("  Type 'q' then Enter to quit\n")

    while rclpy.ok():
        s = input().strip().lower()
        if s == "q":
            rclpy.shutdown()
            break
        node.toggleRecording()


def main(args=None):
    rclpy.init(args=args)
    node = RecordDemoClient()

    #Run input loop in a separate thread so rclpy.spin can keep callbacks alive
    thread = threading.Thread(target=runToggleLoop, args=(node,), daemon=True)
    thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

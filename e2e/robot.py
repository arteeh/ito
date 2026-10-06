"""Instrumented robot for real process/network runs; no product adapter uses this."""

import json
import time
from fractions import Fraction

from aiortc import AudioStreamTrack, VideoStreamTrack
from av import VideoFrame

from ito.driver import Adapter
from ito.protocol import Camera, Depth, FrameMetadata, Intrinsics, Pose, RobotDescription


class CameraTrack(VideoStreamTrack):
    def __init__(self, robot):
        super().__init__()
        self._id = "front-video"
        self.robot = robot
        self.sequence = 0

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = VideoFrame(160, 120, "yuv420p")
        for index, plane in enumerate(frame.planes):
            plane.update(
                bytes([32 + self.sequence % 160 if index == 0 else 128]) * plane.buffer_size
            )
        frame.pts, frame.time_base = pts, time_base
        metadata = FrameMetadata(
            camera="front",
            sequence=self.sequence,
            capture_time=time.monotonic(),
            camera_pose=Pose(position=(1.0, 2.0, 3.0)),
            depth=Depth.from_bytes(160, 120, b"\xe8\x03" * (160 * 120)),
        )
        self.robot.publish_frame(metadata)
        self.sequence += 1
        return frame


class Robot(Adapter):
    def __init__(
        self,
        journal: str,
        fail_apply: bool = False,
        fail_telemetry: bool = False,
        fail_neutral_once: bool = False,
    ):
        self.journal = open(journal, "a", buffering=1)
        self.active = False
        self.applied = 0
        self.last_sequence = -1
        self.fail_apply = fail_apply
        self.fail_telemetry = fail_telemetry
        self.fail_neutral_once = fail_neutral_once
        self.audio_tasks = []

    @property
    def description(self):
        return RobotDescription(
            name="e2e instrumented robot",
            capabilities=("audio-input", "audio-output", "depth"),
            cameras=(
                Camera(
                    name="front",
                    track_id="front-video",
                    intrinsics=Intrinsics(
                        width=160, height=120, fx=120.0, fy=120.0, cx=80.0, cy=60.0
                    ),
                ),
            ),
        )

    def media_tracks(self):
        return [CameraTrack(self), AudioStreamTrack()]

    def record(self, event, **fields):
        self.journal.write(json.dumps({"event": event, "time": time.monotonic(), **fields}) + "\n")

    def apply(self, state):
        if self.fail_apply:
            self.active = True
            self.record("apply_failed")
            raise OSError("injected robot command failure")
        self.active = True
        self.applied += 1
        self.last_sequence = state.sequence
        self.record(
            "apply",
            sequence=state.sequence,
            head=state.head.model_dump() if state.head else None,
            axes=state.axes,
        )

    def neutral(self):
        if self.fail_neutral_once:
            self.fail_neutral_once = False
            self.record("neutral_failed")
            raise OSError("injected robot neutral failure")
        self.active = False
        self.record("neutral")

    def telemetry(self):
        if self.fail_telemetry:
            self.record("telemetry_failed")
            raise OSError("injected robot telemetry failure")
        return {
            "active": self.active,
            "applied": float(self.applied),
            "last_sequence": float(self.last_sequence),
        }

    def incoming_audio(self, track):
        import asyncio

        from aiortc.mediastreams import MediaStreamError

        async def consume():
            try:
                while True:
                    frame = await track.recv()
                    self.record(
                        "incoming_audio",
                        samples=frame.samples,
                        seconds=float(frame.time_base or Fraction(1, 48000)),
                    )
            except MediaStreamError:
                pass

        self.audio_tasks.append(asyncio.create_task(consume()))

    async def close(self):
        import asyncio

        for task in self.audio_tasks:
            task.cancel()
        await asyncio.gather(*self.audio_tasks, return_exceptions=True)
        self.journal.close()

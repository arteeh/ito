"""Instrumented robot for real process/network runs; no product adapter uses this."""

import json
import os
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
        width, height = self.robot.resolution
        frame = VideoFrame(width, height, "yuv420p")
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
            depth=self.robot.depth[self.sequence % len(self.robot.depth)],
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
        resolution: tuple[int, int] = (160, 120),
        noisy_depth: bool = False,
    ):
        self.journal = open(journal, "a", buffering=1)
        self.active = False
        self.applied = 0
        self.last_sequence = -1
        self.fail_apply = fail_apply
        self.fail_telemetry = fail_telemetry
        self.fail_neutral_once = fail_neutral_once
        self.audio_tasks = []
        self.resolution = width, height = tuple(resolution)
        # Noisy depth does not compress: the size a real depth camera puts on the link.
        # Prepared once, so the instrument spends its time sending, not generating.
        self.depth = [
            Depth.from_bytes(width, height, os.urandom(width * height * 2))
            for _ in range(4 if noisy_depth else 0)
        ] or [Depth.from_bytes(width, height, b"\xe8\x03" * (width * height))]

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
                        width=self.resolution[0],
                        height=self.resolution[1],
                        fx=0.75 * self.resolution[0],
                        fy=0.75 * self.resolution[0],
                        cx=self.resolution[0] / 2,
                        cy=self.resolution[1] / 2,
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

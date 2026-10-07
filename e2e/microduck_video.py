"""Capture the real sensor TCP stream and received pilot video during microduck.py."""

import json
import socket
import struct
import threading
import time

import cv2
import numpy as np
import pygame
from PIL import Image


class VideoEvidence:
    def __init__(self, sim, out):
        self.out = out
        command = next(child.args for name, child in sim.children if name == "body")
        self.port = int(command[command.index("--frame-port") + 1])
        self.stop = threading.Event()
        self.latest = None
        self.error = None
        self.rows = []
        self.sensor_times = []
        self.writers = {}
        self.last_saved = 0
        self.last_video = None
        self.thread = threading.Thread(target=self.sensor, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def write(self, name, rgb):
        if name not in self.writers:
            height, width = rgb.shape[:2]
            writer = cv2.VideoWriter(
                str(self.out / f"{name}.mkv"),
                cv2.VideoWriter_fourcc(*"FFV1"),
                15,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not open lossless evidence video {name}")
            self.writers[name] = writer
        self.writers[name].write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def sensor(self):
        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=3) as sock:
                with sock.makefile("rb") as stream:
                    while not self.stop.is_set():
                        header = stream.read(4)
                        if len(header) != 4:
                            raise RuntimeError("sensor closed before evidence capture ended")
                        length = struct.unpack("<I", header)[0]
                        if length != 640 * 360 * 2:
                            raise RuntimeError(f"unexpected sensor frame length: {length}")
                        packed = stream.read(length)
                        frame = np.frombuffer(packed, np.uint8).reshape(360, 640, 2)
                        rgb = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_UYVY)
                        rgb = np.ascontiguousarray(np.rot90(rgb, -1))
                        self.write("sensor", rgb)
                        captured = time.monotonic()
                        self.sensor_times.append(captured)
                        self.latest = (captured, rgb, packed)
        except Exception as exc:
            if not self.stop.is_set():
                self.error = exc

    def capture(self, app, stage):
        if self.error:
            raise self.error
        rgb = app.state.video
        if rgb is None or self.latest is None or app.state.video_time == self.last_video:
            return
        self.last_video = app.state.video_time
        self.write("received", rgb)
        now = time.monotonic()
        captured, raw, packed = self.latest
        row = dict(
            time=now,
            stage=stage,
            sensor_received=captured,
            video_time=app.state.video_time,
            flat_video=app.state.flat_video,
        )
        if now - self.last_saved > 1:
            self.last_saved = now
            name = f"boundary-{len(self.rows):04d}-stage-{stage}"
            Image.fromarray(raw).save(self.out / f"{name}-sensor.png")
            Image.fromarray(rgb).save(self.out / f"{name}-received.png")
            (self.out / f"{name}-sensor.uyvy").write_bytes(packed)
            row["capture"] = name
            for event in (pygame.KEYDOWN, pygame.KEYUP):
                pygame.event.post(pygame.event.Event(event, key=pygame.K_F12))
        self.rows.append(row)

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=4)
        for writer in self.writers.values():
            writer.release()
        (self.out / "video-boundaries.json").write_text(json.dumps(self.rows, indent=2) + "\n")
        (self.out / "sensor-times.json").write_text(json.dumps(self.sensor_times) + "\n")
        if exc[0] is None:
            if self.error:
                raise self.error
            assert not self.thread.is_alive(), "sensor capture did not stop"
            assert self.rows, "no paired video boundary samples"

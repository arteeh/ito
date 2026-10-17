"""ITO_RECORD=<dir> records the window as the pilot sees it (screen/), each robot camera
frame Ito receives (camera/), and each frame reconstruction mapped beside the map drawn from
the pose it mapped that frame by (map/): JPEGs plus frames.jsonl ("t" in Ito clock seconds),
which tools/session.py turns into videos. ITO_RECORD_FPS caps the screen rate (default 30)."""

import json
import logging
import os
import queue
import threading
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

from ito import clock

log = logging.getLogger(__name__)

WIDTH = 1280


def from_env() -> "Recorder | None":
    folder = os.environ.get("ITO_RECORD")
    if not folder:
        return None
    return Recorder(Path(folder), float(os.environ.get("ITO_RECORD_FPS", "30")))


class _Stream:
    def __init__(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        self.folder = folder
        self.index = (folder / "frames.jsonl").open("w", encoding="utf-8")
        self.count = 0

    def write(self, rgb: np.ndarray, t: float, extra=None) -> None:
        image = Image.fromarray(np.ascontiguousarray(rgb[..., :3]))
        if image.width > WIDTH:
            image = image.resize((WIDTH, round(image.height * WIDTH / image.width)), Image.BILINEAR)
        image = image.crop((0, 0, image.width & ~1, image.height & ~1))
        name = f"{self.count:06d}.jpg"
        image.save(self.folder / name, quality=90)
        self.index.write(json.dumps({"file": name, "t": t} | (extra or {})) + "\n")
        self.count += 1

    def close(self) -> None:
        self.index.close()


class Recorder:
    def __init__(self, folder: Path, fps: float = 30):
        self.folder = folder
        self.period = 1 / max(1.0, fps)
        self.next_screen = 0.0
        self.last_video = None
        self.dropped = 0
        self.streams = {name: _Stream(folder / name) for name in ("screen", "camera", "map")}
        self.videos = deque(maxlen=90)
        self.last_map = None
        self.queue: queue.Queue = queue.Queue(maxsize=30)
        self.thread = threading.Thread(target=self._writer, name="ito-record", daemon=True)
        self.thread.start()
        log.info("Recording to %s", folder)

    def _put(self, item) -> None:
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def _writer(self) -> None:
        while (item := self.queue.get()) is not None:
            name, rgb, t, *extra = item
            try:
                self.streams[name].write(rgb, t, *extra)
            except Exception:
                log.exception("Recording %s frame failed", name)

    def due(self, now: float) -> bool:
        return now >= self.next_screen

    def screen(self, now: float, pixels: bytes, size, eye=None) -> None:
        """pixels: bottom-up RGB from glReadPixels, size (width, height); eye: the drawn
        world-from-eye, kept in frames.jsonl to tell a moving world from a changing map."""
        self.next_screen = max(self.next_screen + self.period, now)
        rgb = np.frombuffer(pixels, np.uint8).reshape(size[1], size[0], 3)[::-1]
        extra = None if eye is None else {"eye": np.round(eye, 5).ravel().tolist()}
        self._put(("screen", rgb, now, extra))

    def camera(self, video, video_time) -> None:
        if video is None or video_time == self.last_video:
            return
        self.last_video = video_time
        rgb = np.asarray(video)
        self.videos.append((video_time, rgb))
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * (255 if rgb.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
        self._put(("camera", rgb.copy(), clock.now()))

    def wants_map(self, mapped) -> bool:
        return mapped[1] != self.last_map

    def video_at(self, captured):
        self.last_map = captured
        return next((rgb for t, rgb in self.videos if t == captured), None)

    def map(self, camera, pixels: bytes, size, captured) -> None:
        """The mapped frame (left) and the map drawn from its pose (right, bottom-up RGB)."""
        drawn = np.frombuffer(pixels, np.uint8).reshape(size[1], size[0], 3)[::-1]
        frame = np.asarray(Image.fromarray(np.ascontiguousarray(camera[..., :3])).resize(size))
        self._put(("map", np.concatenate((frame, drawn), axis=1), captured))

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join(timeout=30)
        for stream in self.streams.values():
            stream.close()
        summary = {name: stream.count for name, stream in self.streams.items()}
        summary["dropped"] = self.dropped
        (self.folder / "summary.json").write_text(json.dumps(summary) + "\n")
        log.info("Recording closed: %s", summary)

"""Opt-in motion recording of what the pilot sees, for motion sheets (e2e/motion_sheet.py).

Set ITO_RECORD=<dir> to record. Two streams are written, each a folder of JPEGs plus an
index (frames.jsonl, one JSON object per frame, "t" in Ito clock seconds):

- pilot/: the drawn world view (before the overlay), at most ITO_RECORD_FPS (default 30)
- camera/: each new robot camera frame Ito received, as Ito holds it

Reading pixels back costs a few milliseconds per recorded frame, so leave this off for
timing measurements. JPEG encoding runs on a background thread.
"""

import json
import logging
import math
import os
import queue
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from ito import clock

log = logging.getLogger(__name__)

WIDTH = 640  # Recorded frames are scaled to this width; sheets show them smaller still.


def from_env() -> "Recorder | None":
    folder = os.environ.get("ITO_RECORD")
    if not folder:
        return None
    return Recorder(Path(folder), float(os.environ.get("ITO_RECORD_FPS", "30")))


def heading(matrix) -> float | None:
    """Yaw of a 4x4 camera pose in degrees (camera looks down -Z), or None."""
    try:
        forward = -np.asarray(matrix, dtype=float)[:3, 2]
    except (TypeError, ValueError, IndexError):
        return None
    return round(math.degrees(math.atan2(forward[0], -forward[2])), 2)


class _Stream:
    def __init__(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        self.folder = folder
        self.index = (folder / "frames.jsonl").open("w", encoding="utf-8")
        self.count = 0

    def write(self, rgb: np.ndarray, row: dict) -> None:
        image = Image.fromarray(np.ascontiguousarray(rgb[..., :3]))
        if image.width > WIDTH:
            image = image.resize((WIDTH, round(image.height * WIDTH / image.width)), Image.BILINEAR)
        name = f"{self.count:06d}.jpg"
        image.save(self.folder / name, quality=88)
        self.index.write(json.dumps({"file": name, **row}) + "\n")
        self.count += 1

    def close(self) -> None:
        self.index.close()


class Recorder:
    def __init__(self, folder: Path, fps: float = 30):
        self.folder = folder
        self.period = 1 / max(1.0, fps)
        self.next_pilot = 0.0
        self.last_video = None
        self.dropped = 0
        self.streams = {name: _Stream(folder / name) for name in ("pilot", "camera")}
        self.queue: queue.Queue = queue.Queue(maxsize=90)
        self.thread = threading.Thread(target=self._writer, name="ito-record", daemon=True)
        self.thread.start()
        (folder / "recording.json").write_text(
            json.dumps({"started": clock.now(), "fps": fps, "streams": list(self.streams)}) + "\n"
        )
        log.info("Recording the pilot view to %s", folder)

    def _put(self, item) -> None:
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def _writer(self) -> None:
        while (item := self.queue.get()) is not None:
            name, rgb, row = item
            try:
                self.streams[name].write(rgb, row)
            except Exception:
                log.exception("Recording %s frame failed", name)

    def due(self, now: float) -> bool:
        return now >= self.next_pilot

    def pilot(self, now: float, pixels: bytes, size, **row) -> None:
        """pixels: bottom-up RGB from glReadPixels, size (width, height)."""
        self.next_pilot = max(self.next_pilot + self.period, now)
        rgb = np.frombuffer(pixels, np.uint8).reshape(size[1], size[0], 3)[::-1]
        self._put(("pilot", rgb, {"t": now, **row}))

    def camera(self, video, video_time, now: float, robot_camera=None) -> None:
        if video is None or video_time == self.last_video:
            return
        self.last_video = video_time
        rgb = np.asarray(video)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * (255 if rgb.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
        self._put(
            ("camera", rgb.copy(), {"t": now, "captured": video_time, "yaw": heading(robot_camera)})
        )

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join(timeout=30)
        for stream in self.streams.values():
            stream.close()
        summary = {name: stream.count for name, stream in self.streams.items()}
        summary["dropped_by_recorder"] = self.dropped
        (self.folder / "summary.json").write_text(json.dumps(summary) + "\n")
        log.info("Recording closed: %s", summary)

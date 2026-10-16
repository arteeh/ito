"""ITO_RECORD=<dir> records the window as the pilot sees it (screen/) and each robot camera
frame Ito receives (camera/): JPEGs plus frames.jsonl ("t" in Ito clock seconds), which
tools/session.py turns into videos. ITO_RECORD_FPS caps the screen rate (default 30)."""

import json
import logging
import os
import queue
import threading
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

    def write(self, rgb: np.ndarray, t: float) -> None:
        image = Image.fromarray(np.ascontiguousarray(rgb[..., :3]))
        if image.width > WIDTH:
            image = image.resize((WIDTH, round(image.height * WIDTH / image.width)), Image.BILINEAR)
        # Video encoders need even sizes.
        image = image.crop((0, 0, image.width & ~1, image.height & ~1))
        name = f"{self.count:06d}.jpg"
        image.save(self.folder / name, quality=90)
        self.index.write(json.dumps({"file": name, "t": t}) + "\n")
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
        self.streams = {name: _Stream(folder / name) for name in ("screen", "camera")}
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
            name, rgb, t = item
            try:
                self.streams[name].write(rgb, t)
            except Exception:
                log.exception("Recording %s frame failed", name)

    def due(self, now: float) -> bool:
        return now >= self.next_screen

    def screen(self, now: float, pixels: bytes, size) -> None:
        """pixels: bottom-up RGB from glReadPixels, size (width, height)."""
        self.next_screen = max(self.next_screen + self.period, now)
        rgb = np.frombuffer(pixels, np.uint8).reshape(size[1], size[0], 3)[::-1]
        self._put(("screen", rgb, now))

    def camera(self, video, video_time) -> None:
        if video is None or video_time == self.last_video:
            return
        self.last_video = video_time
        rgb = np.asarray(video)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * (255 if rgb.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
        self._put(("camera", rgb.copy(), clock.now()))

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join(timeout=30)
        for stream in self.streams.values():
            stream.close()
        summary = {name: stream.count for name, stream in self.streams.items()}
        summary["dropped"] = self.dropped
        (self.folder / "summary.json").write_text(json.dumps(summary) + "\n")
        log.info("Recording closed: %s", summary)

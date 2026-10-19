"""ITO_MICRODUCK_RECORD=<dir>: the driver saves everything reconstruction is given, for replaying
it offline (a research dataset, not a product feature):

    meta.json       camera geometry (raw matrix, distortion, the rectified pinhole Ito is told),
                    motion noise, frame delay, exposure, ToF field of view
    frames.jsonl    one line per received frame: seq, received (driver clock, the stamp Ito maps
                    it by), rtp (seconds), and with the sim's truth: its exposure time, exact
                    camera pose in Ito's world (position, xyzw), the MuJoCo row, depth file
    frames/N.png    the frame as received, before rectification (lossless)
    depth/N.png     the sim's exact axial depth, uint16 millimetres, on its own (coarser) grid
    motion.jsonl    every MotionSample (driver clock), as published
    sensors.jsonl   every Sensors message (head IMU, ToF grids), as published

All times are the driver's CLOCK_MONOTONIC seconds, which the sim's truth also uses."""

import base64
import json
import logging
import os
import queue
import threading
import zlib
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def from_env():
    folder = os.environ.get("ITO_MICRODUCK_RECORD")
    return Recorder(Path(folder)) if folder else None


class Recorder:
    def __init__(self, folder: Path, writers=3):
        import cv2

        self.cv = cv2
        self.folder = folder
        (folder / "frames").mkdir(parents=True, exist_ok=True)
        (folder / "depth").mkdir(exist_ok=True)
        self.frames = (folder / "frames.jsonl").open("a", buffering=1)
        self.motion_log = (folder / "motion.jsonl").open("a", buffering=1)
        self.sensor_log = (folder / "sensors.jsonl").open("a", buffering=1)
        self.lock = threading.Lock()
        self.seq = 0
        self.dropped = 0
        self.queue = queue.Queue(maxsize=240)
        self.threads = [
            threading.Thread(target=self._writer, name=f"record-{i}", daemon=True)
            for i in range(writers)
        ]
        for thread in self.threads:
            thread.start()
        log.info("Recording the driver's inputs to %s", folder)

    def meta(self, **values):
        (self.folder / "meta.json").write_text(json.dumps(values, indent=1, default=_plain))

    def _writer(self):
        while True:
            item = self.queue.get()
            if item is None:
                return
            path, image = item
            try:
                self.cv.imwrite(str(path), image, [self.cv.IMWRITE_PNG_COMPRESSION, 1])
            except Exception:
                log.exception("Recording %s failed", path)

    def _put(self, path, image):
        try:
            self.queue.put_nowait((path, image))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def frame(self, raw, received, rtp, truth=None):
        """raw: the RGB frame as received; truth: (row, position, xyzw) or None."""
        with self.lock:
            seq = self.seq
            self.seq += 1
        name = f"{seq:06d}.png"
        saved = self._put(self.folder / "frames" / name, raw[..., ::-1].copy())
        line = dict(seq=seq, received=received, rtp=rtp, file=name if saved else None)
        if truth is not None:
            row, position, orientation = truth
            line |= dict(
                exposed=row["time"],
                position=list(position),
                orientation=list(orientation),
                sim=dict(position=row["position"], orientation=row["orientation"]),
            )
            if "depth" in row:
                height, width = row["depth_shape"]
                mm = np.frombuffer(zlib.decompress(base64.b64decode(row["depth"])), "<u2")
                depth = mm.reshape(height, width)
                if self._put(self.folder / "depth" / name, depth.copy()):
                    line["depth"] = name
        self.frames.write(json.dumps(line) + "\n")

    def motion(self, sample):
        self.motion_log.write(sample.model_dump_json() + "\n")

    def sensors(self, message):
        self.sensor_log.write(message.model_dump_json() + "\n")

    def close(self):
        for _ in self.threads:
            self.queue.put(None)
        for thread in self.threads:
            thread.join(timeout=30)
        for stream in (self.frames, self.motion_log, self.sensor_log):
            stream.close()
        (self.folder / "summary.json").write_text(
            json.dumps(dict(frames=self.seq, dropped_images=self.dropped)) + "\n"
        )


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return str(value)

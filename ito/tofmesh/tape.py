"""ITO_TOFMESH_TAPE=<file>: the ToF mesh worker appends every input it gets (motion, head IMU, ToF
grids, camera frames as JPEG) with its pilot-clock arrival time to <file>, a stream of pickles.
tools/tofmesh_tape.py feeds such a file through Fusion again, so two versions of the pipeline (or
one with a feature on and off) can be compared on exactly the inputs a live session had.
Unset (the default), nothing is recorded.

For tests only: encoding every camera frame takes the worker's time, so a recorded session's mesh
lags more (Ceres, live sim: ToF to pilot median 105 -> ~140 ms). Compare lag between sessions
recorded alike, or not at all."""

import io
import os
import pickle

from ito import clock

PATH = os.environ.get("ITO_TOFMESH_TAPE")
_file = None


def record(item):
    global _file
    if _file is None:
        _file = open(PATH, "ab")  # noqa: SIM115 (open for the worker's lifetime)
    if item[0] == "frame":
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(item[1]).save(buffer, "JPEG", quality=93)
        item = ("frame_jpeg", buffer.getvalue(), *item[2:])
    pickle.dump((clock.now(), item), _file)
    _file.flush()


def load(path):
    """[(arrival, item), ...] of a recording; frames come back as ("frame_jpeg", bytes, ...)."""
    items = []
    with open(path, "rb") as f:
        while True:
            try:
                items.append(pickle.load(f))
            except EOFError:
                return items

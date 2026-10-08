"""Exercise real reconstruction process failures: uv run python e2e/reconstruction_faults.py."""

import os
import time

import numpy as np

from ito import clock
from ito.protocol import Intrinsics
from ito.reconstruction import Reconstruction
from ito.render import pose

CAMERA = Intrinsics(width=32, height=24, fx=24, fy=24, cx=15.5, cy=11.5)
RGB = np.full((24, 32, 3), 180, np.uint8)
DEPTH = np.ones((24, 32), np.float32)


def receive(source, *, timeout=5):
    deadline = clock.now() + timeout
    while clock.now() < deadline:
        packet = source.poll()
        if packet is not None:
            if packet.acknowledge is not None:
                packet.acknowledge()
            return packet
        time.sleep(0.01)
    raise AssertionError("Worker stopped producing updates")


def main():
    with Reconstruction(CAMERA, max_splats=64, device="cpu") as source:
        # Simulate concurrent ownership of the latest-frame mailbox.
        source.input_lock.acquire()
        try:
            began = clock.now()
            assert not source.submit(RGB, DEPTH, pose())
            assert clock.now() - began < 0.05, "Input waited on reconstruction"
        finally:
            source.input_lock.release()
        source.submit(RGB, DEPTH, pose())
        first = receive(source)
        assert first.count == 64
        source.submit(RGB, DEPTH, pose())
        second = receive(source)
        assert second.count == first.count
        assert np.array_equal(first.indices, second.indices)
        assert np.array_equal(first.records[:, 0, :3], second.records[:, 0, :3]), (
            "Re-observing an identical frame duplicated the region"
        )
        source.set_max_splats(128)
        source.submit(RGB, DEPTH, pose())
        grown = receive(source)
        assert grown.capacity == 128 and grown.count == 128
        assert np.isfinite(grown.records).all()
        invalid = DEPTH.copy()
        invalid[:, ::4] = np.nan
        invalid[:, 1::4] = np.inf
        invalid[:, 2::4] = -1
        invalid[:, 3::4] = 0
        source.submit(RGB, invalid, pose())
        # The worker must remain healthy after an entirely missing depth observation.
        time.sleep(0.15)
        source.submit(RGB, DEPTH, pose())
        assert np.isfinite(receive(source).records).all()
        source.process.terminate()
        source.process.join(timeout=3)
        try:
            source.poll()
        except RuntimeError as exc:
            assert "exited" in str(exc), exc
        else:
            raise AssertionError("Worker death was not reported")

    # An explicitly requested unavailable CUDA backend must have an actionable error.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    with Reconstruction(CAMERA, max_splats=64, device="cuda") as source:
        try:
            receive(source)
        except RuntimeError as exc:
            assert "CUDA" in str(exc) and "device" in str(exc), exc
        else:
            raise AssertionError("CUDA unexpectedly started with all devices hidden")
    print("PASS: busy mailbox, repeated frames, growth, missing depth, worker death, CUDA failure")


if __name__ == "__main__":
    main()

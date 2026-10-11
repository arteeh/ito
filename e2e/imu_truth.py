"""Run #32's worker/renderer benchmark with measured camera priors for #33 A/B.

Requires scene_truth.py from fix/scene-truth. Pass its CLI, plus optional --prior-noise.
ITO_SLAM_IMU=0/1 python e2e/imu_truth.py --backend slam --label imu-off/on
"""

import sys

import numpy as np
import scene_truth

from ito.reconstruction.prior import turn


class PriorRoom(scene_truth.Room):
    latest = None

    def view(self, *args, **kwargs):
        rgb, depth, camera = super().view(*args, **kwargs)
        PriorRoom.latest = camera.copy()
        return rgb, depth, camera


class PriorWorker(scene_truth.Reconstruction):
    noise = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng(33)
        self.prior_stamp = None
        self.prior_camera = None

    def submit(self, rgb, depth, camera, **kwargs):
        # Backpressure retries the same exposure; never draw different sensor noise for it.
        if kwargs["captured_at"] != self.prior_stamp:
            self.prior_stamp = kwargs["captured_at"]
            camera = PriorRoom.latest.copy()
            if self.noise:
                camera[:3, :3] = turn(np.radians(self.rng.normal(0, 1))) @ camera[:3, :3]
                camera[:3, 3] += self.rng.normal(0, 0.005, 3)
            self.prior_camera = camera
        camera = self.prior_camera
        kwargs["measured"] = True
        return super().submit(rgb, depth, camera, **kwargs)


if __name__ == "__main__":
    if "--prior-noise" in sys.argv:
        sys.argv.remove("--prior-noise")
        PriorWorker.noise = True
    scene_truth.Room = PriorRoom
    scene_truth.Reconstruction = PriorWorker
    scene_truth.main()

"""Manual session entry: inject a gauge correction or allocation failure for visual inspection.

Used by session.py --correction-after. The released application and simulator are unchanged.
The correction changes the estimator's coordinate gauge, not the robot's measured motion.
"""

import json
import os
import runpy
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ito.reconstruction.slam import SLAMBackend

integrate = SLAMBackend.integrate


def corrected(self, *args, captured, **kwargs):
    if not hasattr(self, "manual_correction_start"):
        self.manual_correction_start = captured
        self.manual_correction_done = False
        self.manual_oom_done = False
    oom_after = float(os.environ.get("ITO_MANUAL_OOM_AFTER", "inf"))
    if not self.manual_oom_done and captured - self.manual_correction_start >= oom_after:
        import torch

        self.manual_oom_done = True
        with Path(os.environ["ITO_TIMELINE"]).with_suffix(".events.jsonl").open("a") as out:
            out.write(
                json.dumps(
                    dict(event="manual_cuda_oom", captured=captured, precision=self.precision)
                )
                + "\n"
            )
        raise torch.cuda.OutOfMemoryError("Manual session allocation failure")
    after = float(os.environ.get("ITO_MANUAL_CORRECTION_AFTER", "inf"))
    if not self.manual_correction_done and captured - self.manual_correction_start >= after:
        graph = self.graph
        change = np.eye(4)
        change[:3, :3] = Rotation.from_euler("y", 35, degrees=True).as_matrix()
        change[:3, 3] = [1.2, 0, -0.4]
        for node in graph.nodes.values():
            node.pose = change @ node.pose
        if graph.anchor is not None:
            graph.anchor = change @ graph.anchor
        graph.published = deque(
            ((t, node, change @ pose) for t, node, pose in graph.published),
            maxlen=graph.published.maxlen,
        )
        graph.mapped = {
            node: (change @ pose, sigma) for node, (pose, sigma) in graph.mapped.items()
        }
        poses = {node: value.pose for node, value in graph.nodes.items()}
        self.frontend.correct(poses, 1.0)
        self.revise(poses)
        self.manual_correction_done = True
        event = dict(
            event="manual_estimator_correction",
            captured=captured,
            transform=change.tolist(),
            local_refits=self.refits,
        )
        with Path(os.environ["ITO_TIMELINE"]).with_suffix(".events.jsonl").open("a") as out:
            out.write(json.dumps(event) + "\n")
        print(f"Manual estimator correction at capture {captured:.6f}", flush=True)
    return integrate(self, *args, captured=captured, **kwargs)


SLAMBackend.integrate = corrected

if __name__ == "__main__":
    runpy.run_module("ito.app", run_name="__main__")

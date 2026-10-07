"""MASt3R-SLAM's tracker and local graph, with a bounded live keyframe window."""

import time

import numpy as np

from .rgbd import RGBDBackend


class Keyframes(list):
    def last_keyframe(self):
        return self[-1]

    def update_T_WCs(self, poses, indices):
        for transform, index in zip(poses, indices, strict=True):
            self[int(index)].T_WC = transform


class SLAMBackend(RGBDBackend):
    def __init__(self, capacity, intrinsics, *, report, origin, **options):
        from .mast3r_runtime import load_model, prepare

        source, weights = prepare(report)
        import lietorch
        import torch
        from mast3r_slam.config import config, load_config
        from mast3r_slam.tracker import FrameTracker

        load_config(str(source / "config/base.yaml"))
        config["single_thread"] = True
        config["tracking"]["filtering_mode"] = "recent"
        config["local_opt"]["max_iters"] = 5
        self.model = load_model(weights, report)
        options["device"] = "cuda"
        super().__init__(capacity, intrinsics, **options)
        self.report = report
        self.frames = Keyframes()
        self.tracker = FrameTracker(self.model, self.frames, "cuda")
        self.transform = lietorch.Sim3.Identity(1, device="cuda")
        self.camera_pose = np.eye(4, dtype=np.float32)
        self.tracked = 0
        self.frame_id = 0
        self.lost = False
        self.origin = torch.as_tensor(origin, device="cuda", dtype=torch.float32)
        self.axes = torch.tensor([1, -1, -1], device="cuda")
        report("MASt3R-SLAM ready; waiting for camera")

    def integrate(self, rgb, depth, camera, now):
        import torch
        from mast3r_slam.frame import create_frame
        from mast3r_slam.global_opt import FactorGraph
        from mast3r_slam.mast3r_utils import mast3r_inference_mono

        started = time.monotonic()
        with torch.inference_mode():
            frame = create_frame(self.frame_id, rgb.astype(np.float32) / 255, self.transform)
            self.frame_id += 1
            if not self.frames:
                points, confidence = mast3r_inference_mono(self.model, frame)
                frame.update_pointmap(points, confidence)
                self.frames.append(frame)
            else:
                # Try one retained keyframe per observation during loss. Never reset
                # the world origin or fuse a failed pose into the live map.
                if self.lost:
                    self.tracker.reset_idx_f2k()
                    self.frames.insert(0, self.frames.pop())
                    frame.T_WC = self.frames[-1].T_WC
                new_keyframe, _, lost = self.tracker.track(frame)
                self.lost = bool(lost)
                if self.lost:
                    self.report("SLAM tracking lost; move back toward the last view")
                    return
                if new_keyframe:
                    self.frames.append(frame)
                    # Bounded graph includes adjacent edges and a local loop to the
                    # oldest retained view. Upstream rejects non-overlapping edges.
                    del self.frames[:-4]
                    graph = FactorGraph(self.model, self.frames, device="cuda")
                    last = len(self.frames) - 1
                    graph.add_factors(list(range(last)), list(range(1, last + 1)), 0.1)
                    if last > 1:
                        graph.add_factors([0], [last], 0.1)
                    graph.solve_GN_rays()
                    self.tracker.reset_idx_f2k()
            self.transform = frame.T_WC
            matrix = self.transform.matrix()[0]
            scale = torch.linalg.det(matrix[:3, :3]).abs().pow(1 / 3)
            if not torch.isfinite(matrix).all() or not 0.01 < float(scale) < 100:
                raise RuntimeError("MASt3R-SLAM produced an invalid camera pose")
            # OpenCV camera/world axes (+Y down, +Z forward) to Ito; remove Sim3
            # scale from the renderer pose, but retain it on the dense world points.
            rigid = matrix.clone()
            rigid[:3, :3] /= scale
            rigid[:3, :3] *= self.axes[:, None] * self.axes[None, :]
            rigid[:3, 3] *= self.axes
            rigid = self.origin @ rigid
            self.camera_pose = rigid.cpu().numpy().astype(np.float32)
            points = self.transform.act(frame.X_canon) * self.axes
            points = points @ self.origin[:3, :3].T + self.origin[:3, 3]
            valid = (
                (frame.get_average_conf().flatten() > 1.5)
                & torch.isfinite(points).all(dim=1)
                & (frame.X_canon[:, 2] > 0)
            )
            colors = (frame.uimg.to("cuda").reshape(-1, 3) * 255).to(torch.uint8)
            # DLPack shares CUDA memory; only the fused slot records cross to the ring.
            self.integrate_points(
                self.xp.from_dlpack(points[valid].contiguous()),
                self.xp.from_dlpack(colors[valid].contiguous()),
                now + time.monotonic() - started,
            )
            self.tracked += 1
            self.report(f"MASt3R-SLAM tracking | {self.tracked} frames")

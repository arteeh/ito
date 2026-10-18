"""Depth that agrees across views: Depth Anything 3 (DA3-BASE, Apache-2.0) over the new frame
and the retained views around it.

One network call sees all of them together, so their depths agree with each other up to one
scale per call. That scale is taken from the retained views' own metric depth: the new frame's
surfaces land where the map already holds them, instead of where a pairwise guess put them.
"""

import os
import sys
from pathlib import Path

import numpy as np

from .mast3r_runtime import MODELS

ROOT = MODELS / "depth"  # depth_anything_3 source and DA3-BASE weights, from ito.build
# ITO_DA3=giant (an experiment): DA3-GIANT-1.1 (CC BY-NC 4.0), weights from ITO_DA3_WEIGHTS.
DA3_SIZE = os.environ.get("ITO_DA3", "base")
SIZE = (504, 280)  # width, height the network sees: multiples of its 14-pixel patches
MEAN = np.array((0.485, 0.456, 0.406), np.float32)
STD = np.array((0.229, 0.224, 0.225), np.float32)


class MultiViewDepth:
    def __init__(self):
        import torch
        from safetensors.torch import load_file

        if not (ROOT / "model.safetensors").is_file():
            raise RuntimeError(
                f"Depth model missing from {ROOT}: rebuild the models with python -m ito.build"
            )
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from depth_anything_3.cfg import create_object, load_config
        from depth_anything_3.registry import MODEL_REGISTRY

        weights_folder = Path(os.environ.get("ITO_DA3_WEIGHTS") or ROOT)
        net = create_object(load_config(MODEL_REGISTRY[f"da3-{DA3_SIZE}"]))
        weights = load_file(weights_folder / "model.safetensors")
        loaded = net.load_state_dict(
            {k.removeprefix("model."): v for k, v in weights.items()}, strict=False
        )
        # The release weights leave out an auxiliary head that inference never runs.
        if loaded.unexpected_keys or any("_aux" not in k for k in loaded.missing_keys):
            raise RuntimeError(f"Depth model weights do not fit DA3-BASE: {loaded}")
        if DA3_SIZE != "base":
            # The ViT in bfloat16 halves its weights (giant: 4.5 to 2.3 GB); the heads stay in
            # float32 and run outside autocast, so the ViT hands them float32.
            net.backbone.to(torch.bfloat16)
            net.backbone.register_forward_hook(lambda module, args, output: _float(output))
        self.net = net.to("cuda").eval()
        self.torch = torch
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def __call__(self, images):
        """Depths (N, SIZE[1], SIZE[0]) of uint8 RGB images seen together, one unknown scale."""
        import cv2

        torch = self.torch
        batch = np.stack(
            [
                (cv2.resize(image, SIZE, interpolation=cv2.INTER_AREA) / np.float32(255) - MEAN)
                / STD
                for image in images
            ]
        )
        x = torch.as_tensor(batch, device="cuda").permute(0, 3, 1, 2)[None]
        with torch.inference_mode(), torch.autocast("cuda", dtype=self.dtype):
            output = self.net(x, None, None, [], False, False)
        return output.depth[0].float().cpu().numpy()


MOGE = MODELS / "moge"  # MoGe-2 source, utils3d-moge and ViT-B weights, from ito.build
MOGE_TOKENS = 1800  # the ViT's patch tokens: about 1024 x 576 for a 16:9 frame


class MonoGeometry:
    """One frame's depth from MoGe-2 (ViT-B, MIT): a single-image geometry network whose
    surfaces stay put across low-texture walls and stop at silhouettes, where a pairwise
    matcher smears. Its metric scale is a guess; callers scale it to theirs."""

    def __init__(self):
        import torch

        if not (MOGE / "model.pt").is_file():
            raise RuntimeError(
                f"MoGe-2 missing from {MOGE}: rebuild the models with python -m ito.build"
            )
        if str(MOGE) not in sys.path:
            sys.path.insert(0, str(MOGE))
        from moge.model.v2 import MoGeModel

        self.torch = torch
        self.net = MoGeModel.from_pretrained(str(MOGE / "model.pt")).to("cuda").eval()

    def __call__(self, rgb, fov_x, size):
        """Depth (size[1] x size[0], metres up to an unknown scale, 0: none) of an RGB frame
        whose horizontal field of view is fov_x degrees."""
        import cv2

        torch = self.torch
        image = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        x = torch.as_tensor(image, device="cuda").permute(2, 0, 1).float() / 255
        with torch.inference_mode():
            out = self.net.infer(x, fov_x=fov_x, num_tokens=MOGE_TOKENS, use_fp16=True)
        depth = out["depth"].float()
        return torch.nan_to_num(depth, nan=0, posinf=0, neginf=0).cpu().numpy()


def _float(output):
    import torch

    if isinstance(output, torch.Tensor) and output.is_floating_point():
        return output.float()
    if isinstance(output, list | tuple):
        return type(output)(_float(item) for item in output)
    return output


def nearest_views(views, pose, count):
    """Retained views facing most like pose, and nearest it."""
    scored = []
    for view in views:
        cosine = float(view.pose[:3, 2] @ pose[:3, 2])
        distance = float(np.linalg.norm(view.pose[:3, 3] - pose[:3, 3]))
        if cosine > 0.5:
            scored.append((1 - cosine + 0.2 * distance, view))
    scored.sort(key=lambda item: item[0])
    return [view for _, view in scored[:count]]


def resample(depth, shape):
    """Nearest-pixel resample of a depth map covering the whole frame."""
    rows, columns = depth.shape
    y = ((np.arange(shape[0]) + 0.5) * rows / shape[0]).astype(int)
    x = ((np.arange(shape[1]) + 0.5) * columns / shape[1]).astype(int)
    return depth[y][:, x]


def scale_to(known, estimated):
    """Median known/estimated over pixels both have, or None if they share too few."""
    known = resample(known, estimated.shape)
    both = (known > 0) & (estimated > 0)
    if both.sum() < 0.2 * both.size:
        return None
    return float(np.median(known[both] / estimated[both]))


class MapFrontend:
    """A frontend without a visual tracker (ITO_TRACKER=map): each frame's pose comes from the
    robot's motion and from laying its depth onto the map (align.py), not from MASt3R."""

    KEYFRAME_EVERY = 3  # frames: about a second at the worker's rate

    def __init__(self, intrinsics, input_resolution):
        import torch  # noqa: F401  CuPy compiles with torch's NVRTC: load it first, as MASt3R does

        from .slam_image import SLAMImage

        self.image = SLAMImage(intrinsics, input_resolution)
        k = self.image.calibration
        self.intrinsics = (float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]))
        self.frames = 0

    def track(self, rgb, node, predicted, prior_sigma):
        import cv2
        import torch

        from .tracking import Cloud, VisualResult

        width, height = self.image.width, self.image.height
        image = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        cloud = Cloud(
            torch.zeros((width * height, 3), device="cuda"),
            torch.as_tensor(image.reshape(-1, 3), device="cuda"),
            torch.ones(width * height, dtype=torch.bool, device="cuda"),
            (height, width),
            self.intrinsics,
        )
        keyframe = self.frames % self.KEYFRAME_EVERY == 0
        self.frames += 1
        return VisualResult(None, None, np.full(6, np.inf), 0.0, cloud, keyframe)

    def correct(self, keyframe_poses, scale):
        pass

    def forget(self, nodes):
        pass


def log_field(depth, reference, grid=(2, 2), prior=None, prior_weight=0.0, tolerance=0.03):
    """A smooth multiplier taking depth onto reference: the exp of a log-scale field, bilinear
    between grid nodes over the image, fitted where both have depth. Pixels off by more than
    tolerance (log) count less, so silhouettes and wrong pixels do not bend it. prior (node
    values) holds nodes the reference barely covers. Returns (field like depth, node values),
    or None if they share too few pixels."""
    reference = resample(reference, depth.shape)
    step = max(1, depth.shape[0] // 72)
    d, r = depth[::step, ::step], reference[::step, ::step]
    both = (d > 0) & (r > 0) & np.isfinite(d) & np.isfinite(r)
    if both.sum() < 200:
        return None
    gy, gx = grid
    wy = _hat(d.shape[0], gy)
    wx = _hat(d.shape[1], gx)
    rows, columns = np.nonzero(both)
    basis = (wy[rows][:, :, None] * wx[columns][:, None, :]).reshape(len(rows), gy * gx)
    residual = np.log(r[both]) - np.log(d[both])
    smooth = []
    for j in range(gy):
        for i in range(gx):
            for dj, di in ((0, 1), (1, 0)):
                if j + dj < gy and i + di < gx:
                    row = np.zeros(gy * gx)
                    row[j * gx + i], row[(j + dj) * gx + i + di] = 1, -1
                    smooth.append(row)
    smooth = np.array(smooth).reshape(-1, gy * gx)
    prior = np.zeros(gy * gx) if prior is None else np.asarray(prior, np.float64)
    weight = np.ones(len(residual))
    nodes = np.full(gy * gx, np.median(residual))
    for _ in range(4):
        a = (basis * weight[:, None]).T @ basis / len(residual)
        a += 1e-3 * smooth.T @ smooth + (prior_weight + 1e-6) * np.eye(gy * gx)
        b = (basis * weight[:, None]).T @ residual / len(residual) + prior_weight * prior
        nodes = np.linalg.solve(a, b)
        error = np.abs(basis @ nodes - residual)
        weight = np.where(error < tolerance, 1.0, tolerance / np.maximum(error, 1e-9))
    field = np.exp(_hat(depth.shape[0], gy) @ nodes.reshape(gy, gx) @ _hat(depth.shape[1], gx).T)
    return field.astype(np.float32), nodes


def _hat(size, nodes):
    """(size, nodes) bilinear weights of pixel centres between evenly spread nodes."""
    if nodes == 1:
        return np.ones((size, 1))
    position = (np.arange(size) + 0.5) / size * (nodes - 1)
    weights = np.maximum(0, 1 - np.abs(position[:, None] - np.arange(nodes)[None, :]))
    return weights

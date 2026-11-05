"""Prior Depth Anything on the GPU, behind the Model the depth process (ito.tofmesh.depth) runs:
a camera frame and the ToF's zone centres in it (metric, sparse) in, a dense metric depth map
out. The zones are the prior as they are: points at their own pixels, nothing spread or
invented between them; the model aligns its estimate to them and completes the rest."""

import numpy as np
import torch

from ito._vendor.prior_da import PriorDepthAnything
from ito.recon import models

from . import FOLDER, SIZES


class Model:
    """predict(): a camera frame's depth (h, w) in metres along -Z, or None where the ToF gave
    fewer points in the frame than the model's alignment takes (its K nearest)."""

    metric = True

    def __init__(self, size):
        frozen, conditioned, version = SIZES[size]
        folder = str(models() / FOLDER)
        self.net = PriorDepthAnything(
            device="cuda",
            version=version,
            mde_dir=folder,
            ckpt_dir=folder,
            frozen_model_size=frozen,
            conditioned_model_size=conditioned,
        )

    def predict(self, image, intrinsics, support, zones, sensor_from_camera):
        h, w = image.shape[:2]
        if len(support) < self.net.args.K:
            return None
        u = np.clip(np.rint(support[:, 0]), 0, w - 1).astype(np.int64)
        v = np.clip(np.rint(support[:, 1]), 0, h - 1).astype(np.int64)
        sparse = torch.zeros((1, 1, h, w), device="cuda")
        z = torch.from_numpy(support[:, 2]).to("cuda")
        sparse[0, 0, torch.from_numpy(v), torch.from_numpy(u)] = z
        rgb = torch.from_numpy(image).to("cuda").permute(2, 0, 1)[None]
        depth = self.net(images=rgb, sparse_depths=sparse, sparse_masks=sparse > 0)
        return depth[0, 0].float().cpu().numpy()

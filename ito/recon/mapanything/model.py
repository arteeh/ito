"""MapAnything on the GPU, behind the Model the depth process (ito.tofmesh.depth) runs: a camera
frame and its pinhole intrinsics in, a metric depth map out (upstream's inference, its mask of
ambiguous pixels applied: those pixels have no depth)."""

import json

import torch
from safetensors.torch import load_model

from ito._vendor.mapanything.models.mapanything.model import MapAnything
from ito.recon import models

from . import FOLDER

WIDTH = 518  # the frame's width as the model sees it (upstream's images: 518 pixels across)
PATCH = 14
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # DINOv2's normalisation


class Model:
    """predict(): a camera frame's depth (h, w) in metres along -Z."""

    metric = True

    def __init__(self, size):
        folder = models() / FOLDER
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        # As upstream's from_pretrained: the encoder's own pretrained weights are not fetched,
        # the checkpoint holds them all.
        config["encoder_config"]["torch_hub_pretrained"] = False
        net = MapAnything(**config)
        # load_model: the checkpoint stores tensors the model shares between heads once.
        load_model(net, str(folder / "model.safetensors"), strict=True)
        self.net = net.to("cuda").eval()
        self.mean = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device="cuda").view(1, 3, 1, 1)

    def predict(self, image, intrinsics, support, zones, sensor_from_camera):
        resize = torch.nn.functional.interpolate
        h, w = image.shape[:2]
        height = max(PATCH, round(WIDTH * h / w / PATCH) * PATCH)
        x = torch.from_numpy(image).to("cuda").permute(2, 0, 1)[None].float() / 255
        x = resize(x, (height, WIDTH), mode="bilinear", antialias=True, align_corners=False)
        fx, fy, cx, cy = intrinsics
        sx, sy = WIDTH / w, height / h
        k = torch.tensor(
            [[fx * sx, 0, (cx + 0.5) * sx - 0.5], [0, fy * sy, (cy + 0.5) * sy - 0.5], [0, 0, 1]],
            dtype=torch.float32,
            device="cuda",
        )
        view = dict(img=(x - self.mean) / self.std, data_norm_type=["dinov2"], intrinsics=k[None])
        # Its mask of ambiguous pixels applies; its edge mask does not: the dense mesh cuts its
        # triangles at depth jumps itself (tofmesh.dense), as with every model, and the edge
        # mask also punched holes in flat floor (sim, 5 Nov).
        (out,) = self.net.infer([view], memory_efficient_inference=False, mask_edges=False)
        depth = out["depth_z"][..., 0][None].float()  # (1, 1, height, WIDTH), 0 where masked
        # To the frame's size: interpolated only between pixels that all have a depth.
        known = (depth > 0).float()
        depth = resize(depth * known, (h, w), mode="bilinear", align_corners=False)
        known = resize(known, (h, w), mode="bilinear", align_corners=False)
        depth = torch.where(known > 0.999, depth / known.clamp(min=1e-6), float("nan"))
        return depth[0, 0].cpu().numpy()

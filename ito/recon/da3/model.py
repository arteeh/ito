"""Depth Anything 3 on the GPU: DA3Metric-Large's monocular backbone and head, with the official
checkpoint names, behind the Model the depth process (ito.tofmesh.depth) runs.

Upstream 3d835ec1 (Apache-2.0) is in ito._vendor.da3. We do not need its pose/GS/export
pipeline or its sky replacement. Keep the head in float32, as upstream does.
"""

import torch
from safetensors.torch import load_file
from torch import nn

from ito._vendor.da3.model.dinov2.dinov2 import DinoV2
from ito._vendor.da3.model.dpt import DPT

from . import SIZES, folder

INPUT = (364, 644)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class MetricDepth(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.backbone = DinoV2("vitl", [4, 11, 17, 23], cat_token=False)
        self.head = DPT(
            dim_in=1024, output_dim=1, features=256, out_channels=[256, 512, 1024, 1024]
        )
        state = load_file(str(path / "model.safetensors"))
        # The official hub checkpoint is the API wrapper's state, with a 'model.' prefix.
        self.load_state_dict({k.removeprefix("model."): v for k, v in state.items()}, strict=True)

    def forward(self, image):
        h, w = image.shape[-2:]
        with torch.autocast("cuda", dtype=torch.float16):
            feats, _ = self.backbone(image[:, None], export_feat_layers=[])
        feats = [(x.float(), y.float()) for x, y in feats]
        return self.head(feats, h, w, patch_start_idx=0).depth[:, 0]


class Model:
    """predict(): a camera frame's depth (h, w) in metres along -Z, from the frame alone."""

    metric = True

    def __init__(self, size):
        self.net = MetricDepth(folder(SIZES[size])).to("cuda").eval()
        self.mean = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device="cuda").view(1, 3, 1, 1)

    def predict(self, image, intrinsics, support, zones, sensor_from_camera):
        resize = torch.nn.functional.interpolate
        h, w = image.shape[:2]
        x = torch.from_numpy(image).to("cuda").permute(2, 0, 1)[None].float() / 255
        x = resize(x, INPUT, mode="bilinear", antialias=True, align_corners=False)
        output = self.net((x - self.mean) / self.std)
        # DA3Metric's training focal is 300 pixels. Use the *resized* rectified K,
        # not the original fisheye lens or the output depth map's focal length.
        focal = (intrinsics[0] * INPUT[1] / w + intrinsics[1] * INPUT[0] / h) / 2
        depth = resize(output[:, None].float(), (h, w), mode="bilinear", align_corners=False)
        return (depth[0, 0] * (focal / 300.0)).cpu().numpy()

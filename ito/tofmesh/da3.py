"""DA3Metric-Large's monocular backbone/head, with the official checkpoint names.

Upstream 3d835ec1 (Apache-2.0) is in ito._vendor.da3. We do not need its pose/GS/export
pipeline or its sky replacement. Keep the head in float32, as upstream does.
"""

import torch
from safetensors.torch import load_file
from torch import nn

from ito._vendor.da3.model.dinov2.dinov2 import DinoV2
from ito._vendor.da3.model.dpt import DPT


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

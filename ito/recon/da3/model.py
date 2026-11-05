"""Depth Anything 3 on the GPU, behind the Model the depth process (ito.tofmesh.depth) runs: its
backbone and depth head as each model's config.json describes them, with the official
checkpoint names. DA3Metric-Large's DPT head gives metric depth; the main series' DualDPT head
(Small, Base, Large, Giant) depth up to a scale, which the ToF gives each map.

Upstream 3d835ec1 (Apache-2.0) is in ito._vendor.da3. We do not need its camera encoder and
decoder, pose, GS or export pipeline, or its sky replacement. Keep the head in float32, as
upstream does.
"""

import json

import torch
from safetensors.torch import load_file
from torch import nn

from ito._vendor.da3.model.dinov2.dinov2 import DinoV2
from ito._vendor.da3.model.dpt import DPT
from ito._vendor.da3.model.dualdpt import DualDPT

from . import SIZES, folder

INPUT = (364, 644)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
HEADS = {"DPT": DPT, "DualDPT": DualDPT}


class Network(nn.Module):
    """A DA3 model's monocular depth: its DinoV2 backbone and depth head."""

    def __init__(self, path):
        super().__init__()
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))["config"]

        def args(part):
            return {k: v for k, v in config[part].items() if k != "__object__"}

        self.backbone = DinoV2(**args("net"))
        self.head = HEADS[config["head"]["__object__"]["name"]](**args("head"))
        state = load_file(str(path / "model.safetensors"))
        # The official hub checkpoint is the API wrapper's state, with a 'model.' prefix; the
        # main series' camera encoder and decoder are left out with the pose pipeline.
        state = {k.removeprefix("model."): v for k, v in state.items()}
        mine = {k: v for k, v in state.items() if k.startswith(("backbone.", "head."))}
        missing, unexpected = self.load_state_dict(mine, strict=False)
        # As upstream loads them (non-strict): the main series' checkpoints have no LayerNorm
        # weights for the DualDPT ray head's levels; they keep their initial ones. Nothing else
        # may be missing.
        missing = [k for k in missing if not k.startswith("head.scratch.output_conv2_aux.")]
        if missing or unexpected:
            raise RuntimeError(f"{path.name} does not fit: {missing or unexpected}")

    def forward(self, image):
        h, w = image.shape[-2:]
        with torch.autocast("cuda", dtype=torch.float16):
            feats, _ = self.backbone(image[:, None], export_feat_layers=[])
        feats = [(x.float(), y.float()) for x, y in feats]
        return self.head(feats, h, w, patch_start_idx=0).depth[:, 0]


class Model:
    """predict(): a camera frame's depth (h, w) along -Z: metres for Metric, else relative."""

    def __init__(self, size):
        self.metric = size == "metric"
        self.net = Network(folder(SIZES[size])).to("cuda").eval()
        self.mean = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device="cuda").view(1, 3, 1, 1)

    def predict(self, image, intrinsics, support, zones, sensor_from_camera):
        resize = torch.nn.functional.interpolate
        h, w = image.shape[:2]
        x = torch.from_numpy(image).to("cuda").permute(2, 0, 1)[None].float() / 255
        x = resize(x, INPUT, mode="bilinear", antialias=True, align_corners=False)
        output = self.net((x - self.mean) / self.std)
        depth = resize(output[:, None].float(), (h, w), mode="bilinear", align_corners=False)
        if self.metric:
            # DA3Metric's training focal is 300 pixels. Use the *resized* rectified K,
            # not the original fisheye lens or the output depth map's focal length.
            focal = (intrinsics[0] * INPUT[1] / w + intrinsics[1] * INPUT[0] / h) / 2
            depth = depth * (focal / 300.0)
        return depth[0, 0].cpu().numpy()

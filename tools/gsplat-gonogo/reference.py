"""Save gsplat references for a held-out camera, without retraining."""

import sys

import numpy as np
import torch
from train import Dataset, Model

run = sys.argv[1]
name = sys.argv[2] if len(sys.argv) > 2 else "true"
mode = sys.argv[3] if len(sys.argv) > 3 else "true"
d = Dataset(run, mode)
m = Model(d, maxpoints=100)
folder = d.out / name
ckpt = torch.load(folder / "model.pt", map_location="cuda", weights_only=True)
m.p = torch.nn.ParameterDict(
    {k: torch.nn.Parameter(v, requires_grad=False) for k, v in ckpt["params"].items()}
)
m.filter = ckpt["filter"].cuda()
m.degree = ckpt["degree"]
i = int(d.test[len(d.test) // 2])
with torch.no_grad():
    aa = m.render(i)[0][0].clamp(0, 1).cpu().numpy()
    classic = m.render(i, mode="classic")[0][0].clamp(0, 1).cpu().numpy()
np.savez(
    folder / "match-reference.npz",
    pose=d.poses[i],
    K=d.K.cpu().numpy(),
    frame=i,
    antialiased=aa,
    classic=classic,
)
print("reference", folder, i, flush=True)

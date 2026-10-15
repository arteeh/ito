"""Isolate visible selection from stored geometry on one real causal near-object map.

Software GL only; these images make no timing claim.
"""

import argparse
import os
from pathlib import Path

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["GALLIUM_DRIVER"] = "llvmpipe"
import numpy as np
from measure_gl import View
from oracle_report import grid
from PIL import Image

from ito.protocol import Intrinsics
from ito.render.renderer import LIVE_RENDER_SPLATS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("snapshot", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    data = np.load(args.snapshot)
    k = Intrinsics(width=900, height=900, fx=450, fy=450, cx=450, cy=450)
    cells = [[], []]
    for budget in (None, LIVE_RENDER_SPLATS):
        view = View(k, render_budget=budget)
        try:
            view.upload(data["records"], stable_slots=True)
            for n, offset in enumerate((0.0, 0.03)):
                camera = data["camera"].copy()
                camera[:3, 3] += camera[:3, 0] * offset
                rgb, _ = view.draw(camera)
                # Letterbox whole eye, preserve aspect; no hidden crop or stretch.
                pic = Image.new("RGB", (360, 640))
                pic.paste(
                    Image.fromarray((rgb * 255).astype(np.uint8)).resize((360, 360)), (0, 140)
                )
                cells[n].append(
                    (
                        pic,
                        f"TT / prefix {float(data['deadline']):.1f}s / head +{offset:.2f}m\n"
                        "Full / Input100 / Max1M / 360p30 / round2 0634bbd\n"
                        f"90deg whole eye / visible cap {budget or 'none'}",
                    )
                )
        finally:
            view.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    grid(
        cells,
        args.out,
        "Same stored map/camera: no cap | bounded selection; software GL, no timing",
    )


if __name__ == "__main__":
    main()

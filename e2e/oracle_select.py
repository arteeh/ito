"""Exercise the renderer on saved causal maps and check the bounds-reduction protocol.

Run with the native Windows runtime under ceres.lock; --baseline-shader is the saved
pre-reduction selector. This checks integer IDs/bounds, not numerical image quality.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).parent)]


def main():
    from measure_gl import View

    from ito.protocol import Intrinsics
    from ito.render.renderer import LIVE_RENDER_SPLATS

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--baseline-shader", type=Path, required=True)
    args = p.parse_args()
    k = Intrinsics(width=2160, height=2160, fx=1080, fy=1080, cx=1080, cy=1080)
    view = View(k, render_budget=LIVE_RENDER_SPLATS)
    renderer = view.renderer
    fast = renderer.selection
    old = view.context.compute_shader(args.baseline_shader.read_text())
    cases = []
    try:
        for seq in ("sweep180", "tour"):
            for arm in ("TT", "TF", "TE", "ET", "EE"):
                for moment in (1, 12 if seq == "sweep180" else 78):
                    snap = np.load(
                        args.out / "visual" / f"{seq}-{arm}" / f"snapshot-{moment:.1f}.npz"
                    )
                    view.upload(snap["records"], stable_slots=True)
                    camera = snap["camera"].astype(np.float32)
                    for x in (-0.03, 0, 0.03):
                        head = np.eye(4, dtype=np.float32)
                        head[0, 3] = x
                        outputs, elapsed = [], []
                        for program in (old, fast):
                            renderer.selection = program
                            with view.context:
                                began = time.perf_counter()
                                renderer.sort(camera, head, view.projection)
                                view.context.finish()
                                elapsed.append((time.perf_counter() - began) * 1000)
                                outputs.append(renderer.selected.read())
                        if outputs[0] != outputs[1]:
                            raise AssertionError(f"Selection changed: {seq}/{arm}/{moment}/{x}")
                        cases.append(
                            dict(sequence=seq, arm=arm, at=moment, head_x=x, sort_ms=elapsed)
                        )
        (args.out / "selection-contract.json").write_text(json.dumps(cases, indent=2))
        print(f"Identical selected slot IDs and bounds in {len(cases)} real-map views", flush=True)
    finally:
        renderer.selection = fast
        old.release()
        view.close()


if __name__ == "__main__":
    main()

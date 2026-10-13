"""Fuzz real MASt3R CUDA kernels across Ito's query-batch boundaries.

Windows SLAM environment, under ceres.lock: python e2e/gpu_matching.py --models DIR
No model inference or synthetic replacement of the native kernels.
"""

import argparse
import multiprocessing as mp


def main():
    from pathlib import Path

    from ito.reconstruction import mast3r_runtime
    from ito.reconstruction.admission import Admission, bind_matching

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    args = parser.parse_args()
    mast3r_runtime.MODELS = args.models
    mast3r_runtime.prepare(print)
    import mast3r_slam_backends as native
    import torch

    torch.manual_seed(39)
    original = {name: getattr(native, name) for name in ("iter_proj", "refine_matches")}
    context = mp.get_context("spawn")
    admission = Admission(context, context.RawValue("b", False))
    bind_matching(admission)
    admission.enabled.value = True
    from ito import clock

    admission.deadline.value = clock.now() + 120
    for members in (1, 3):
        for count in (16, 4096, 4112, 8192):
            image = torch.randn(members, 32, 32, 24, device="cuda", dtype=torch.float16)
            queries = torch.randn(members, count, 24, device="cuda", dtype=torch.float16)
            initial = torch.randint(0, 32, (members, count, 2), device="cuda")
            expected = original["refine_matches"](image, queries, initial, 2, 3)
            actual = native.refine_matches(image, queries, initial, 2, 3)
            assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
            rays = torch.randn(members, 32, 32, 9, device="cuda")
            points = torch.randn(members, count, 3, device="cuda")
            guesses = initial.float()
            expected = original["iter_proj"](rays, points, guesses, 10, 0.1, 0.001)
            actual = native.iter_proj(rays, points, guesses, 10, 0.1, 0.001)
            assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
            # A non-block-sized tail is padded internally, then trimmed exactly.
            shorter = count - 1
            actual = native.refine_matches(
                image, queries[:, :shorter].contiguous(), initial[:, :shorter].contiguous(), 2, 3
            )
            expected = original["refine_matches"](image, queries, initial, 2, 3)
            assert torch.equal(actual[0], expected[0][:, :shorter])
    torch.cuda.synchronize()
    print(
        f"PASS: exact native outputs across query/batch boundaries; {admission.chunks.value} fenced chunks"
    )


if __name__ == "__main__":
    main()

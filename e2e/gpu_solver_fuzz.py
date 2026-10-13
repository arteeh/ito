"""Replay real tracking and compare each admitted native solve with the original solve.

ITO_GPU_ENTRY=e2e/gpu_solver_fuzz.py bash e2e/gpu_budget.sh OUTPUT --load slam \
    --admit --frames 2700 --warm-prefix 1560
Diagnostic duplicate solves are excluded from deadline acceptance.
"""

import json

import gpu_budget

from ito.reconstruction import admission as policy

original_bind = policy.bind_native
checked = 0


def bind(admission):
    import mast3r_slam_backends as native
    import torch

    global checked
    names = ("gauss_newton_points", "gauss_newton_rays", "gauss_newton_calib")
    originals = {name: getattr(native, name) for name in names}
    original_bind(admission)
    for name in names:
        bounded = getattr(native, name)
        original = originals[name]

        def verify(*args, bounded=bounded, original=original):
            global checked
            if not admission.enabled.value:
                return bounded(*args)
            expected_poses = args[0].clone()
            result = bounded(*args)
            with admission.cuda_chunk(
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                    lambda start, end: start.elapsed_time(end),
                ),
                kind="verify-solve",
            ):
                expected = original(expected_poses, *args[1:])
                torch.testing.assert_close(args[0], expected_poses, rtol=0, atol=0, equal_nan=True)
                for actual, wanted in zip(result, expected, strict=True):
                    torch.testing.assert_close(actual, wanted, rtol=0, atol=0, equal_nan=True)
            checked += 1
            return result

        setattr(native, name, verify)


policy.bind_native = bind
original_worker = gpu_budget.load_worker


def worker(args, *events):
    original_worker(args, *events)
    (args.out / "solver-fuzz.json").write_text(json.dumps(dict(exact_solves=checked)))
    if not checked:
        raise RuntimeError("Recording did not exercise a native graph solve")


if __name__ == "__main__":
    gpu_budget.load_worker = worker
    gpu_budget.main()

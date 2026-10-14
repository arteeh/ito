r"""Run one released settings choice on a fresh Ceres Microduck and the same short drive.

Under ceres.lock, on Windows:
  python e2e/settings_matrix.py --release C:\path\ito --out C:\path\run
      --precision half --resolution 25
Copies only e2e drivers into the output; Ito and all SLAM code come from the release.
"""

import argparse
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path


def wsl(*args):
    return subprocess.check_output(["wsl.exe", "--exec", *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--precision", choices=("full", "half"), default="full")
    parser.add_argument("--resolution", type=int, choices=(25, 50, 75, 100), default=100)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--route", choices=("short", "pan"), default="short")
    parser.add_argument("--max-splats", type=int, default=1_000_000)
    parser.add_argument("--fps", type=int, default=90)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--profile-network", action="store_true")
    parser.add_argument("--inspect-points", action="store_true")
    parser.add_argument("--network-fault", choices=("oom", "nan"))
    parser.add_argument("--fp16", action="store_true", help="exercise the non-bf16 GPU path")
    parser.add_argument("--limit-gib", type=float)
    args = parser.parse_args()
    assert os.name == "nt", "Run in the Windows desktop session"
    import psutil

    # No live Ito or busy graphics workload may be disturbed by a test.
    assert not any(p.name().lower() == "ito.exe" for p in psutil.process_iter()), (
        "Ito already running"
    )
    args.release, args.out = args.release.resolve(), args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    scripts = args.out / "scripts/e2e"
    scripts.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__).with_name("slam_drive.py"), scripts / "slam_drive.py")
    bootstrap = args.out / "drive.py"
    fault = ""
    if args.network_fault:
        fault = """from ito.reconstruction import mast3r_runtime as runtime
original = runtime.NetworkPrecision.__init__
def inject(self, *args, **kwargs):
    import torch
    original(self, *args, **kwargs)
    calls = 0
"""
        if args.network_fault == "oom":
            fault += """    encoder = self.model._encode_image
    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if 8 <= calls <= 10:
            torch.cuda.set_per_process_memory_fraction(0.0)
            try:
                # Force a real allocator OOM inside the call, even on a roomy GPU.
                torch.empty(2**32, device='cuda', dtype=torch.uint8)
                raise AssertionError('Expected CUDA allocator failure')
            finally:
                torch.cuda.set_per_process_memory_fraction(1.0)
        return encoder(*args, **kwargs)
    self.model._encode_image = fail
"""
        else:
            assert args.fp16, "Non-finite cleanup applies to fp16; use --fp16"
            fault += """    head = self.model._downstream_head
    def poison(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = head(*args, **kwargs)
        if calls == 12:
            result['pts3d'][:, 0, 0] = float('nan')
            result['desc'][:, 0, 1] = float('inf')
        return result
    self.model._downstream_head = poison
"""
        fault += "runtime.NetworkPrecision.__init__ = inject\n"
    if args.profile_network:
        fault += """from ito.reconstruction import mast3r_runtime as runtime
load = runtime.load_model
def measured_load(*args, **kwargs):
    import json, torch
    from mast3r_slam import mast3r_utils
    model = load(*args, **kwargs)
    decode = mast3r_utils.decoder
    calls = 0
    def measured(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls != 4:
            return decode(*args, **kwargs)
        with torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA
        ]) as profile:
            result = decode(*args, **kwargs)
        records = [dict(name=e.key, count=e.count, cpu_us=e.self_cpu_time_total,
                        gpu_us=e.self_device_time_total) for e in profile.key_averages()]
        with open(PROFILE_OUTPUT, 'w') as output:
            json.dump(records, output, indent=2)
        return result
    mast3r_utils.decoder = measured
    return model
runtime.load_model = measured_load
""".replace("PROFILE_OUTPUT", repr(str(args.out / "network-profile.json")))
    if args.inspect_points:
        fault += """from ito.reconstruction.mast3r_frontend import MASt3RFrontend
cloud = MASt3RFrontend._cloud
def inspected_cloud(self, frame, scale):
    import json, numpy as np
    result = cloud(self, frame, scale)
    if frame.frame_id in (30, 75, 120):
        colors = result.colors.cpu().numpy()
        valid = result.valid.cpu().numpy()
        sensor = np.zeros((self.image.height, self.image.width), dtype=bool)
        w, h = self.image.scaled
        x, y = self.image.left, self.image.top
        sensor[y:y+h, x:x+w] = True
        record = dict(frame=frame.frame_id, points_dtype=str(result.points.dtype),
                      color_dtype=str(result.colors.dtype), rgb_dtype=str(frame.uimg.dtype),
                      padding_pixels=int((~sensor).sum()),
                      admitted_padding=int((valid & ~sensor.ravel()).sum()),
                      black_padding=int((valid & ~sensor.ravel() & (colors == 0).all(1)).sum()),
                      black_sensor=int((valid & sensor.ravel() & (colors == 0).all(1)).sum()))
        with open(INSPECT_OUTPUT, 'a') as output:
            output.write(json.dumps(record) + '\\n')
        np.savez(INSPECT_POINTS + str(frame.frame_id) + '.npz',
                 points=result.points.cpu().numpy(), colors=colors, valid=valid,
                 sensor=sensor, confidence=frame.C.cpu().numpy())
    return result
MASt3RFrontend._cloud = inspected_cloud
""".replace("INSPECT_OUTPUT", repr(str(args.out / "points.jsonl"))).replace(
            "INSPECT_POINTS", repr(str(args.out / "points-"))
        )
    if args.fp16:
        fault = "import torch\ntorch.cuda.is_bf16_supported = lambda **kwargs: False\n" + fault
    bootstrap.write_text(
        "import runpy, sys\n"
        + fault
        + "if __name__ == '__main__':\n"
        + f"    sys.path.insert(0, {str(scripts.parent)!r})\n"
        + "    runpy.run_module('e2e.slam_drive', run_name='__main__')\n"
    )
    settings = args.out / "config/ito/settings.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(
        json.dumps(
            dict(
                precision=args.precision,
                input_resolution=args.resolution,
                max_splats=args.max_splats,
                diagnostic_logging=True,
            )
        )
    )
    shared = (
        "/" + "/".join(args.out.parts[1:])
        if args.out.drive.lower().startswith("\\\\wsl.localhost\\")
        else wsl("wslpath", "-u", str(args.out))
    )
    ip = wsl("hostname", "-I").split()[0]
    with socket.socket() as probe:
        probe.settimeout(1)
        assert probe.connect_ex((ip, 8081)) != 0, "Microduck already running; leave it alone"
    pidfile = shared + "/sim.pid"
    command = (
        f"echo $$ > '{pidfile}'; cd ~; "
        f"exec .local/share/ito-microduck/microduck-sim --viewer --logs '{shared}/sim-logs' "
        f"--pairing-file '{shared}/pairing-code'"
    )
    filler = None
    env = os.environ | {"PYTHONUTF8": "1"}
    if args.record:
        env["ITO_RECORD"] = str(args.out / "motion")
    simlog = (args.out / "sim.log").open("w")
    sim = subprocess.Popen(
        ["wsl.exe", "--exec", "sh", "-c", command],
        stdout=simlog,
        stderr=simlog,
    )
    try:
        deadline = time.monotonic() + 90
        while True:
            assert sim.poll() is None, "Microduck failed; read sim.log"
            with socket.socket() as probe:
                probe.settimeout(0.5)
                ready = probe.connect_ex((ip, 8081)) == 0
            if ready:
                break
            assert time.monotonic() < deadline, "Microduck startup timeout"
            time.sleep(0.5)
        code = (args.out / "pairing-code").read_text().strip()
        # This allocation belongs to the test only; the product has no memory cap lever.
        if args.limit_gib:
            fill = args.out / "fill.py"
            fill.write_text(
                "import torch, time\n"
                f"available = {args.limit_gib} * 2**30\n"
                "free, total = torch.cuda.mem_get_info()\n"
                "filler = torch.empty(max(0, int(total - available)), device='cuda', "
                "dtype=torch.uint8)\n"
                "filler.zero_(); torch.cuda.synchronize()\n"
                "print(torch.cuda.mem_get_info(), flush=True)\n"
                "time.sleep(180)\n"
            )
            filler = subprocess.Popen(
                [str(args.release / "runtime/python.exe"), "-I", str(fill)],
                stdout=(args.out / "fill.log").open("w"),
                env=env,
            )
            time.sleep(5)
            assert filler.poll() is None
        with (args.out / "drive.log").open("w") as log:
            result = subprocess.run(
                [
                    str(args.release / "runtime/python.exe"),
                    "-I",
                    str(bootstrap),
                    f"{ip}:8081",
                    "--code",
                    code,
                    "--out",
                    str(args.out),
                    "--fps",
                    str(args.fps),
                    "--max-splats",
                    str(args.max_splats),
                    "--route",
                    args.route,
                    "--turn-rate",
                    "15" if args.route == "pan" else "45",
                    "--open-loop",
                    "--no-shots",
                    "--startup-timeout",
                    "120",
                ],
                cwd=args.out,
                env=env,
                stdout=log,
                stderr=log,
                timeout=210,
            )
        assert result.returncode == 0, f"Drive failed ({result.returncode}); read drive.log"
        rows = [json.loads(line) for line in (args.out / "metrics.jsonl").read_text().splitlines()]
        live = [r for r in rows if r["link"] == "CONNECTED" and not r["flat_video"]]
        assert live, "No reconstructed view"
        values = [r["performance"] for r in live if r.get("performance", {}).get("slam_precision")]
        if args.baseline:
            values = [r["performance"] for r in live if r.get("performance")]
        else:
            assert values, "No active precision in overlay metrics"
            assert all(v["slam_precision"].startswith(args.precision.title()) for v in values)
            if args.network_fault == "oom":
                warnings = [v for v in values if v["reconstruction_oom"]]
                assert warnings, "Out-of-memory warning never appeared"
                assert not values[-1]["reconstruction_oom"], "Warning did not clear after recovery"
                assert values[-1]["network_oom_count"] == 3
                assert values[-1]["backend_calls"] > warnings[-1]["backend_calls"]
            elif args.network_fault == "nan":
                assert values[-1]["network_nonfinite_points"] >= 2
            else:
                assert values[-1]["network_oom_count"] == 0
                assert values[-1]["network_nonfinite_points"] == 0
        report = dict(
            precision=args.precision,
            resolution=args.resolution,
            reconstructed_frames=len(live),
            overlay=values[-1],
        )
        (args.out / "settings-result.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    finally:
        if filler:
            filler.terminate()
            filler.wait(10)
        if sim.poll() is None:
            pid = wsl("cat", pidfile)
            assert pid.isdigit()
            wsl("kill", "-TERM", pid)
            sim.wait(30)
        simlog.close()


if __name__ == "__main__":
    main()

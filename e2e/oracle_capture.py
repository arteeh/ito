"""Record current velstand-v8 sweep and walking tour, including exact RGB-D.

PYTHONPATH=. .venv/bin/python e2e/oracle_capture.py --out ~/Projects/ito/out/oracle-40
Run under ceres.lock. Uses the installed sim's environments without modifying them.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from drivers.microduck import lens
from drivers.microduck.sim import simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sequence", choices=("sweep180", "tour", "short"), required=True)
    parser.add_argument("--board", choices=("beta", "zero3"), default="beta")
    parser.add_argument("--beta-camera-hfov", type=float, default=lens.BETA_CAMERA_HFOV_DEG)
    parser.add_argument("--head-imu", action="store_true")
    parser.add_argument("--imu-noise", type=float, default=0)
    parser.add_argument("--tof-noise", type=float, default=1)
    parser.add_argument(
        "--record-stride",
        type=int,
        default=1,
        help="subsample saved evaluation pixels; live camera is unchanged",
    )
    parser.add_argument(
        "--no-live-record", action="store_true", help="retain source and observation records only"
    )
    args = parser.parse_args()
    args.out = args.out.resolve() / args.sequence
    args.out.mkdir(parents=True, exist_ok=True)
    install = Path.home() / ".local/share/ito-microduck"
    env = os.environ | {
        "PYTHONPATH": str(Path.cwd()),
        "GALLIUM_DRIVER": "d3d12",
        "ITO_RECORD": str(args.out / "live"),
        "XDG_CONFIG_HOME": str(args.out / "config"),
    }
    os.environ["ITO_MICRODUCK_RECORD_STRIDE"] = str(args.record_stride)
    if args.no_live_record:
        env.pop("ITO_RECORD", None)
    (args.out / "capture.json").write_text(
        json.dumps(
            dict(
                sim_camera="GC2093 1920x1080/30 upright"
                if args.board == "beta"
                else "IMX219 360p30 rotated",
                board=args.board,
                evaluation_pixel_stride=args.record_stride,
                imu_noise=args.imu_noise,
                tof_noise=args.tof_noise,
                seed=0,
                lens_long_side_degrees=args.beta_camera_hfov
                if args.board == "beta"
                else lens.HFOV_DEG,
                route="sweep" if args.sequence == "sweep180" else args.sequence,
                turn_rate=180 if args.sequence == "sweep180" else 45,
                policy=json.loads((install / "policies/current/manifest.json").read_text()),
            ),
            indent=2,
        )
    )
    with simulation(
        install / "microduck",
        install / "microduck_rl",
        install / "policies/current",
        logs=args.out / "sim-logs",
        pairing_file=args.out / "pairing-code",
        viewer=True,
        board=args.board,
        beta_camera_hfov=args.beta_camera_hfov,
        gl="glfw",
        rgbd_record=args.out / "source",
        head_imu=args.head_imu,
        imu_noise=args.imu_noise,
        tof_noise=args.tof_noise,
    ) as sim:
        command = [
            sys.executable,
            "e2e/slam_drive.py",
            f"127.0.0.1:{sim.port}",
            "--code",
            sim.code,
            "--reconstruction",
            "video",
            "--record",
            str(args.out / "received"),
            "--out",
            str(args.out / "drive"),
            "--route",
            "sweep" if args.sequence == "sweep180" else args.sequence,
            "--turn-rate",
            "180" if args.sequence == "sweep180" else "45",
            "--fps",
            "90",
            "--open-loop",
            "--no-shots",
        ]
        subprocess.run(command, env=env, check=True, timeout=240)


if __name__ == "__main__":
    main()

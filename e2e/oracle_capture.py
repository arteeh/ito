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

from drivers.microduck.sim import simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sequence", choices=("sweep180", "tour", "short"), required=True)
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
    (args.out / "capture.json").write_text(
        json.dumps(
            dict(
                sim_camera="360p30 (upright 360x640)",
                lens_long_side_degrees=62,
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
        gl="glfw",
        rgbd_record=args.out / "source",
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

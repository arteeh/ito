"""Run a Windows build of Ito against the virtual Microduck, drive it with a script and
record videos of it. Run in WSL on the Windows PC:

    uv run tools/session.py BUILD NAME [--script FILE] [--sim-prefix DIR] [--viewer]

BUILD is a folder under C:\\Users\\me\\Projects\\ito\\builds (release.ps1 -Branch ...).
Output goes to C:\\Users\\me\\Projects\\ito\\out\\sessions\\NAME: screen.mp4 (the Ito window),
camera.mp4 (the robot camera frames Ito received), steps.txt (when each script step
started, in video time), ito.log, sim.log and the sim's own logs. Script steps are
described in ito/desktop/input.py (PilotScript). Look at frames with tools/frame.py.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import imageio_ffmpeg

HERE = Path(__file__).resolve().parents[1]
ROOT = Path("/mnt/c/Users/me/Projects/ito")


def windows(path: Path) -> str:
    return subprocess.run(
        ["wslpath", "-w", str(path)], capture_output=True, text=True
    ).stdout.strip()


def stop_sim():
    subprocess.run(["sh", str(ROOT / "stop-sim.sh")], capture_output=True)


def encode(stream: Path, video: Path, fps: int = 30) -> int:
    """JPEGs + frames.jsonl -> mp4 at a steady rate, each frame shown for as long as it was."""
    rows = [json.loads(line) for line in (stream / "frames.jsonl").read_text().splitlines()]
    if not rows:
        return 0
    listing = stream / "frames.txt"
    with listing.open("w") as out:
        for row, after in zip(rows, rows[1:] + [None], strict=True):
            duration = 1 / fps if after is None else max(after["t"] - row["t"], 1e-3)
            out.write(f"file '{stream / row['file']}'\nduration {duration:.4f}\n")
        out.write(f"file '{stream / rows[-1]['file']}'\n")
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-fps_mode",
            "cfr",
            "-r",
            str(fps),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    return len(rows)


def step_times(log: Path, start: float) -> str:
    """When each script step started, in seconds into screen.mp4 (from ito.log)."""
    lines = []
    for found in re.finditer(
        r"Pilot script step (\d+) t=([\d.]+): (.*)", log.read_text("utf-8", "replace")
    ):
        lines.append(f"{float(found[2]) - start:7.2f}s  step {found[1]}: {found[3].strip()}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("build")
    parser.add_argument("name")
    parser.add_argument("--script", type=Path, default=HERE / "tools/scripts/room-tour.json")
    parser.add_argument(
        "--sim-source",
        type=Path,
        default=HERE,
        help="checkout whose drivers/microduck runs the sim (default this one)",
    )
    parser.add_argument(
        "--sim-prefix",
        type=Path,
        default=Path.home() / ".local/share/ito-microduck",
        help="sim install from drivers/microduck/setup-sim.sh",
    )
    parser.add_argument("--viewer", action="store_true", help="also open the MuJoCo viewer")
    parser.add_argument(
        "--sim-args", default="", help='extra sim options, e.g. "--camera-height 1080"'
    )
    args = parser.parse_args()

    exe = ROOT / "builds" / args.build / "ito" / "ito.exe"
    if not exe.exists():
        sys.exit(f"no build at {exe}")
    out = ROOT / "out" / "sessions" / args.name
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    shutil.copy(args.script, out / "script.json")
    source = subprocess.run(
        ["git", "-C", str(args.sim_source), "log", "--oneline", "-1"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    (out / "about.txt").write_text(
        f"build {args.build}\nsim source {source}\nsim prefix {args.sim_prefix}\n"
        f"sim args {args.sim_args}\n"
    )

    stop_sim()
    prefix = args.sim_prefix
    sim_log = (out / "sim.log").open("w")
    sim = subprocess.Popen(
        [
            str(prefix / "ito-venv/bin/python"),
            "-m",
            "drivers.microduck.sim",
            "--microduck",
            str(prefix / "microduck"),
            "--rl",
            str(prefix / "microduck_rl"),
            "--policies",
            str(prefix / "policies/current"),
            *(["--viewer"] if args.viewer else []),
            *args.sim_args.split(),
        ],
        cwd=args.sim_source,
        stdout=sim_log,
        stderr=subprocess.STDOUT,
        env=os.environ
        | {
            "PYTHONPATH": str(args.sim_source),
            "GST_PLUGIN_PATH": str(prefix / "gst-plugins-rs/target/debug"),
        },
    )
    try:
        deadline = time.monotonic() + 90
        while "Pairing code" not in (out / "sim.log").read_text(errors="replace"):
            if sim.poll() is not None or time.monotonic() > deadline:
                sys.exit("sim did not start:\n" + (out / "sim.log").read_text()[-2000:])
            time.sleep(0.5)
        ip = subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split()[0]
        code = (Path.home() / ".config/ito/microduck-sim-code").read_text().strip()
        rec = out / "rec"
        print(f"recording {args.build} into {windows(out)}", flush=True)
        command = (
            f"set ITO_RECORD={windows(rec)}&& set ITO_PILOT_SCRIPT={windows(out / 'script.json')}"
            # The build's own Python rather than ito.exe, whose log never reaches a pipe.
            f"&& runtime\\python.exe -I -X utf8 -m ito.app {ip}:8081 --code {code}"
            f" --capture-dir {windows(out / 'captures')}"
        )
        with (out / "ito.log").open("w") as log:
            result = subprocess.run(
                ["cmd.exe", "/c", command], cwd=exe.parent, stdout=log, stderr=subprocess.STDOUT
            )
        if result.returncode:
            print(f"ito exited {result.returncode}", flush=True)
    finally:
        stop_sim()
        sim.wait(timeout=30)
        shutil.copytree(
            Path.home() / ".local/state/ito/microduck-sim", out / "sim-logs", dirs_exist_ok=True
        )

    for name in ("screen", "camera"):
        if (rec / name / "frames.jsonl").exists():
            count = encode(rec / name, out / f"{name}.mp4")
            print(f"{name}.mp4: {count} frames", flush=True)
    index = rec / "screen/frames.jsonl"
    if index.exists() and index.stat().st_size:
        start = json.loads(index.read_text().splitlines()[0])["t"]
        (out / "steps.txt").write_text(step_times(out / "ito.log", start))
    shutil.rmtree(rec, ignore_errors=True)
    print(f"done: {out}")


if __name__ == "__main__":
    main()

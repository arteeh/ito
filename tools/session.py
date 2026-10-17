"""Run a Windows build of Ito against the virtual Microduck, drive it with a script and
record videos of it. Run in WSL on the Windows PC:

    uv run tools/session.py BUILD NAME [--script FILE] [--sim-prefix DIR] [--viewer]

BUILD is a folder under C:\\Users\\me\\Projects\\ito\\builds (release.ps1 -Branch ...).
Output goes to C:\\Users\\me\\Projects\\ito\\out\\sessions\\NAME: screen.mp4 (the Ito window),
camera.mp4 (the robot camera frames Ito received), map.mp4 (each frame reconstruction
mapped, beside the map drawn from the pose it mapped it by), steps.txt (when each script step
started, in seconds and as frame numbers of both videos), ito.log, sim.log and the sim's
own logs. Script steps are described in ito/desktop/input.py (PilotScript). Look at frames
with tools/frame.py.
"""

import argparse
import bisect
import json
import os
import re
import shutil
import signal
import socket
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


def stop_sim(sim):
    """Ctrl+C the sim this run started; if it hangs, end it and the daemons it spawned
    (each runs in its own session), never anything else."""
    if sim.poll() is not None:
        return
    tree, parents = [], [sim.pid]
    while parents:
        found = subprocess.run(
            ["ps", "-o", "pid=", "--ppid", ",".join(map(str, parents))],
            capture_output=True,
            text=True,
        ).stdout.split()
        parents = [int(pid) for pid in found]
        tree += parents
    sim.send_signal(signal.SIGINT)
    try:
        sim.wait(timeout=30)
    except subprocess.TimeoutExpired:
        print("sim ignored Ctrl+C; terminating it", flush=True)
        sim.terminate()
    for sig in (signal.SIGTERM, signal.SIGKILL):
        alive = [pid for pid in tree if Path(f"/proc/{pid}").exists()]
        for pid in alive:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        if alive:
            time.sleep(3)
    sim.wait(timeout=10)


def encode(stream: Path, video: Path, fps: int = 30) -> int:
    """JPEGs + frames.jsonl -> mp4 holding each recorded frame exactly once (frame N of the
    video is recorded frame N), each shown for as long as it was."""
    rows = [json.loads(line) for line in (stream / "frames.jsonl").read_text().splitlines()]
    if not rows:
        return 0
    listing = stream / "frames.txt"
    with listing.open("w") as out:
        for row, after in zip(rows, rows[1:] + [None], strict=True):
            duration = 1 / fps if after is None else max(after["t"] - row["t"], 1e-3)
            out.write(f"file '{stream / row['file']}'\nduration {duration:.4f}\n")
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
            "passthrough",
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


def frame_times(stream: Path) -> list[float]:
    index = stream / "frames.jsonl"
    if not index.exists():
        return []
    return [json.loads(line)["t"] for line in index.read_text().splitlines()]


def step_times(log: Path, screen: list[float], camera: list[float], mapped: list[float]) -> str:
    """When each script step started (from ito.log): seconds into screen.mp4, and the first
    frame number of screen.mp4, camera.mp4 and map.mp4 at or after it, for tools/frame.py."""
    lines = ["seconds  screen  camera     map  step"]
    for found in re.finditer(
        r"Pilot script step (\d+) t=([\d.]+): (.*)", log.read_text("utf-8", "replace")
    ):
        t = float(found[2])
        frames = [bisect.bisect_left(times, t) for times in (screen, camera, mapped)]
        lines.append(
            f"{t - screen[0]:7.2f}  {frames[0]:6d}  {frames[1]:6d}  {frames[2]:6d}"
            f"  {found[1]}: {found[3].strip()}"
        )
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
    parser.add_argument("--precision", choices=("full", "half"))
    parser.add_argument("--input-resolution", type=int, choices=(25, 50, 75, 100))
    parser.add_argument("--max-splats", type=int)
    parser.add_argument(
        "--correction-after",
        type=float,
        help="manually inject a 35-degree/1.2-metre estimator gauge correction after N seconds",
    )
    parser.add_argument(
        "--oom-after",
        type=float,
        help="inject one CUDA allocation failure after N seconds; inspect warning and precision",
    )
    parser.add_argument(
        "--truth",
        choices=("poses", "depth", "poses,depth", "compare"),
        help="map by the sim's exact camera poses and/or depth instead of the estimates"
        " (ITO_TRUTH, a diagnostic)",
    )
    parser.add_argument("--depth", choices=("mast3r", "da3"), help="ITO_DEPTH, an experiment")
    parser.add_argument("--tracker", choices=("mast3r", "map"), help="ITO_TRACKER, an experiment")
    parser.add_argument(
        "--sim-args", default="", help='extra sim options, e.g. "--camera-height 1080"'
    )
    args = parser.parse_args()

    exe = ROOT / "builds" / args.build / "ito" / "ito.exe"
    if not exe.exists():
        sys.exit(f"no build at {exe}")
    out = ROOT / "out" / "sessions" / args.name
    out.mkdir(parents=True, exist_ok=False)
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
    build_log = ROOT / "builds" / f"{args.build}.log"
    if build_log.exists():
        commit = re.search(r"^commit .+$", build_log.read_text("utf-8-sig", errors="replace"), re.M)
        if commit:
            with (out / "about.txt").open("a") as about:
                about.write(commit[0].strip() + "\n")

    with socket.socket() as probe:
        try:
            probe.bind(("0.0.0.0", 8081))
        except OSError:
            sys.exit("port 8081 is in use; leaving the existing simulator alone")
    settings = Path("/mnt/c/Users/me/AppData/Roaming/ito/settings.json")
    values = (
        json.loads(settings.read_text())
        if settings.exists()
        else {
            "precision": "full",
            "input_resolution": 100,
            "max_splats": 1_000_000,
            "diagnostic_logging": False,
        }
    )
    for name in ("precision", "input_resolution", "max_splats"):
        if getattr(args, name) is not None:
            values[name] = getattr(args, name)
    profile = out / "profile"
    (profile / "ito").mkdir(parents=True)
    (out / "settings.json").write_text(json.dumps(values) + "\n")
    shutil.copy(out / "settings.json", profile / "ito/settings.json")
    if (settings.parent / "robots").exists():
        shutil.copytree(settings.parent / "robots", profile / "ito/robots")
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
            "--logs",
            str(out / "sim-logs"),
            "--pairing-file",
            str(out / "pairing-code"),
            *(["--viewer"] if args.viewer else []),
            *args.sim_args.split(),
        ],
        cwd=args.sim_source,
        stdout=sim_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=os.environ
        | ({"ITO_MICRODUCK_TRUTH_DEPTH": "3"} if args.truth and args.truth != "poses" else {})
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
        code = (out / "pairing-code").read_text().strip()
        rec = out / "rec"
        print(f"recording {args.build} into {windows(out)}", flush=True)
        entry = "-m ito.app"
        correction_env = ""
        if args.correction_after is not None or args.oom_after is not None:
            shutil.copy(HERE / "tools/scene_correction.py", out / "scene_correction.py")
            entry = windows(out / "scene_correction.py")
        if args.correction_after is not None:
            correction_env = f"&& set ITO_MANUAL_CORRECTION_AFTER={args.correction_after}"
            with (out / "about.txt").open("a") as about:
                about.write(f"manual estimator gauge correction after {args.correction_after} s\n")
        if args.oom_after is not None:
            correction_env += f"&& set ITO_MANUAL_OOM_AFTER={args.oom_after}"
            with (out / "about.txt").open("a") as about:
                about.write(f"manual CUDA allocation failure after {args.oom_after} s\n")
        for name in ("depth", "tracker"):
            if getattr(args, name):
                correction_env += f"&& set ITO_{name.upper()}={getattr(args, name)}"
                with (out / "about.txt").open("a") as about:
                    about.write(f"ITO_{name.upper()}={getattr(args, name)}\n")
        if args.truth:
            correction_env += f"&& set ITO_TRUTH={args.truth}"
            with (out / "about.txt").open("a") as about:
                about.write(f"ITO_TRUTH={args.truth}\n")
        command = (
            f"set APPDATA={windows(profile)}&& set ITO_RECORD={windows(rec)}"
            f"&& set ITO_PILOT_SCRIPT={windows(out / 'script.json')}"
            f"&& set ITO_TIMELINE={windows(out / 'timeline.jsonl')}"
            f"{correction_env}"
            # The build's own Python rather than ito.exe, whose log never reaches a pipe.
            f"&& runtime\\python.exe -I -X utf8 {entry} {ip}:8081 --code {code}"
            f" --capture-dir {windows(out / 'captures')}"
            f" --metrics {windows(out / 'metrics.jsonl')}"
        )
        with (out / "ito.log").open("w") as log:
            result = subprocess.run(
                ["cmd.exe", "/c", command], cwd=exe.parent, stdout=log, stderr=subprocess.STDOUT
            )
        if result.returncode:
            print(f"ito exited {result.returncode}", flush=True)
    finally:
        stop_sim(sim)
        (out / "pairing-code").unlink(missing_ok=True)
        shutil.rmtree(profile, ignore_errors=True)

    for name in ("screen", "camera", "map"):
        if (rec / name / "frames.jsonl").exists():
            count = encode(rec / name, out / f"{name}.mp4")
            print(f"{name}.mp4: {count} frames", flush=True)
    if (rec / "screen/frames.jsonl").exists():
        shutil.copy(rec / "screen/frames.jsonl", out / "screen.jsonl")
    screen = frame_times(rec / "screen")
    if screen:
        steps = step_times(
            out / "ito.log", screen, frame_times(rec / "camera"), frame_times(rec / "map")
        )
        (out / "steps.txt").write_text(steps)
    shutil.rmtree(rec, ignore_errors=True)
    print(f"done: {out}")


if __name__ == "__main__":
    main()

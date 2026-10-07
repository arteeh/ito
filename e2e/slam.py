"""Real RGB-only MuJoCo -> WebRTC -> pilot. Add --cuda to require real MASt3R-SLAM.

uv run python e2e/slam.py
uv run --python 3.12 --extra slam python e2e/slam.py --cuda
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import psutil
import pygame

from ito.app.__main__ import main as pilot_main
from ito.reconstruction.mast3r_runtime import MODELS


def key(code, down=None):
    if down is None:
        key(code, True)
        key(code, False)
    else:
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN if down else pygame.KEYUP, key=code))


@contextmanager
def model_condition(condition):
    if condition == "present":
        yield
        return
    saved = MODELS.with_name(".slam-e2e-saved")
    if saved.exists():
        raise RuntimeError(f"Restore interrupted e2e models from {saved} first")
    existed = MODELS.exists()
    if existed:
        MODELS.rename(saved)
    try:
        if condition == "corrupt":
            MODELS.mkdir()
            with (MODELS / "model.safetensors").open("wb") as damaged:
                damaged.write(b"damaged model")
                if existed:
                    damaged.truncate((saved / "model.safetensors").stat().st_size)
        yield
    finally:
        if condition == "corrupt":
            for path in MODELS.iterdir():
                path.unlink()
            MODELS.rmdir()
        if existed:
            saved.rename(MODELS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="require real CUDA tracking and splats")
    parser.add_argument(
        "--models",
        choices=("missing", "corrupt", "present"),
        default="missing",
        help="model failure to exercise without --cuda",
    )
    parser.add_argument("--mode", choices=("desktop", "xr"), default="desktop")
    parser.add_argument("--startup-timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.cuda:
        args.models = "present"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    out = Path(
        "e2e/out/slam-"
        + ("cuda" if args.cuda else "no-cuda")
        + ("-xr" if args.mode == "xr" else "")
        + ("-" + args.models if not args.cuda else "")
    )
    out.mkdir(parents=True, exist_ok=True)
    for capture in out.glob("capture-*.png"):
        capture.unlink()
    os.environ["XDG_CONFIG_HOME"] = os.environ["APPDATA"] = str(out / "config")
    if not args.cuda:
        os.environ["XDG_CACHE_HOME"] = os.environ["LOCALAPPDATA"] = str(out / "cache")
    network_log = (out / "network.log").resolve()
    network_log.write_text("")
    network_log.with_suffix(".pids").write_text("")
    os.environ["ITO_E2E_NETWORK_LOG"] = str(network_log)
    audit_path = str(Path(__file__).resolve().parent / "offline")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        filter(None, [audit_path, os.environ.get("PYTHONPATH")])
    )
    # Audit this interpreter too; sitecustomize covers the spawned reconstruction process.
    sys.path.insert(0, audit_path)
    import sitecustomize

    sitecustomize.log = str(network_log)
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        os.environ[name] = "http://127.0.0.1:1"
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost,::1"
    with socket.socket() as port:
        port.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{port.getsockname()[1]}"
    env = os.environ.copy()
    if sys.platform == "linux":
        env.update(MUJOCO_GL="osmesa", LD_LIBRARY_PATH="/opt/data/lib/osmesa")
    began = changed = time.monotonic()
    stage = 0
    previous = began
    display_times = []
    camera_frames = []
    poses = []
    tracking_updates = []
    counts = []
    messages = set()
    workers = set()
    suspended = None
    first_position = None
    tracked_before_stall = 0
    video_before_failure = 0
    error = None
    model = MODELS / "model.safetensors"
    expected = {
        "missing": f"MASt3R model missing from {model}",
        "corrupt": f"MASt3R model checksum failed at {model}",
        "present": f"MASt3R CUDA extensions missing from {MODELS / 'native'}",
    }[args.models]
    if not args.cuda and args.models == "present":
        manifest = json.loads((MODELS / "manifest.json").read_text())
        if manifest["native"]:
            expected = "MASt3R needs an NVIDIA CUDA device and driver"

    def drive(app, window, value):
        nonlocal stage, changed, previous, suspended, first_position, tracked_before_stall
        nonlocal video_before_failure, error
        now = time.monotonic()
        limit = args.startup_timeout if args.cuda and stage == 0 else 60
        assert now - changed < limit, (stage, app.reconstruction_status, app.failure)
        display_times.append(now - previous)
        previous = now
        messages.add(app.reconstruction_status)
        if app.worker:
            workers.add(app.worker.process.pid)
        counts.append(window.renderer.count)
        if app.state.video is not None and len(camera_frames) < 2 and stage in (0, 2):
            camera_frames.append(app.state.video.copy())
        t = app.telemetry
        if app.tracked_frames:
            poses.append(app.camera_pose.copy())
            # Measure live tracking before deliberately suspending/killing the worker.
            if stage < 4 and (
                not tracking_updates or app.tracked_frames != tracking_updates[-1][1]
            ):
                tracking_updates.append((now, app.tracked_frames))
        if args.cuda and app.failure and stage < 5:
            raise AssertionError("Real CUDA SLAM failed: " + app.failure)
        ready = (
            app.tracked_frames >= 3 and window.renderer.count > 500 and not app.state.flat_video
            if args.cuda
            else app.failure == expected
            and app.state.status.reconstruction == expected + "; showing flat camera feed"
        )
        if stage == 0 and ready and app.state.video is not None and t:
            assert app.backend == "slam" and app.connections == 1
            if not args.cuda:
                assert app.state.flat_video and window.renderer.count == 0
                assert app.state.status.reconstruction == expected + "; showing flat camera feed"
                assert not any(
                    word in app.failure.lower() for word in ("build", "uv ", "pip ", "download")
                )
                assert "flat camera feed" in app.reconstruction_status
            first_position = np.array([t["base_x"], t["base_y"]])
            key(pygame.K_F12)
            key(pygame.K_r)
            key(pygame.K_w, True)
            stage, changed = 1, now
        elif stage == 1 and now - changed > 3:
            distance = np.linalg.norm(np.array([t["base_x"], t["base_y"]]) - first_position)
            assert distance > 0.15, ("Robot did not move while reconstructing/falling back", t)
            key(pygame.K_w, False)
            key(pygame.K_e)
            key(pygame.K_F12)
            stage, changed = 2, now
        elif stage == 2 and now - changed > 1:
            assert app.state.status.e_stop
            assert t["left_command"] == t["right_command"] == 0
            assert app.connections == 1, "Reconstruction failure restarted the link"
            if args.cuda:
                assert len(poses) > 5 and np.isfinite(poses).all()
                assert np.linalg.norm(poses[-1][:3, 3] - poses[0][:3, 3]) > 0.05
                app.set_max_splats(2048)
                stage, changed = 3, now
            else:
                error = app.failure
                pygame.event.post(pygame.event.Event(pygame.QUIT))
                stage = 7
        elif stage == 3 and now - changed > 5:
            assert 100 < window.renderer.count <= 2048, window.renderer.count
            suspended = psutil.Process(app.worker.process.pid)
            suspended.suspend()
            tracked_before_stall = app.tracked_frames
            stage, changed = 4, now
        elif stage == 4 and now - changed > 3:
            assert app.tracked_frames == tracked_before_stall
            assert app.state.flat_video, "Stalled tracking did not switch to flat video"
            assert app.state.status.link == "CONNECTED"
            assert now - app.state.video_time < 1
            key(pygame.K_F12)
            suspended.resume()
            suspended = None
            stage, changed = 5, now
        elif stage == 5 and app.tracked_frames > tracked_before_stall + 2:
            assert not app.state.flat_video
            # A native crash must leave the camera and safety controls usable too.
            app.worker.process.kill()
            video_before_failure = app.state.video_time
            stage, changed = 6, now
        elif stage == 6 and now - changed > 3:
            assert app.failure and app.state.flat_video
            assert app.state.video_time > video_before_failure
            assert app.connections == 1 and app.state.status.link == "CONNECTED"
            error = app.failure
            key(pygame.K_F12)
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            stage = 7

    def run_pilot():
        result = pilot_main(
            [
                address,
                "--mode",
                args.mode,
                "--size",
                "960",
                "720",
                "--fps",
                "60",
                "--reconstruction",
                "auto",
                "--max-splats",
                "16384",
                "--capture-dir",
                str(out),
                "--metrics",
                str(out / "metrics.jsonl"),
            ],
            on_frame=drive,
        )
        assert result == 0 and stage == 7, (result, stage)
        # A pilot's explicit selection must survive the next app launch.
        for override in (["--reconstruction", "video"], []):
            verified = False

            def check_setting(app, window, value):
                nonlocal verified
                if app.connections and app.state.video is not None:
                    assert app.settings.reconstruction == "video"
                    assert app.backend == "video" and app.worker is None and app.state.flat_video
                    verified = True
                    pygame.event.post(pygame.event.Event(pygame.QUIT))

            assert (
                pilot_main(
                    [address, "--size", "640", "480", "--frames", "1200", *override],
                    on_frame=check_setting,
                )
                == 0
            )
            assert verified

    with (out / "driver.log").open("w") as log:
        driver = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "drivers.mujoco.cli",
                "--port",
                address.split(":")[1],
                "--rgb-only",
                "--speed",
                "0.2",
                "--fps",
                "20",
            ],
            env=env,
            stdout=log,
            stderr=log,
        )
        try:
            with model_condition(args.models if not args.cuda else "present"):
                run_pilot()
        finally:
            if suspended:
                suspended.resume()
            driver.terminate()
            driver.wait(timeout=10)
    captures = sorted(out.glob("capture-*-left.png" if args.mode == "xr" else "capture-*.png"))
    assert len(captures) >= 2
    pictures = [
        pygame.surfarray.array3d(pygame.image.load(p))[
            :, 400 if args.mode == "desktop" else 0 :
        ].astype(float)
        for p in captures[:2]
    ]
    assert pictures[0].std() > 15, "Flat panel/scene is blank"
    assert np.abs(pictures[0] - pictures[1]).mean() > 2, "Camera/scene did not update during motion"
    frame_p95 = float(np.percentile(display_times[10:], 95))
    assert frame_p95 < 0.15, frame_p95
    assert network_log.read_text() == "", (
        "Pilot attempted external network I/O: " + network_log.read_text()
    )
    audited = {int(pid) for pid in network_log.with_suffix(".pids").read_text().splitlines()}
    assert workers and workers <= audited, ("Worker network audit not installed", workers, audited)
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    input_latency = [
        row["pilot_input_to_robot_ms"] for row in rows if row["pilot_input_to_robot_ms"] is not None
    ]
    splat_latency = [
        row["capture_to_splat_visible_ms"]
        for row in rows
        if not row["flat_video"] and row["capture_to_splat_visible_ms"] is not None
    ]
    report = dict(
        cuda=args.cuda,
        external_network_attempts=0,
        models=args.models,
        display_frame_ms_p95=frame_p95 * 1000,
        max_splats=max(counts),
        tracked_pose_samples=len(poses),
        tracking_hz=(
            (tracking_updates[-1][1] - tracking_updates[0][1])
            / (tracking_updates[-1][0] - tracking_updates[0][0])
            if len(tracking_updates) > 1
            else None
        ),
        pilot_input_to_robot_ms_median=(float(np.median(input_latency)) if input_latency else None),
        camera_capture_to_splat_visible_ms_median=(
            float(np.median(splat_latency)) if splat_latency else None
        ),
        fallback=error,
        messages=sorted(messages),
        captures=list(map(str, captures)),
    )
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(
        "PASS: RGB-only auto selection, live video/SLAM, input, e-stop, persisted override"
        + (
            ", budget eviction, worker stall/crash"
            if args.cuda
            else ", model fallback, zero external network I/O"
        )
    )


if __name__ == "__main__":
    main()

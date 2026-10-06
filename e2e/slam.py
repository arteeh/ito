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
from pathlib import Path

import numpy as np
import psutil
import pygame

from ito.app.__main__ import main as pilot_main


def key(code, down=None):
    if down is None:
        key(code, True)
        key(code, False)
    else:
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN if down else pygame.KEYUP, key=code))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="require real CUDA tracking and splats")
    parser.add_argument("--mode", choices=("desktop", "xr"), default="desktop")
    parser.add_argument("--startup-timeout", type=float, default=1800)
    args = parser.parse_args()
    if not args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    out = Path(
        "e2e/out/slam-"
        + ("cuda" if args.cuda else "no-cuda")
        + ("-xr" if args.mode == "xr" else "")
    )
    out.mkdir(parents=True, exist_ok=True)
    for capture in out.glob("capture-*.png"):
        capture.unlink()
    os.environ["XDG_CONFIG_HOME"] = os.environ["APPDATA"] = str(out / "config")
    if not args.cuda:
        os.environ["XDG_CACHE_HOME"] = os.environ["LOCALAPPDATA"] = str(out / "cache")
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
    counts = []
    messages = set()
    suspended = None
    first_position = None
    tracked_before_stall = 0
    video_before_failure = 0
    error = None

    def drive(app, window, value):
        nonlocal stage, changed, previous, suspended, first_position, tracked_before_stall
        nonlocal video_before_failure, error
        now = time.monotonic()
        limit = args.startup_timeout if args.cuda and stage == 0 else 60
        assert now - changed < limit, (stage, app.reconstruction_status, app.failure)
        display_times.append(now - previous)
        previous = now
        messages.add(app.reconstruction_status)
        counts.append(window.renderer.count)
        if app.state.video is not None and len(camera_frames) < 2 and stage in (0, 2):
            camera_frames.append(app.state.video.copy())
        t = app.telemetry
        if app.tracked_frames:
            poses.append(app.camera_pose.copy())
        if args.cuda and app.failure and stage < 5:
            raise AssertionError("Real CUDA SLAM failed: " + app.failure)
        ready = (
            app.tracked_frames >= 3 and window.renderer.count > 500 and not app.state.flat_video
            if args.cuda
            else app.failure and "CUDA" in app.failure
        )
        if stage == 0 and ready and app.state.video is not None and t:
            assert app.backend == "slam" and app.connections == 1
            if not args.cuda:
                assert app.state.flat_video and window.renderer.count == 0
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
                        assert (
                            app.backend == "video" and app.worker is None and app.state.flat_video
                        )
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
    if not args.cuda:
        assert not list((out / "cache").rglob("*.safetensors")), "No-CUDA path downloaded weights"
    report = dict(
        cuda=args.cuda,
        display_frame_ms_p95=frame_p95 * 1000,
        max_splats=max(counts),
        tracked_pose_samples=len(poses),
        fallback=error,
        messages=sorted(messages),
        captures=list(map(str, captures)),
    )
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(
        "PASS: RGB-only auto selection, live video/SLAM, input, e-stop, persisted override"
        + (", budget eviction, worker stall/crash" if args.cuda else ", no-CUDA fallback")
    )


if __name__ == "__main__":
    main()

"""CPU end-to-end: Pollen's MuJoCo body, robotd and mediad (built upstream checkouts).

uv run --extra microduck python e2e/microduck.py --microduck ../microduck \
    --rl ../microduck_rl --policies /path/to/policies/current
"""

import argparse
import json
import os
import tempfile
import time
from collections import deque
from pathlib import Path

import numpy as np
import pygame

from drivers.microduck.sim import port, simulation
from ito.app.__main__ import main as pilot_main
from ito.link.pairing import display

OUT = Path("e2e/out/microduck")


def key(code, down=None):
    if down is None:
        key(code, True)
        key(code, False)
    else:
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN if down else pygame.KEYUP, key=code))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microduck", type=Path, required=True)
    parser.add_argument("--rl", type=Path, required=True)
    parser.add_argument("--policies", type=Path, required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for capture in OUT.glob("capture-*.png"):
        capture.unlink()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    with tempfile.TemporaryDirectory(prefix="ito-duck-pilot-") as temporary:
        state = Path(temporary)
        os.environ["XDG_CONFIG_HOME"] = os.environ["APPDATA"] = str(state / "config")
        with simulation(
            args.microduck,
            args.rl,
            args.policies,
            logs=OUT,
            pairing_file=state / "pairing-code",
            host="127.0.0.1",
            driver_port=port(),
        ) as sim:
            return run(sim)


def run(sim):
    body, children, code, driver_port = sim.body, sim.children, sim.code, sim.port
    samples = deque()
    last_sample = 0.0
    physical = None
    stage = 0
    changed = time.monotonic()
    initial = None
    stopped = None
    report = {}
    telemetry = []
    video_times = set()

    def drive(app, window, value):
        nonlocal stage, changed, initial, stopped, last_sample, physical
        now = time.monotonic()
        for name, child in children:
            assert child.poll() is None, f"{name} exited: see {OUT / (name + '.log')}"
        assert now - changed < 45, (stage, app.state.status, app.telemetry)
        if now - last_sample < 0.1:
            return
        last_sample = now
        body.write('{"op":"read"}\n')
        body.flush()
        physical = json.loads(body.readline())
        position = np.array(physical["trunk"][:2])
        samples.append((now, position))
        while samples and now - samples[0][0] > 3.5:
            samples.popleft()
        t = app.telemetry
        if app.state.video is not None:
            video_times.add(app.state.video_time)
        if t:
            telemetry.append({"time": now, "stage": stage, "physical": physical, **t})
        if stage == 0 and len(video_times) > 15 and app.failure and t and now - changed > 8:
            assert app.backend == "slam" and app.state.flat_video
            assert "flat camera feed" in app.reconstruction_status
            assert app.state.video is not None and app.state.video.std() > 10
            assert t["healthy"] and t["camera_calibration"] == "sim"
            # Require three quiet seconds after standing up, measured in the actual world.
            if now - samples[0][0] < 3 or physical["trunk_z"] < 0.095:
                return
            idle_drift = max(float(np.linalg.norm(p - position)) for _, p in samples)
            if idle_drift > 0.01:
                return
            assert physical["imu"]["gravity"][2] < -0.9, physical
            assert "imu_gyro_0" in t and "imu_quat_0" in t
            initial = position.copy()
            report["idle_drift_m"] = idle_drift
            report.update(
                backend=app.backend,
                fallback=app.reconstruction_status,
                initial_head_yaw=t["head_yaw"],
            )
            key(pygame.K_F12)
            key(pygame.K_r)
            key(pygame.K_TAB)
            key(pygame.K_w, True)
            key(pygame.K_g, True)
            stage, changed = 1, now
        elif stage == 1 and now - changed > 1:
            report["initial_head_yaw"] = t["head_yaw"]
            pygame.event.post(
                pygame.event.Event(
                    pygame.MOUSEMOTION, pos=(600, 400), rel=(-280, -30), buttons=(0, 0, 0)
                )
            )
            stage, changed = 2, now
        elif stage == 2 and now - changed > 4:
            distance = float(np.linalg.norm(position - initial))
            # Forward motion must exceed the measured quiet baseline, without falling.
            assert physical["trunk_z"] > 0.095, physical
            assert physical["imu"]["gravity"][2] < -0.9, physical
            if position[0] - initial[0] <= 0.06:
                return
            assert t["applied_vx"] > 0.05, t
            assert abs(t["head_yaw"] + t["base_yaw"] - report["initial_head_yaw"]) > 0.2, t
            assert t["base_yaw"] > 0.15, t
            # Upstream's alpha MJCF has no mouth actuator; verify robotd's real target.
            assert abs(t["mouth_target"]) > 0.1, t
            report["mouth_target"] = t["mouth_target"]
            report["beak"] = "unverified: simulator has no beak actuator"
            report.update(distance_m=distance, head_yaw=t["head_yaw"], frames=len(video_times))
            key(pygame.K_e)  # Keep W held: the latch must override continuing pilot input.
            key(pygame.K_F12)
            stage, changed = 3, now
        elif stage == 3 and now - changed > 1:
            assert app.state.status.e_stop
            assert all(t[f"requested_{a}"] == 0 for a in ("vx", "vy", "vyaw")), t
            # robotd exponentially smooths applied velocity toward the zero intent.
            assert all(abs(t[f"applied_{a}"]) < 0.001 for a in ("vx", "vy", "vyaw")), t
            stopped = position.copy()
            stage, changed = 4, now
        elif stage == 4 and now - changed > 2:
            drift = float(np.linalg.norm(position - stopped))
            assert drift < 0.02 and app.state.status.e_stop, (drift, t)
            assert abs(t["applied_vx"]) < 0.001
            report.update(e_stop_drift_m=drift, frames_after_stop=len(video_times))
            key(pygame.K_w, False)
            key(pygame.K_F12)
            stage, changed = 5, now
        elif stage == 5 and now - changed > 0.2:
            pygame.event.post(pygame.event.Event(pygame.QUIT))

    try:
        result = pilot_main(
            [
                f"127.0.0.1:{driver_port}",
                "--code",
                code,
                "--size",
                "800",
                "600",
                "--fps",
                "60",
                "--capture-dir",
                str(OUT),
                "--metrics",
                str(OUT / "metrics.jsonl"),
            ],
            on_frame=drive,
        )
        assert result == 0 and stage == 5
    finally:
        (OUT / "telemetry.json").write_text(json.dumps(telemetry) + "\n")
    assert f"Pairing code: {display(code)}" in (OUT / "driver.log").read_text()
    report["paired"] = True
    first, last = telemetry[0], telemetry[-1]
    realtime = (last["physical"]["sim_time"] - first["physical"]["sim_time"]) / (
        last["time"] - first["time"]
    )
    assert 0.9 < realtime < 1.1, f"simulation must keep real time: {realtime:.2f}x"
    report["simulation_realtime"] = realtime
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("PASS: Microduck native simulation, mediad camera, pilot head/walking, e-stop latch")


if __name__ == "__main__":
    main()

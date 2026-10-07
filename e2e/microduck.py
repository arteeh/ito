"""CPU end-to-end: Pollen's MuJoCo body, robotd and mediad (built upstream checkouts).

uv run --extra microduck python e2e/microduck.py --microduck ../microduck \
    --rl ../microduck_rl --policies /path/to/policies/current
"""

import argparse
import contextlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import numpy as np
import pygame

from ito.app.__main__ import main as pilot_main
from ito.driver import pairing
from ito.link.pairing import display

OUT = Path("e2e/out/microduck")


def key(code, down=None):
    if down is None:
        key(code, True)
        key(code, False)
    else:
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN if down else pygame.KEYUP, key=code))


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


@contextlib.contextmanager
def process(name, command, env=None):
    with (OUT / f"{name}.log").open("w") as log:
        child = subprocess.Popen(list(map(str, command)), env=env, stdout=log, stderr=log)
        try:
            yield child
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microduck", type=Path, required=True)
    parser.add_argument("--rl", type=Path, required=True)
    parser.add_argument("--policies", type=Path, required=True)
    args = parser.parse_args()
    upstream, rl = args.microduck.resolve(), args.rl.resolve()
    binaries = upstream / "target/debug"
    OUT.mkdir(parents=True, exist_ok=True)
    for capture in OUT.glob("capture-*.png"):
        capture.unlink()
    body_port, camera_port, media_port, web_port, driver_port = [port() for _ in range(5)]
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sim_env = os.environ | {
        "PYTHONPATH": str(rl / "src"),
        "MUJOCO_GL": "osmesa",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    ort = list(
        (rl / ".venv/lib").glob("python*/site-packages/onnxruntime/capi/libonnxruntime.so.*")
    )
    assert ort, "Install ONNX Runtime in the upstream simulator's .venv"
    media_env = os.environ.copy()
    if "MEDIAD_LD_LIBRARY_PATH" in os.environ:
        media_env["LD_LIBRARY_PATH"] = os.environ["MEDIAD_LD_LIBRARY_PATH"]
    with (
        tempfile.TemporaryDirectory(prefix="ito-duck-") as temporary,
        contextlib.ExitStack() as stack,
    ):
        state = Path(temporary)
        code_file = state / "pairing-code"
        show = subprocess.run(
            [
                sys.executable,
                "-m",
                "drivers.microduck.cli",
                "--pairing-file",
                str(code_file),
                "--show-code",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        code = pairing.read(code_file)
        assert f"Pairing code: {display(code)}" in show.stdout
        robot_socket = state / "robot.sock"
        media_config = state / "media.toml"
        media_config.write_text('[media]\nquality = "360p30"\n')
        os.environ["XDG_CONFIG_HOME"] = os.environ["APPDATA"] = str(state / "config")
        # Match robotd's shipped default: velstand walks and also stands still at zero command.
        policies = {
            "walk": "velstand",
            "sitstand": "alpha_sitstand",
            "ground_pick": "alpha_ground_pick",
            "kick_left": "ball_kick_left",
            "kick_right": "ball_kick_right",
            "roulade": "roulade",
        }
        params = state / "robotd.toml"
        lines = ["[policy]", "enabled = true"]
        for name, filename in policies.items():
            path = args.policies.resolve() / f"{filename}.onnx"
            assert path.is_file(), path
            lines.append(f"{name} = {json.dumps(str(path))}")
        params.write_text("\n".join(lines) + "\n")
        # Software shadows must not slow physics below the policy's 50 Hz control clock.
        source_scene = rl / "src/mjlab_microduck/robot/microduck/scene.xml"
        for asset in source_scene.parent.iterdir():
            if asset.name not in {"scene.xml", "robot_groundcontact.xml"}:
                (state / asset.name).symlink_to(asset, target_is_directory=asset.is_dir())
        # Hide CAD surfaces from the head camera; keep every inertial, collision and actuator.
        robot_model = ET.parse(source_scene.parent / "robot_groundcontact.xml")
        for geom in robot_model.findall(".//worldbody//geom"):
            geom.attrib.pop("material", None)
            geom.set("rgba", "0 0 0 0")
        robot_model.write(state / "robot_groundcontact.xml")
        scene = ET.parse(source_scene)
        ET.SubElement(scene.find("visual"), "quality", shadowsize="256", offsamples="1")
        scene_path = state / "scene.xml"
        scene.write(scene_path)
        children = []

        def launch(name, command, env=None):
            child = stack.enter_context(process(name, command, env))
            children.append((name, child))
            return child

        launch(
            "body",
            [
                rl / ".venv/bin/python",
                "-m",
                "mjlab_microduck.sim.body_server",
                "--headless",
                "--cameras",
                "a",
                "--port",
                body_port,
                "--frame-port",
                camera_port,
                "--scene",
                scene_path,
            ],
            sim_env,
        )
        deadline = time.monotonic() + 30
        while True:
            try:
                body_socket = stack.enter_context(
                    socket.create_connection(("127.0.0.1", int(body_port)), timeout=2)
                )
                break
            except OSError:
                assert children[0][1].poll() is None, "body exited; see body.log"
                assert time.monotonic() < deadline, "body did not listen; see body.log"
                time.sleep(0.2)
        body = stack.enter_context(body_socket.makefile("rw"))
        body.write('{"op":"hello","protocol":1,"joints":15}\n')
        body.flush()
        assert json.loads(body.readline())["protocol"] == 1
        launch(
            "robot",
            [
                binaries / "robotd",
                "--sim",
                f"127.0.0.1:{body_port}",
                "--socket",
                robot_socket,
                "--params",
                params,
            ],
            os.environ | {"ORT_DYLIB_PATH": str(ort[0])},
        )
        deadline = time.monotonic() + 20
        ctl = [str(binaries / "robotctl"), "--robot-socket", str(robot_socket)]
        while True:
            result = subprocess.run(ctl + ["robot", "enable"], capture_output=True, timeout=3)
            if result.returncode == 0:
                break
            assert time.monotonic() < deadline, result.stderr.decode()
            time.sleep(0.2)
        launch(
            "media",
            [
                binaries / "mediad",
                "--no-remote",
                "--host",
                "127.0.0.1",
                "--port",
                media_port,
                "--web-port",
                web_port,
                "--robot-socket",
                robot_socket,
                "--tof-socket",
                state / "absent-tof.sock",
                "--frame-socket",
                state / "media.sock",
                "--config",
                media_config,
                "--sim-camera",
                f"127.0.0.1:{camera_port}",
            ],
            media_env,
        )
        # Wait for mediad before the driver's deliberately bounded connection attempt.
        deadline = time.monotonic() + 20
        while True:
            try:
                with socket.create_connection(("127.0.0.1", int(media_port)), timeout=1):
                    break
            except OSError:
                assert time.monotonic() < deadline, "mediad did not listen; see media.log"
                time.sleep(0.2)
        launch(
            "driver",
            [
                sys.executable,
                "-m",
                "drivers.microduck.cli",
                "--host",
                "127.0.0.1",
                "--port",
                driver_port,
                "--pairing-file",
                code_file,
                "--robot",
                f"ws://127.0.0.1:{media_port}",
            ],
        )
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

"""The virtual Microduck's robot side, as compose.yaml runs it: Pollen's robotd, tofd and mediad
(built at the third_party/ pins) and Ito's Microduck driver, against the MuJoCo body that
drivers.microduck.sim runs on the host. Logs go to /logs, the pairing code to /config."""

import contextlib
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from drivers.microduck import lens

BODY_PORT, FRAME_PORT = 7801, 7901  # the body's bus and head camera (sim.py)
MEDIA_PORT, WEB_PORT = 8443, 8444
DRIVER_PORT = 8081
BIN = Path("/opt/microduck/bin")
POLICIES = Path("/opt/microduck/policies/current")
STATE = Path("/run/microduck")
LOGS = Path("/logs")
PAIRING = Path("/config/microduck-sim-code")
WALK = {
    "walk": "velstand",
    "sitstand": "alpha_sitstand",
    "ground_pick": "alpha_ground_pick",
    "kick_left": "ball_kick_left",
    "kick_right": "ball_kick_right",
    "roulade": "roulade",
}


@contextlib.contextmanager
def process(logs, name, command, env=None):
    with (logs / f"{name}.log").open("w") as log:
        child = subprocess.Popen(
            list(map(str, command)), env=env, stdout=log, stderr=log, start_new_session=True
        )
        try:
            yield child
        finally:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(child.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    child.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    continue
            child.wait()


def check(children, logs):
    for name, child in children:
        if child.poll() is not None:
            raise RuntimeError(f"{name} exited ({child.returncode}); see {logs / (name + '.log')}")


def wait(what, ready, children, timeout):
    deadline = time.monotonic() + timeout
    while not ready():
        check(children, LOGS)
        if time.monotonic() >= deadline:
            raise RuntimeError(f"{what} timed out after {timeout:.0f} s")
        time.sleep(0.2)


def listening(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def stepping():
    """The body is stepping once its head camera sends a frame: it answers the bus seconds
    before that, while it builds the camera, and a robotd enabled then walks against a frozen
    duck that is flung over when physics starts."""
    try:
        with socket.create_connection(("127.0.0.1", FRAME_PORT), timeout=1) as camera:
            camera.settimeout(5)
            return len(camera.recv(16)) > 0
    except OSError:
        return False


def configure():
    STATE.mkdir(parents=True, exist_ok=True)
    robot = ["[board]", 'version = "beta"', "[policy]", "enabled = true"]
    robot += [f'{slot} = "{POLICIES / name}.onnx"' for slot, name in WALK.items()]
    robot += ["[head_imu]", "enabled = true"]
    (STATE / "robotd.toml").write_text("\n".join(robot) + "\n")
    # A sim mediad publishes a plain pinhole unless told the lens: the one the body renders
    # (lens.py), which is the family calibration a real beta's mediad publishes.
    media = ["[board]", 'version = "beta"', "[media]"]
    media += [f'quality = "{lens.HEIGHT}p30"', "[media.intrinsics]"]
    media += [f"width = {lens.WIDTH}", f"height = {lens.HEIGHT}"]
    media += [f"{k} = {getattr(lens, k.upper())!r}" for k in ("fx", "fy", "cx", "cy")]
    media += [f"distortion = [{', '.join(map(repr, lens.DISTORTION))}]"]
    (STATE / "media.toml").write_text("\n".join(media) + "\n")


def main():
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    configure()
    body = f"127.0.0.1:{BODY_PORT}"
    robot_socket, tof_socket = STATE / "robot.sock", STATE / "tof.sock"
    ort = next(
        Path(sys.prefix).glob("lib/python*/site-packages/onnxruntime/capi/libonnxruntime.so*")
    )
    children = []
    with contextlib.ExitStack() as stack:

        def launch(name, command, env=None):
            children.append((name, stack.enter_context(process(LOGS, name, command, env))))

        launch(
            "robot",
            [
                BIN / "robotd",
                "--sim",
                body,
                "--socket",
                robot_socket,
                "--params",
                STATE / "robotd.toml",
            ],
            os.environ | {"ORT_DYLIB_PATH": str(ort)},
        )
        launch(
            "tof",
            [
                BIN / "tofd",
                "--sim",
                body,
                "--socket",
                tof_socket,
                "--config",
                STATE / "robotd.toml",
            ],
        )
        launch(
            "media",
            [
                BIN / "mediad",
                "--no-remote",
                "--host",
                "127.0.0.1",
                "--port",
                MEDIA_PORT,
                "--web-port",
                WEB_PORT,
                "--robot-socket",
                robot_socket,
                "--tof-socket",
                tof_socket,
                "--frame-socket",
                STATE / "media.sock",
                "--config",
                STATE / "media.toml",
                "--sim-camera",
                f"127.0.0.1:{FRAME_PORT}",
            ],
            # Pollen's hardware encoder presets are not portable to every desktop NVENC driver.
            os.environ | {"GST_PLUGIN_FEATURE_RANK": "x264enc:1024"},
        )
        wait("mediad", lambda: listening(MEDIA_PORT), children, 20)
        print("waiting for the body (python -m drivers.microduck.sim on the host)", flush=True)
        wait("the body", stepping, children, float("inf"))
        enable = [BIN / "robotctl", "--robot-socket", robot_socket, "robot", "enable"]
        wait(
            "robot enable",
            lambda: subprocess.run(enable, capture_output=True).returncode == 0,
            children,
            20,
        )
        launch(
            "driver",
            [
                sys.executable,
                "-m",
                "drivers.microduck.cli",
                "--port",
                DRIVER_PORT,
                "--pairing-file",
                PAIRING,
                "--robot",
                f"ws://127.0.0.1:{MEDIA_PORT}",
            ],
        )
        wait("the driver", lambda: listening(DRIVER_PORT), children, 30)
        print(f"Virtual Microduck ready on port {DRIVER_PORT}", flush=True)
        while True:
            check(children, LOGS)
            time.sleep(0.5)


if __name__ == "__main__":
    main()

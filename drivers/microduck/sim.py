"""Run Pollen's real body, policy daemon and WebRTC camera as one virtual Microduck."""

import argparse
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

from drivers.microduck.scene import furnish
from ito.driver import pairing
from ito.link.pairing import display


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


@contextlib.contextmanager
def process(logs, name, command, env=None):
    with (logs / f"{name}.log").open("w") as log:
        child = subprocess.Popen(
            list(map(str, command)), env=env, stdout=log, stderr=log, start_new_session=True
        )
        try:
            yield child
        finally:
            # Kill the whole owned group, including any grandchildren, on every exit path.
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


@contextlib.contextmanager
def simulation(
    upstream,
    rl,
    policies_dir,
    *,
    logs,
    pairing_file,
    host="0.0.0.0",
    driver_port=8081,
    viewer=False,
    gl=None,
):
    upstream, rl, policies_dir = (p.resolve() for p in (upstream, rl, policies_dir))
    logs.mkdir(parents=True, exist_ok=True)
    binaries = upstream / "target/debug"
    body_port, camera_port, media_port, web_port = [port() for _ in range(4)]
    sim_env = os.environ | {
        "PYTHONPATH": os.pathsep.join([str(Path(__file__).resolve().parents[2]), str(rl / "src")]),
        "MUJOCO_GL": gl or ("glfw" if viewer else "osmesa"),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    if Path("/dev/dxg").exists() and "GALLIUM_DRIVER" not in os.environ:
        # WSL's Mesa defaults to llvmpipe; d3d12 renders the viewer and head camera on the GPU.
        sim_env["GALLIUM_DRIVER"] = "d3d12"
    ort = list(
        (rl / ".venv/lib").glob("python*/site-packages/onnxruntime/capi/libonnxruntime.so.*")
    )
    if not ort:
        raise FileNotFoundError("Install ONNX Runtime in the simulator's .venv")
    media_env = os.environ.copy()
    # Pollen's hardware encoder presets are not portable to every desktop NVENC driver.
    media_env.setdefault("GST_PLUGIN_FEATURE_RANK", "x264enc:1024")
    if "MEDIAD_LD_LIBRARY_PATH" in os.environ:
        media_env["LD_LIBRARY_PATH"] = os.environ["MEDIAD_LD_LIBRARY_PATH"]
    with (
        tempfile.TemporaryDirectory(prefix="ito-duck-") as temporary,
        contextlib.ExitStack() as stack,
    ):
        state = Path(temporary)
        pairing_file = pairing_file.resolve()
        subprocess.run(
            [
                sys.executable,
                "-m",
                "drivers.microduck.cli",
                "--pairing-file",
                str(pairing_file),
                "--show-code",
            ],
            check=True,
            capture_output=True,
            timeout=10,
        )
        code = pairing.read(pairing_file)
        robot_socket = state / "robot.sock"
        media_config = state / "media.toml"
        media_config.write_text('[media]\nquality = "360p30"\n')
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
            path = policies_dir / f"{filename}.onnx"
            if not path.is_file():
                raise FileNotFoundError(path)
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
            geom.set("group", "2" if geom.get("class") == "visual" else "3")
        robot_model.write(state / "robot_groundcontact.xml")
        scene = furnish(source_scene)
        ET.SubElement(scene.find("visual"), "quality", shadowsize="256", offsamples="1")
        scene_path = state / "scene.xml"
        scene.write(scene_path)
        children = []

        def launch(name, command, env=None):
            child = stack.enter_context(process(logs, name, command, env))
            children.append((name, child))
            return child

        launch(
            "body",
            [
                rl / ".venv/bin/python",
                "-m",
                "drivers.microduck.sim_body",
                *([] if viewer else ["--headless"]),
                "--cameras",
                "a",
                "--host",
                "127.0.0.1",
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
                check(children, logs)
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"body did not listen; see {logs / 'body.log'}") from None
                time.sleep(0.2)
        body = stack.enter_context(body_socket.makefile("rw"))
        body.write('{"op":"hello","protocol":1,"joints":15}\n')
        body.flush()
        if json.loads(body.readline())["protocol"] != 1:
            raise RuntimeError("incompatible Pollen body protocol")
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
            check(children, logs)
            if time.monotonic() >= deadline:
                raise RuntimeError(f"robot enable failed: {result.stderr.decode()}")
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
                check(children, logs)
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"mediad did not listen; see {logs / 'media.log'}") from None
                time.sleep(0.2)
        launch(
            "driver",
            [
                sys.executable,
                "-m",
                "drivers.microduck.cli",
                "--host",
                host,
                "--port",
                driver_port,
                "--pairing-file",
                pairing_file,
                "--robot",
                f"ws://127.0.0.1:{media_port}",
            ],
        )
        deadline = time.monotonic() + 30
        while True:
            check(children, logs)
            try:
                with socket.create_connection(("127.0.0.1", int(driver_port)), timeout=1):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"driver did not listen; see {logs / 'driver.log'}"
                    ) from None
                time.sleep(0.2)
        yield SimpleNamespace(body=body, children=children, code=code, port=driver_port)


def arguments(parser):
    parser.add_argument("--microduck", type=Path, required=True)
    parser.add_argument("--rl", type=Path, required=True)
    parser.add_argument("--policies", type=Path, required=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    arguments(parser)
    parser.add_argument("--viewer", action="store_true", help="open Pollen's MuJoCo viewer")
    parser.add_argument("--gl", choices=("osmesa", "egl", "glfw"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--logs", type=Path, default=Path.home() / ".local/state/ito/microduck-sim")
    parser.add_argument(
        "--pairing-file", type=Path, default=Path.home() / ".config/ito/microduck-sim-code"
    )
    args = parser.parse_args()
    if args.viewer and args.gl not in (None, "glfw"):
        parser.error("--viewer requires --gl glfw")

    def stop(signum, frame):
        raise KeyboardInterrupt

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    try:
        with simulation(
            args.microduck,
            args.rl,
            args.policies,
            logs=args.logs,
            pairing_file=args.pairing_file,
            host=args.host,
            driver_port=args.port,
            viewer=args.viewer,
            gl=args.gl,
        ) as sim:
            addresses = [args.host]
            if args.host == "0.0.0.0":
                addresses = subprocess.check_output(["hostname", "-I"], text=True).split()
            print("Virtual Microduck ready. In Ito, connect to:", flush=True)
            for address in addresses:
                if ":" not in address:
                    print(f"  {address}:{args.port}", flush=True)
            print(
                f"Pairing code: {display(sim.code)}\nLogs: {args.logs}\nCtrl+C stops everything.",
                flush=True,
            )
            while True:
                # Closing the viewer also stops the complete robot.
                if args.viewer and sim.children[0][1].poll() == 0:
                    break
                check(sim.children, args.logs)
                time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"microduck-sim: {exc}\n")


if __name__ == "__main__":
    main()

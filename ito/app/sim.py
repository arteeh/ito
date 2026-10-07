"""The bundled MuJoCo robot, so a pilot can try Ito without hardware."""

import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

from ito.desktop.settings import settings_path
from ito.driver import pairing


class SimulatedRobot:
    def __init__(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self.address = f"127.0.0.1:{port}"
        self.log = settings_path().parent / "simulated-robot.log"
        self.log.parent.mkdir(parents=True, exist_ok=True)
        # Each launched robot has its own code, including concurrent app instances.
        self._directory = tempfile.TemporaryDirectory(prefix="ito-sim-")
        self.viewer = None
        self.retired_viewers = []
        self.viewer_error = None
        self.viewer_state = Path(self._directory.name) / "state.npy"
        code_file = Path(self._directory.name) / "pairing-code"
        self.code = pairing.rotate(code_file)
        with self.log.open("w") as log:
            self.process = subprocess.Popen(
                [sys.executable, *(["-I"] if sys.flags.isolated else [])]
                + ["-m", "drivers.mujoco.cli", "--host", "127.0.0.1", "--port", str(port)]
                + ["--pairing-file", str(code_file), "--viewer-state", str(self.viewer_state)]
                # It runs on the pilot's own PC: a robot microphone would hear the pilot's
                # room and its speaker would play the pilot back to themselves.
                + ["--audio-source", "none", "--audio-sink", "none"],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )

    @property
    def visible(self):
        if self.viewer and self.viewer.poll() is not None:
            if self.viewer.returncode:
                self.viewer_error = f"Simulation viewer failed; see {self.log}"
            self.viewer = None
        return self.viewer is not None

    def show(self, visible):
        if visible == self.visible:
            return
        if visible:
            from drivers.mujoco.adapter import ROOM

            self.viewer_error = None
            self.viewer_state.with_suffix(".viewer").unlink(missing_ok=True)
            with self.log.open("a") as log:
                self.viewer = subprocess.Popen(
                    [sys.executable, *(["-I"] if sys.flags.isolated else [])]
                    + ["-m", "drivers.mujoco.viewer", str(ROOM), str(self.viewer_state)],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
        else:
            self.viewer.terminate()
            self.retired_viewers.append(self.viewer)
            self.viewer = None

    def failure(self):
        self.retired_viewers = [p for p in self.retired_viewers if p.poll() is None]
        code = self.process.poll()
        return None if code is None else f"Simulated robot stopped ({code}); see {self.log}"

    def close(self):
        self.show(False)
        for viewer in self.retired_viewers:
            try:
                viewer.wait(3)
            except subprocess.TimeoutExpired:
                viewer.kill()
                viewer.wait(3)
        self.retired_viewers.clear()
        self.process.terminate()
        try:
            self.process.wait(5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(5)
        self._directory.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

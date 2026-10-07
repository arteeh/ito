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
        code_file = Path(self._directory.name) / "pairing-code"
        self.code = pairing.rotate(code_file)
        with self.log.open("w") as log:
            self.process = subprocess.Popen(
                [sys.executable, *(["-I"] if sys.flags.isolated else [])]
                + ["-m", "drivers.mujoco.cli", "--host", "127.0.0.1", "--port", str(port)]
                + ["--pairing-file", str(code_file)]
                # It runs on the pilot's own PC: a robot microphone would hear the pilot's
                # room and its speaker would play the pilot back to themselves.
                + ["--audio-source", "none", "--audio-sink", "none"],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )

    def failure(self):
        code = self.process.poll()
        return None if code is None else f"Simulated robot stopped ({code}); see {self.log}"

    def close(self):
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

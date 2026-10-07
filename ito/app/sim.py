"""The bundled MuJoCo robot, so a pilot can try Ito without hardware."""

import os
import socket
import subprocess
import sys

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
        # A fresh code per run pairs this pilot without a prompt and nothing else on the PC.
        code_file = self.log.with_name("simulated-robot.code")
        self.code = pairing.rotate(code_file)
        with self.log.open("w") as log:
            self.process = subprocess.Popen(
                [sys.executable, *(["-I"] if sys.flags.isolated else [])]
                + ["-m", "drivers.mujoco.cli", "--host", "127.0.0.1", "--port", str(port)]
                + ["--pairing-file", str(code_file)],
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

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

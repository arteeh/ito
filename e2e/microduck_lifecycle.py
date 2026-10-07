"""Run the real simulator CLI; interrupt it and kill its body to verify complete cleanup.

uv run --extra microduck python e2e/microduck_lifecycle.py --microduck ... --rl ... --policies ...
"""

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

from drivers.microduck.sim import arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    arguments(parser)
    args = parser.parse_args()
    out = Path("e2e/out/microduck-lifecycle")
    out.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "drivers.microduck.sim",
        "--microduck",
        str(args.microduck),
        "--rl",
        str(args.rl),
        "--policies",
        str(args.policies),
        "--logs",
        str(out),
    ]
    for mode in ("interrupt", "body-failure"):
        log_path = out / f"{mode}.log"
        with log_path.open("w") as log:
            child = subprocess.Popen(command, stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 30
                while "Virtual Microduck ready" not in log_path.read_text():
                    assert child.poll() is None, log_path.read_text()
                    assert time.monotonic() < deadline, log_path.read_text()
                    time.sleep(0.1)
                descendants = psutil.Process(child.pid).children(recursive=True)
                if mode == "interrupt":
                    child.send_signal(signal.SIGINT)
                else:
                    body = next(
                        p for p in descendants if "drivers.microduck.sim_body" in p.cmdline()
                    )
                    body.kill()
                assert child.wait(timeout=12) == (0 if mode == "interrupt" else 1)
                assert not [p.pid for p in descendants if p.is_running()]
                print(f"PASS: {mode}; {len(descendants)} children reaped")
            finally:
                if child.poll() is None:
                    child.send_signal(signal.SIGINT)
                    child.wait(timeout=12)


if __name__ == "__main__":
    main()

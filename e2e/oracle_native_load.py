"""Run the real online mapper under a Windows GL deadline countercheck.

The Windows parent holds ceres.lock. Publication is drained here; Windows draws the
saved mature map with controlled uploads. This is a frozen-renderer load experiment.
"""

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

from oracle_run import worker

from ito.reconstruction.admission import Admission


class Flag:
    def __init__(self, path):
        self.path = path

    @property
    def value(self):
        return self.path.exists()

    def set(self):
        self.path.touch()


class Start:
    def __init__(self, path):
        self.path = path

    @property
    def value(self):
        try:
            return time.perf_counter() - (time.time() - float(self.path.read_text()))
        except (OSError, ValueError):
            return 0


class Drain:
    batch = 4096

    def publish(self, *args):
        return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("recording", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--arm", choices=("TT", "TE", "ET", "EE", "TF"), default="TT")
    p.add_argument("--precision", choices=("full", "half"), default="full")
    args = p.parse_args()
    (args.out / "pid").write_text(str(os.getpid()))
    args.estimates = args.recording / "causal-full"
    args.max_splats, args.oom_at = 1_000_000, -1
    ctx = mp.get_context("spawn")
    args.oom_warning = ctx.RawValue("d", -1)
    stop = Flag(args.out / "stop")

    class Feedback(Admission):
        def available(self):
            try:
                until = float((args.out / "pause").read_text())
                self.pause_until.value = time.perf_counter() + max(0, until - time.time())
            except (OSError, ValueError):
                pass
            return super().available()

    status = ctx.Queue()
    # Stop at the integration boundary so the final audit report is always written.
    worker(
        args,
        Drain(),
        Feedback(ctx, ctx.RawValue("b", False)),
        Flag(args.out / "ready"),
        Start(args.out / "start"),
        stop,
        status,
    )
    if not status.empty():
        raise RuntimeError(status.get())


if __name__ == "__main__":
    main()

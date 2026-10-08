"""Processes the pilot app starts stop with it, even when it is killed.

Windows releases put them in the launcher's job object. On Linux the kernel signals a
child when the thread that started it exits; the app passes its pid so a child that
starts after the app already died exits too.
"""

import os
import signal
import sys

PARENT = "ITO_PARENT_PID"


def child_environment() -> dict[str, str]:
    return os.environ | {PARENT: str(os.getpid())}


def follow_parent() -> None:
    """Call first thing in a child process; a no-op unless the pilot app started it."""
    parent = os.environ.pop(PARENT, None)
    if parent is None or not sys.platform.startswith("linux"):
        return
    import ctypes

    PR_SET_PDEATHSIG = 1
    ctypes.CDLL(None).prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    if os.getppid() != int(parent):
        raise SystemExit("the Ito pilot app that started this process has stopped")

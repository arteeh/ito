"""The one clock for wire, input and render timestamps, on both ends of the link.

perf_counter is QueryPerformanceCounter on Windows, which is system-wide and fine-grained,
and CLOCK_MONOTONIC on Linux. Python 3.12's Windows time.monotonic ticks every 15.6 ms.
"""

import contextlib
import sys
from time import perf_counter as now
from time import perf_counter_ns as now_ns

__all__ = ["fine_timers", "now", "now_ns"]


@contextlib.contextmanager
def fine_timers():
    """Millisecond sleeps and event-loop timeouts while inside, on every platform.

    Windows wakes sleeping threads on its 15.6 ms timer tick unless a process asks for
    finer: an asyncio loop pacing pilot poses at 90 Hz then sends at 64. SDL asks for 1 ms
    once a window is up; a link that runs without one must ask itself.
    """
    if sys.platform != "win32":
        yield
        return
    import ctypes

    winmm = ctypes.WinDLL("winmm")
    fine = winmm.timeBeginPeriod(1) == 0  # TIMERR_NOERROR
    try:
        yield
    finally:
        if fine:
            winmm.timeEndPeriod(1)

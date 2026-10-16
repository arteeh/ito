"""The one clock for wire, input and render timestamps, on both ends of the link.

perf_counter is QueryPerformanceCounter on Windows, which is system-wide and fine-grained,
and CLOCK_MONOTONIC on Linux. Python 3.12's Windows time.monotonic ticks every 15.6 ms.
"""

import asyncio
import contextlib
import logging
import sys
from time import perf_counter as now
from time import perf_counter_ns as now_ns

__all__ = ["display_priority", "fine_timers", "now", "now_ns", "sleep_until"]


@contextlib.contextmanager
def fine_timers():
    """Millisecond sleeps and event-loop timeouts while inside, on every platform.

    Windows wakes sleeping threads on its 15.6 ms timer tick unless a process asks for
    finer: an asyncio loop pacing pilot poses at 90 Hz then sends at 64. SDL asks for 1 ms
    once a window is up; a link that runs without one must ask itself. Windows 11 also stops
    honouring the request while a process's windows are occluded, minimized or on no visible
    display, unless the process opts out of that power throttling: a pilot looking at a
    headset, not the desktop window, still drives the robot.
    """
    if sys.platform != "win32":
        yield
        return
    import ctypes

    class Throttling(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulong) for name in ("version", "control", "state")]

    # PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION under control, state off: always honour.
    # Older Windows has no such class and fails the call, harmlessly.
    kernel32 = ctypes.WinDLL("kernel32")
    throttling = Throttling(1, 0x4, 0)
    kernel32.SetProcessInformation(
        ctypes.c_void_p(kernel32.GetCurrentProcess()),
        4,  # ProcessPowerThrottling
        ctypes.byref(throttling),
        ctypes.sizeof(throttling),
    )
    winmm = ctypes.WinDLL("winmm")
    fine = winmm.timeBeginPeriod(1) == 0  # TIMERR_NOERROR
    try:
        yield
    finally:
        if fine:
            winmm.timeEndPeriod(1)


@contextlib.contextmanager
def display_priority():
    if sys.platform != "win32":
        yield "default"
        return
    import ctypes as c

    avrt = c.WinDLL("avrt", use_last_error=True)
    avrt.AvSetMmThreadCharacteristicsW.argtypes = (c.c_wchar_p, c.POINTER(c.c_uint32))
    avrt.AvSetMmThreadCharacteristicsW.restype = c.c_void_p
    avrt.AvSetMmThreadPriority.argtypes = (c.c_void_p, c.c_int)
    avrt.AvRevertMmThreadCharacteristics.argtypes = (c.c_void_p,)
    task = c.c_uint32()
    handle = avrt.AvSetMmThreadCharacteristicsW("Games", c.byref(task))
    active = handle and avrt.AvSetMmThreadPriority(handle, 1)  # AVRT_PRIORITY_HIGH
    if not active:
        logging.getLogger(__name__).warning(
            "Display MMCSS scheduling unavailable: %s", c.get_last_error()
        )
    try:
        with fine_timers():
            yield "MMCSS Games/high" if active else "default"
    finally:
        if handle:
            avrt.AvRevertMmThreadCharacteristics(handle)


async def sleep_until(deadline):
    """Sleep until perf_counter reaches deadline, never early.

    Windows asyncio counts a timer as due once it is within the loop's clock resolution
    (15.6 ms), even with fine_timers on, so a sleep woken by network traffic can return
    that much early: a 90 Hz pose sender then sends in pairs and the driver applies 66/s.
    """
    while (left := deadline - now()) > 0:  # noqa: ASYNC110 (re-checks a clock, not a flag)
        await asyncio.sleep(left)

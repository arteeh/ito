"""The one clock for wire, input and render timestamps, on both ends of the link.

perf_counter is QueryPerformanceCounter on Windows, which is system-wide and fine-grained,
and CLOCK_MONOTONIC on Linux. Python 3.12's Windows time.monotonic ticks every 15.6 ms.
"""

from time import perf_counter as now
from time import perf_counter_ns as now_ns

__all__ = ["now", "now_ns"]

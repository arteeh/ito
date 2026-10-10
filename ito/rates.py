"""Cheap rolling event rates for the pilot's readouts: a tick is one deque append."""

from collections import deque

from ito import clock


class Rate:
    """Events per second over the last `window` seconds (up to `cap` events kept)."""

    def __init__(self, window=1.0, cap=256):
        self.window = window
        self.times = deque(maxlen=cap)

    def tick(self, now=None):
        self.times.append(clock.now() if now is None else now)

    def hz(self, now=None):
        now = clock.now() if now is None else now
        # A snapshot: ticks may come from another thread while this counts.
        times = tuple(self.times)
        if len(times) == self.times.maxlen and now - times[0] < self.window:
            # Saturated: estimate from the span the kept events cover.
            return (len(times) - 1) / max(times[-1] - times[0], 1e-6)
        return sum(1 for t in times if now - t <= self.window) / self.window


class CounterRate:
    """Rate of a monotonic counter another process advances, from periodic samples."""

    def __init__(self, window=1.0):
        self.window = window
        self.samples = deque()

    def hz(self, value, now=None):
        now = clock.now() if now is None else now
        samples = self.samples
        if samples and value < samples[-1][1]:
            samples.clear()  # a new worker restarted the count
        samples.append((now, value))
        while len(samples) > 2 and now - samples[1][0] >= self.window:
            samples.popleft()
        (t0, v0), (t1, v1) = samples[0], samples[-1]
        return (v1 - v0) / (t1 - t0) if t1 - t0 >= 0.25 else None

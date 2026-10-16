from collections import deque

from ito import clock


class Rate:
    def __init__(self, window=1.0, cap=256):
        self.window = window
        self.times = deque(maxlen=cap)

    def tick(self, now=None):
        self.times.append(clock.now() if now is None else now)

    def hz(self, now=None):
        now = clock.now() if now is None else now
        times = tuple(self.times)
        if len(times) == self.times.maxlen and now - times[0] < self.window:
            return (len(times) - 1) / max(times[-1] - times[0], 1e-6)
        return sum(1 for t in times if now - t <= self.window) / self.window


class CounterRate:
    def __init__(self, window=1.0):
        self.window = window
        self.samples = deque()

    def hz(self, value, now=None):
        now = clock.now() if now is None else now
        samples = self.samples
        if samples and value < samples[-1][1]:
            samples.clear()
        samples.append((now, value))
        while len(samples) > 2 and now - samples[1][0] >= self.window:
            samples.popleft()
        (t0, v0), (t1, v1) = samples[0], samples[-1]
        return (v1 - v0) / (t1 - t0) if t1 - t0 >= 0.25 else None

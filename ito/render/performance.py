from collections import deque

import numpy as np

from ito import clock
from ito.rates import Rate

WINDOW_SECONDS = 10.0


class Distribution:
    def __init__(self):
        self.recent = deque(maxlen=4096)

    def add(self, now, value):
        self.recent.append((now, max(0.0, value)))

    def percentiles(self, now, levels=(50, 95, 99)):
        while self.recent and now - self.recent[0][0] > WINDOW_SECONDS:
            self.recent.popleft()
        values = [value for _, value in self.recent]
        return [float(v) for v in np.percentile(values, levels)] if values else [None] * len(levels)


class Performance:
    """The display's timing and the scene's capture-to-visible, for the HUD and diagnostics."""

    def __init__(self):
        self.visible_rate = Rate(window=5, cap=1024)
        self.latency = Distribution()
        self.render = Distribution()
        self.service = Distribution()
        self.frames = self.misses = self.consecutive = self.longest_misses = 0
        self.last_completion_misses = 0
        self.seen = set()
        self.seen_order = deque()
        self.latest_capture = None
        self.latest_latency = None
        self.last = {}
        self.next_snapshot = 0.0

    def completed(self, began, due, period, updates, *, scene=True, statistics=None, gpu_busy=None):
        now = clock.now()
        self.frames += 1
        self.render.add(now, (now - due) * 1000)
        self.service.add(now, (now - began) * 1000)
        missed = max(0, int(np.ceil((now - due) / period)) - 1)
        self.last_completion_misses = missed
        self.misses += missed
        self.consecutive = self.consecutive + missed if missed else 0
        self.longest_misses = max(self.longest_misses, self.consecutive)
        if scene:
            for update in updates:
                for sequence, captured in update.observations:
                    key = (update.epoch, sequence)
                    if key in self.seen:
                        continue
                    self.seen.add(key)
                    self.seen_order.append(key)
                    if len(self.seen_order) > 4096:
                        self.seen.discard(self.seen_order.popleft())
                    latency = max(0.0, (now - captured) * 1000)
                    self.latency.add(now, latency)
                    self.visible_rate.tick(now)
                    if self.latest_capture is None or captured > self.latest_capture:
                        self.latest_capture, self.latest_latency = captured, latency
        if now < self.next_snapshot:
            return self.last
        self.next_snapshot = now + 0.5
        p50, p95, _ = self.latency.percentiles(now)
        _, _, p99 = self.render.percentiles(now)
        _, _, service_p99 = self.service.percentiles(now)
        self.last = dict(
            **(statistics or {}),
            scene_hz=self.visible_rate.hz(now),
            capture_to_visible_ms=self.latest_latency,
            capture_to_visible_p50_ms=p50,
            capture_to_visible_p95_ms=p95,
            scene_age_ms=None
            if self.latest_capture is None
            else (now - self.latest_capture) * 1000,
            gpu_busy_pct=gpu_busy,
            render_completion_p99_ms=p99,
            render_service_p99_ms=service_p99,
            display_frames=self.frames,
            display_deadlines=self.frames + self.misses,
            missed_frames=self.misses,
            longest_consecutive_misses=self.longest_misses,
        )
        return self.last

    def skipped_deadlines(self, count):
        extra = max(0, count - self.last_completion_misses)
        self.misses += extra
        if extra:
            self.consecutive += extra
            self.longest_misses = max(self.longest_misses, self.consecutive)

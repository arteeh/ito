import json
import os
from collections import deque
from pathlib import Path

import numpy as np

from ito import clock
from ito.rates import CounterRate, Rate

WINDOW_SECONDS = 10.0


class Distribution:
    def __init__(self, step_ms, maximum_ms):
        self.step = step_ms
        self.bins = np.zeros(round(maximum_ms / step_ms) + 1, np.uint64)
        self.count = 0
        self.maximum = 0.0
        self.recent = deque(maxlen=4096)

    def add(self, now, value):
        value = max(0.0, value)
        self.bins[min(round(value / self.step), len(self.bins) - 1)] += 1
        self.count += 1
        self.maximum = max(self.maximum, value)
        self.recent.append((now, value))

    def percentiles(self, now, levels=(50, 95, 99)):
        while self.recent and now - self.recent[0][0] > WINDOW_SECONDS:
            self.recent.popleft()
        values = [value for _, value in self.recent]
        return [float(v) for v in np.percentile(values, levels)] if values else [None] * len(levels)

    def summary(self):
        if not self.count:
            return dict(samples=0, p50_ms=None, p95_ms=None, p99_ms=None, max_ms=None)
        cumulative = self.bins.cumsum()
        values = {}
        upper_bounds = []
        for p in (50, 95, 99):
            index = int(np.searchsorted(cumulative, max(1, np.ceil(self.count * p / 100))))
            values[f"p{p}_ms"] = self.maximum if index == len(self.bins) - 1 else index * self.step
            if index == len(self.bins) - 1:
                upper_bounds.append(p)
        return dict(
            samples=self.count,
            **values,
            max_ms=self.maximum,
            bin_ms=self.step,
            upper_bound_percentiles=upper_bounds,
        )


class Performance:
    def __init__(self):
        self.started = clock.now()
        self.visible_rate = Rate(window=5, cap=1024)
        self.backend_rate = CounterRate(window=5)
        self.latency = Distribution(1, 60_000)
        self.render = Distribution(0.1, 1000)
        self.service = Distribution(0.1, 1000)
        self.frames = self.misses = self.consecutive = self.longest_misses = 0
        self.last_completion_misses = 0
        self.visible_frames = 0
        self.seen = set()
        self.seen_order = deque()
        self.latest_capture = None
        self.latest_latency = None
        self.last = {}
        self.next_snapshot = 0.0
        path = os.environ.get("ITO_TIMELINE")
        self.timeline = open(path, "a", buffering=1) if path else None
        self.last_backend = self.total_backend = 0
        self.worker_epoch = None
        self.worker_statistics = {}
        self.gpu_samples = self.gpu_total = self.gpu_maximum = 0

    def completed(self, began, due, period, updates, *, scene=True, statistics=None, gpu_busy=None):
        now = clock.now()
        self.frames += 1
        elapsed = (now - due) * 1000
        self.render.add(now, elapsed)
        self.service.add(now, (now - began) * 1000)
        missed = max(0, int(np.ceil((now - due) / period)) - 1)
        self.last_completion_misses = missed
        self.misses += missed
        self.consecutive = self.consecutive + missed if missed else 0
        self.longest_misses = max(self.longest_misses, self.consecutive)
        if scene:
            for update in updates:
                for sequence, captured in getattr(update, "observations", ()):
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
                    self.visible_frames += 1
                    if self.latest_capture is None or captured > self.latest_capture:
                        self.latest_capture, self.latest_latency = captured, latency
        statistics = statistics or self.worker_statistics
        self.worker_statistics = statistics
        calls = statistics.get("backend_calls", 0)
        epoch = statistics.get("worker_epoch")
        if epoch != self.worker_epoch:
            self.last_backend = 0
            self.backend_rate = CounterRate(window=5)
            self.worker_epoch = epoch
        self.total_backend += max(0, calls - self.last_backend)
        self.last_backend = calls
        backend_hz = self.backend_rate.hz(calls, now)
        if now < self.next_snapshot:
            return self.last
        self.next_snapshot = now + 0.5
        if gpu_busy is not None:
            self.gpu_samples += 1
            self.gpu_total += gpu_busy
            self.gpu_maximum = max(self.gpu_maximum, gpu_busy)
        p50, p95, _ = self.latency.percentiles(now)
        _, _, p99 = self.render.percentiles(now)
        _, _, service_p99 = self.service.percentiles(now)
        self.last = dict(
            **statistics,
            scene_hz=self.visible_rate.hz(now),
            scene_frames=self.visible_frames,
            backend_hz=backend_hz,
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
            statistics_window_s=WINDOW_SECONDS,
        )
        if self.timeline is not None:
            self.timeline.write(
                json.dumps(dict(event="display_statistics", time=now, **self.last)) + "\n"
            )
        return self.last

    def skipped_deadlines(self, count):
        extra = max(0, count - self.last_completion_misses)
        self.misses += extra
        if extra:
            self.consecutive += extra
            self.longest_misses = max(self.longest_misses, self.consecutive)

    def close(self, metrics=None):
        summary = self.summary()
        if metrics is not None and isinstance(getattr(metrics, "name", None), str):
            Path(metrics.name).with_suffix(".summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
        if self.timeline is not None:
            self.timeline.write(json.dumps(summary) + "\n")
            self.timeline.close()
        return summary

    def summary(self):
        elapsed = max(clock.now() - self.started, 1e-6)
        return dict(
            event="run_summary",
            seconds=elapsed,
            scene_frames=self.visible_frames,
            scene_hz=self.visible_frames / elapsed,
            backend_calls=self.total_backend,
            backend_hz=self.total_backend / elapsed,
            capture_to_visible=self.latency.summary(),
            gpu_busy_pct_mean=self.gpu_total / self.gpu_samples if self.gpu_samples else None,
            gpu_busy_pct_max=self.gpu_maximum if self.gpu_samples else None,
            render_completion=self.render.summary(),
            render_service=self.service.summary(),
            display_frames=self.frames,
            display_deadlines=self.frames + self.misses,
            missed_frames=self.misses,
            longest_consecutive_misses=self.longest_misses,
            last_statistics=self.last,
        )

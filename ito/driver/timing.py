"""Where pilot poses bunch on their way to the robot, for live measurement (#31).

ITO_POSE_TIMING=<file> makes the driver append one JSON line per second: poses received,
rejected, applied and superseded (replaced by a newer one before they were applied), and
histograms of the gaps between them as the pilot sent them (capture times, pilot clock)
and as they arrived here. Even sends with bunched arrivals point at the link; bunched sends
at the pilot; even arrivals with few applies at the driver's own loop.
"""

import json
import os
from pathlib import Path

from ito import clock

# Histogram bin edges, milliseconds: a 90 Hz pilot sends every 11.1 ms.
EDGES = (2, 5, 8, 10, 12, 14, 20, 30, 50)


def _histogram(gaps):
    counts = [0] * (len(EDGES) + 1)
    for gap in gaps:
        counts[next((i for i, edge in enumerate(EDGES) if gap < edge), len(EDGES))] += 1
    return counts


def _percentile(values, fraction):
    return round(sorted(values)[int(fraction * (len(values) - 1))], 2) if values else None


class PoseTiming:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = clock.now()
        self._reset()
        self.last_arrival = self.last_capture = self.last_apply = self.last_sequence = None

    @classmethod
    def from_environment(cls):
        path = os.environ.get("ITO_POSE_TIMING")
        return cls(path) if path else None

    def _reset(self):
        self.window = clock.now()
        self.received = self.rejected = self.applied = self.superseded = self.lost = 0
        self.arrivals, self.sends, self.applies, self.ages = [], [], [], []

    def received_pose(self, sequence, capture, now, pending):
        """A pose arrived: its sequence, capture time in local clock, and whether an earlier
        pose was still waiting to be applied (this one replaces it)."""
        self.received += 1
        self.superseded += pending
        if self.last_arrival is not None:
            self.arrivals.append((now - self.last_arrival) * 1000)
            self.sends.append((capture - self.last_capture) * 1000)
            self.lost += max(0, sequence - self.last_sequence - 1)
        self.last_arrival, self.last_capture, self.last_sequence = now, capture, sequence
        self.ages.append((now - capture) * 1000)
        self._flush(now)

    def rejected_pose(self):
        self.rejected += 1

    def applied_pose(self, now):
        self.applied += 1
        if self.last_apply is not None:
            self.applies.append((now - self.last_apply) * 1000)
        self.last_apply = now
        self._flush(now)

    def _flush(self, now):
        seconds = now - self.window
        if seconds < 1:
            return
        record = dict(
            t=round(now - self.started, 2),
            seconds=round(seconds, 3),
            received_hz=round(self.received / seconds, 1),
            applied_hz=round(self.applied / seconds, 1),
            rejected=self.rejected,
            superseded=self.superseded,
            lost=self.lost,
            gap_edges_ms=EDGES,
            send_gaps=_histogram(self.sends),
            arrival_gaps=_histogram(self.arrivals),
            apply_gaps=_histogram(self.applies),
            arrival_gap_ms_p50_p95_max=[
                _percentile(self.arrivals, 0.5),
                _percentile(self.arrivals, 0.95),
                _percentile(self.arrivals, 1.0),
            ],
            send_to_arrival_ms_p50_p95=[_percentile(self.ages, 0.5), _percentile(self.ages, 0.95)],
        )
        with self.path.open("a") as log:
            log.write(json.dumps(record) + "\n")
        self._reset()

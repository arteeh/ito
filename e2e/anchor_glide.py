"""The eye glides between SLAM-rate anchors instead of stepping.

LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a uv run python e2e/anchor_glide.py

The real desktop window draws while the anchor moves forward in 6 cm steps at 5 Hz, a robot
driving at 0.3 m/s seen through SLAM. Each displayed eye position must move in smaller
steps than the updates, never backwards, and trail the newest anchor by about one interval.
"""

import json
from pathlib import Path

import numpy as np

from ito import clock
from ito.desktop import DesktopState, DesktopWindow
from ito.render import pose

OUT = Path("e2e/out/anchor-glide")
STEP, RATE, SECONDS = 0.06, 5, 4


class Empty:
    def poll(self):
        return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    began = clock.now()
    anchors = {}
    drawn = []

    def state():
        index = int((clock.now() - began) * RATE)
        if index not in anchors:
            anchors[index] = pose((0, 0, -STEP * index))
        return DesktopState(anchors[index])

    with DesktopWindow((640, 480), fps=90, capture_dir=OUT) as window:
        draw = window.draw_view

        def recorded(current, head, *args, **kwargs):
            now = clock.now()
            drawn.append((now, float(current.robot_camera[2, 3]), (now - began) * RATE))
            draw(current, head, *args, **kwargs)

        window.draw_view = recorded

        def stop(value):
            if clock.now() - began > SECONDS:
                window.overlay.leave = True

        window.run(Empty(), state=state, on_input=stop, save_settings=lambda _: None)
    positions = np.array([z for _, z, _ in drawn])
    steps = -np.diff(positions)
    settled = [(t, z, u) for t, z, u in drawn if u > 2]
    # Distance behind the newest anchor, in update intervals.
    lag = [(-STEP * int(u) - z) / -STEP for _, z, u in settled]
    result = {
        "display_frames": len(drawn),
        "display_hz": round(len(drawn) / SECONDS, 1),
        "update_step_cm": STEP * 100,
        "max_drawn_step_cm": round(float(steps.max()) * 100, 2),
        "lag_intervals_max": round(max(lag), 2),
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    assert len(drawn) >= SECONDS * RATE * 3, "display too slow to show gliding"
    assert steps.min() > -1e-6, "the eye moved backwards"
    assert steps.max() < STEP * 0.75, result
    assert max(lag) < 2.2, result
    print("PASS: the eye glides between 5 Hz anchors with about one interval of delay")


if __name__ == "__main__":
    main()

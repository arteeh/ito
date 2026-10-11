"""Fusion's corrections reach the pilot slowly, and only while their view moves.

LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a uv run python e2e/correction_ease.py

The real desktop window draws a robot-first eye: the robot's camera, moved by fusion's
correction. One second in, the correction jumps by 4 degrees of yaw and 6 cm, as when vision
corrects the robot's odometry. The robot stands still until 2.5 s, then walks at 0.3 m/s
until 5 s.
The drawn eye must not step at the jump, must not change while the view is still, and must
then turn at most 2 degrees/s and slide at most 5 cm/s until it reaches fusion's pose.
"""

import json
import math
from pathlib import Path

import numpy as np

from ito import clock
from ito.desktop import DesktopState, DesktopWindow
from ito.render import pose

OUT = Path("e2e/out/correction-ease")
JUMP_AT, WALK_AT, HALT_AT, SECONDS = 1.0, 2.5, 5.0, 6.0
TURN, SLIDE = math.radians(4), np.array([0.06, 0.0, 0.0])
SPEED = 0.3


class Empty:
    def poll(self):
        return None


def angle(rotation):
    return math.acos(max(-1.0, min(1.0, (np.trace(rotation) - 1) / 2)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    began = clock.now()
    corrected = pose(SLIDE, yaw=TURN)
    drawn = []
    robots = {}

    def state():
        t = clock.now() - began
        index = int(t * 50)  # the robot's motion arrives at 50 Hz
        if index not in robots:
            walked = min(max(0.0, index / 50 - WALK_AT), HALT_AT - WALK_AT)
            robots[index] = pose((0, 0, -SPEED * walked))
        return DesktopState(robots[index], correction=corrected if t >= JUMP_AT else pose())

    with DesktopWindow((640, 480), fps=90, capture_dir=OUT) as window:
        draw = window.draw_view

        def recorded(current, head, *args, **kwargs):
            now = clock.now()
            # The robot's camera as glided this frame, before fusion's correction.
            robot = np.append(window.glide.position(now), 1.0)
            drawn.append((now - began, np.array(current.robot_camera, dtype=np.float64), robot))
            draw(current, head, *args, **kwargs)

        window.draw_view = recorded

        def stop(value):
            if clock.now() - began > SECONDS:
                window.overlay.leave = True

        window.run(Empty(), state=state, on_input=stop, save_settings=lambda _: None)
    times = np.array([t for t, _, _ in drawn])
    turns = np.array([angle(a[:3, :3]) for _, a, _ in drawn])
    # Where fusion puts the eye (the corrected robot camera) against where it is drawn.
    target = [((corrected if t >= JUMP_AT else np.eye(4)) @ r)[:3] for t, _, r in drawn]
    offsets = np.array([a[:3, 3] for _, a, _ in drawn]) - np.array(target)
    slides = np.linalg.norm(offsets, axis=1)
    # Rates over a quarter second: the drawn frame's timestamp jitters against the window's.
    later = np.searchsorted(times, times + 0.25)
    span = later < len(times)
    first, last = np.flatnonzero(span), later[span]
    dt = times[last] - times[first]
    turn_rate = np.degrees(np.abs(turns[last] - turns[first]) / dt)
    slide_rate = np.abs(slides[last] - slides[first]) / dt
    moving = (times[first] > WALK_AT) & (times[last] < HALT_AT)
    still = (times[1:] > JUMP_AT + 0.05) & (times[1:] < WALK_AT - 0.05)
    result = {
        "display_frames": len(drawn),
        "turn_at_jump_deg": round(math.degrees(turns[np.searchsorted(times, JUMP_AT + 0.05)]), 2),
        "slide_at_jump_cm": round(float(slides[np.searchsorted(times, JUMP_AT + 0.05)]) * 100, 2),
        "still_change_deg": round(float(np.ptp(np.degrees(turns[1:][still]))), 4),
        "still_change_cm": round(float(np.ptp(slides[1:][still])) * 100, 4),
        "turn_rate_max_deg_s": round(float(turn_rate[moving].max()), 2),
        "slide_rate_max_cm_s": round(float(slide_rate[moving].max()) * 100, 2),
        "final_turn_deg": round(math.degrees(turns[-1]), 3),
        "final_slide_cm": round(float(slides[-1]) * 100, 3),
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    assert len(drawn) >= SECONDS * 20, "display too slow to measure easing"
    # Nothing jumps: the whole correction is held back at first.
    assert abs(result["turn_at_jump_deg"] - 4) < 0.1, result
    assert abs(result["slide_at_jump_cm"] - 6) < 0.5, result
    assert result["still_change_deg"] < 1e-3 and result["still_change_cm"] < 1e-3, result
    assert result["turn_rate_max_deg_s"] <= 2.05, result
    assert result["slide_rate_max_cm_s"] <= 5.1, result
    assert result["final_turn_deg"] < 0.01 and result["final_slide_cm"] < 0.1, result
    print("PASS: corrections ease in at 2 degrees/s and 5 cm/s, only while the view moves")


if __name__ == "__main__":
    main()

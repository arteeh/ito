"""Real bundled driver, RGB-D and SDL mouse motion; measure every displayed view.

The pilot turns past the head's pan range, so the body turns too. The view must turn
exactly with the pilot's mouse, never with the robot's own head or body.

DISPLAY=:97 LIBGL_ALWAYS_SOFTWARE=1 uv run python e2e/view_motion.py
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pygame
from pygame._sdl2 import Window

from ito import clock
from ito.app.__main__ import main as pilot_main

OUT = Path("e2e/out/view-motion")


def key(code):
    for kind in (pygame.KEYDOWN, pygame.KEYUP):
        pygame.event.post(pygame.event.Event(kind, key=code))


def run(measure=False):
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CONFIG_HOME"] = os.environ["APPDATA"] = str(OUT / "config")
    began = clock.now()
    stage, changed = 0, began
    samples = []
    updates = []
    installed = False
    previous = began
    replayed = 0
    stale_errors = []

    def drive(app, window, value):
        nonlocal stage, changed, installed, previous
        now = clock.now()
        assert now - began < 35, (stage, app.state.status)
        if not installed:
            draw = window.draw_view
            apply = window.renderer.apply
            submit = app._submit
            last_pair = None

            def delayed(pair, peer):
                nonlocal last_pair, replayed
                submit(pair, peer)
                # Replay real, previously joined exposures after a newer one, as a
                # delayed consumer would. Neither video nor its anchor may go back.
                if pair is not None and last_pair is not None and not measure:
                    newest = app.state.video_time
                    submit(last_pair, peer)
                    replayed += 1
                    if app.state.video_time != newest:
                        stale_errors.append((newest, app.state.video_time))
                if pair is not None:
                    last_pair = pair

            app._submit = delayed

            def display(current, head, *args, **kwargs):
                draw(current, head, *args, **kwargs)
                if stage == 2:
                    view = current.robot_camera @ head
                    samples.append(
                        {
                            "time": clock.now(),
                            "yaw": math.atan2(float(view[0, 2]), float(view[2, 2])),
                            "head_yaw": math.atan2(float(head[0, 2]), float(head[2, 2])),
                            "video_time": current.video_time,
                            "scene_time": updates[-1] if updates else 0,
                            "physical_pan": app.telemetry.get("head_pan", 0),
                            "body_yaw": app.telemetry.get("base_yaw", 0),
                        }
                    )

            def upload(update):
                updates.append(update.captured_at)
                apply(update)

            window.draw_view = display
            window.renderer.apply = upload
            installed = True
        if stage == 0 and app.matched_frames > 15 and window.renderer.count > 1000:
            Window.from_display_module().focus()
            key(pygame.K_TAB)
            key(pygame.K_r)
            stage, changed = 1, now
        elif stage == 1 and now - changed > 1:
            stage, changed = 2, now
        elif stage == 2:
            # Physical mouse events, at a steady 0.4 radians/second, for six seconds: 137
            # degrees, past the head's 80 degree pan range.
            pygame.event.post(
                pygame.event.Event(
                    pygame.MOUSEMOTION,
                    pos=(400, 300),
                    rel=(-0.4 * (now - previous) / window.input.sensitivity, 0),
                    buttons=(0, 0, 0),
                )
            )
            if now - changed > 6:
                stage = 3
                pygame.event.post(pygame.event.Event(pygame.QUIT))
        previous = now

    assert (
        pilot_main(
            [
                "--sim",
                "--size",
                "640",
                "480",
                "--fps",
                "90",
                "--max-splats",
                "16384",
                "--audio-source",
                "none",
                "--audio-sink",
                "none",
            ],
            on_frame=drive,
        )
        == 0
    )
    assert stage == 3 and len(samples) > 30, (stage, len(samples))
    yaw = np.unwrap([s["yaw"] for s in samples])
    head = np.unwrap([s["head_yaw"] for s in samples])
    delta = np.diff(yaw)
    report = {
        "display_frames": len(samples),
        "late_exposures_replayed": replayed,
        "physical_pan_sweep_deg": math.degrees(
            max(s["physical_pan"] for s in samples) - min(s["physical_pan"] for s in samples)
        ),
        "body_turn_deg": math.degrees(
            max(s["body_yaw"] for s in samples) - min(s["body_yaw"] for s in samples)
        ),
        "reversals_over_0.1_deg": int(np.count_nonzero(delta < -math.radians(0.1))),
        "worst_reversal_deg": float(max(0, -np.min(delta)) * 180 / math.pi),
        "anchor_error_p95_deg": float(np.percentile(np.abs(yaw - head), 95) * 180 / math.pi),
        "older_scene_frames": int(np.count_nonzero(np.diff(updates) < 0)),
        "older_video_frames": int(
            np.count_nonzero(np.diff([s["video_time"] for s in samples]) < 0)
        ),
        "scene_video_skew_ms_p95": float(
            np.percentile([abs(s["video_time"] - s["scene_time"]) * 1000 for s in samples], 95)
        ),
    }
    (OUT / ("baseline.json" if measure else "summary.json")).write_text(
        json.dumps(report, indent=2)
    )
    (OUT / ("baseline-frames.json" if measure else "frames.json")).write_text(json.dumps(samples))
    print(json.dumps(report, indent=2))
    if not measure:
        assert replayed > 20 and not stale_errors, stale_errors
        assert report["physical_pan_sweep_deg"] > 40, report
        assert report["reversals_over_0.1_deg"] == 0, report
        assert report["anchor_error_p95_deg"] < 0.2, report
        assert report["older_scene_frames"] == report["older_video_frames"] == 0, report
        assert report["body_turn_deg"] > 20, report
        assert yaw[-1] - yaw[0] > 2, "View did not follow local mouse motion"
        print("PASS: sustained mouse-look remains monotonic while the physical head follows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true", help="record an unfixed baseline")
    run(parser.parse_args().measure)

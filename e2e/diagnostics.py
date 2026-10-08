"""Drive the real pilot UI: opt in/out, environment overrides, rotation and shutdown.

DISPLAY=:0 LIBGL_ALWAYS_SOFTWARE=1 uv run python e2e/diagnostics.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pygame

from e2e.simulated import click, key
from ito import clock, diagnostics
from ito.app.__main__ import main as pilot_main

OUT = Path("e2e/out/diagnostics")


def phase(mode):
    root = OUT / mode
    root.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CONFIG_HOME"] = str(root)
    path = root / "ito/diagnostics.jsonl"
    for old in path.parent.glob("diagnostics.jsonl*"):
        old.rmdir() if old.is_dir() else old.unlink()
    if mode == "unavailable":
        path.mkdir(parents=True)  # A real filesystem error, even when run as root.
    # Exercise the same rotating handler with a small budget in a real running app.
    diagnostics.MAX_BYTES = 8000
    stage, changed = "start", clock.now()
    saved_size = 0

    def go(value):
        nonlocal stage, changed
        stage, changed = value, clock.now()

    def drive(app, window, value):
        nonlocal saved_size
        waited = clock.now() - changed
        assert waited < 25, stage
        layout = window.overlay.layout
        if app.state.status.link != "CONNECTED" or "diagnostics" not in layout:
            return
        if stage == "start":
            go("ready")
        elif stage == "ready" and waited > 0.7:
            assert diagnostics.current().enabled == (mode == "on")
            if mode not in {"on", "unavailable"}:
                assert not path.exists()
            click(layout["diagnostics"])
            go("clicked")
        elif stage == "clicked" and waited > 1:
            if mode == "unavailable" and path.is_dir():
                assert not diagnostics.current().enabled
                assert diagnostics.current().error
                path.rmdir()
                click(layout["diagnostics"])
                go("clicked")
                return
            assert diagnostics.current().enabled == (mode != "off")
            if mode == "off":
                assert not path.exists()
                pygame.event.post(pygame.event.Event(pygame.QUIT))
                go("done")
            else:
                key(pygame.K_TAB)
                go("capture")
        elif stage == "capture" and waited > 1:
            assert window.input.captured
            key(pygame.K_ESCAPE)
            go("release")
        elif stage == "release" and waited > 1:
            assert not window.input.captured
            if mode in {"toggle", "unavailable"}:
                click(layout["diagnostics"])
                go("disabled")
            else:
                go("shutdown")
        elif stage == "disabled" and waited > 1:
            assert not diagnostics.current().enabled
            saved_size = path.stat().st_size
            go("quiet")
        elif stage == "quiet" and waited > 1:
            assert path.stat().st_size == saved_size, "Off still writes diagnostics"
            click(layout["diagnostics"])
            go("shutdown")
        elif stage == "shutdown" and waited > 3:
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            go("done")

    assert (
        pilot_main(
            ["--sim", "--size", "900", "700", "--fps", "30", "--reconstruction", "video"],
            on_frame=drive,
        )
        == 0
    )
    assert stage == "done"
    files = list(path.parent.glob("diagnostics.jsonl*"))
    if mode == "off":
        assert not files
        return
    assert 1 < len(files) <= diagnostics.BACKUPS + 1, files
    assert all(p.stat().st_size <= diagnostics.MAX_BYTES + 1000 for p in files)
    records = [json.loads(line) for p in files for line in p.read_text().splitlines()]
    events = {r["event"] for r in records}
    assert {
        "display_frame",
        "input_freshness",
        "frame_join",
        "input_capture",
        "audio_state",
        "shutdown_stage",
        "diagnostics_closed",
    } <= events, events
    stages = {r["stage"] for r in records if r["event"] == "shutdown_stage" and r["state"] == "end"}
    assert {
        "audio_devices",
        "audio",
        "webrtc",
        "link_thread",
        "window",
        "simulation_viewer",
    } <= stages, stages
    assert len({r["run_id"] for r in records}) == 1
    assert all(r["build"] and r["timestamp"] and r["monotonic"] for r in records)
    forbidden = {"code", "credential", "address", "head", "buttons", "axes", "movement", "samples"}
    assert not any(forbidden & r.keys() for r in records)
    print(mode, len(records), "records; stages", sorted(stages))


def main():
    if len(sys.argv) > 1:
        phase(sys.argv[1])
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for mode, override in (("toggle", ""), ("on", "1"), ("off", "0"), ("unavailable", "")):
        result = subprocess.run(
            [sys.executable, "-m", "e2e.diagnostics", mode],
            env=os.environ | {"ITO_DEBUG": override},
            capture_output=True,
            text=True,
            timeout=90,
        )
        (OUT / f"{mode}.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, (mode, result.stdout, result.stderr)
        print(result.stdout)
    print("PASS: UI toggle, overrides, rotation, private events, shutdown and log failure recovery")


if __name__ == "__main__":
    main()

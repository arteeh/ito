import argparse
import logging
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import moderngl
import pygame

from ito.desktop import DesktopWindow
from ito.desktop.settings import settings_path
from ito.link.audio import Audio, arguments

from . import connect
from .pilot import Pilot
from .settings import Settings
from .sim import SimulatedRobot

log = logging.getLogger(__name__)


def alert(message):
    """Without a console (ito.exe double-clicked), errors still have to reach the pilot."""
    if sys.stderr is None and os.name == "nt":
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, "Ito", 0x10)


def configure_logging():
    if sys.stderr is not None:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        return
    path = settings_path().parent / "ito.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(path, "w", encoding="utf-8")],
    )

    def crashed(kind, value, trace):
        log.critical("Ito stopped", exc_info=(kind, value, trace))
        alert(f"Ito stopped unexpectedly: {value}\n\nDetails are in {path}")

    sys.excepthook = crashed


def main(argv=None, *, on_frame=None):
    parser = argparse.ArgumentParser(prog="ito", description="Pilot one Ito robot")
    parser.add_argument(
        "address", nargs="?", help="driver host:port or HTTP(S) URL (omit to choose on screen)"
    )
    parser.add_argument("--sim", action="store_true", help="pilot the bundled simulated robot")
    parser.add_argument("--mode", choices=("desktop", "xr"), default="desktop")
    parser.add_argument("--reference-space", choices=("seated", "standing"), default="seated")
    parser.add_argument(
        "--reconstruction",
        choices=("auto", "rgbd", "slam", "video"),
        help="persist reconstruction backend (default: automatic)",
    )
    parser.add_argument("--camera", help="camera name (defaults to the first camera)")
    parser.add_argument("--cameras", type=int, default=1, help="number of driver video tracks")
    parser.add_argument("--size", type=int, nargs=2, default=(1280, 720))
    parser.add_argument("--fps", type=int, default=90)
    parser.add_argument("--fov", type=float, help="persist vertical field of view, 30–120 degrees")
    parser.add_argument("--sensitivity", type=float, help="persist mouse radians per pixel")
    parser.add_argument("--invert-y", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--move-x", help="persist robot axis mapped to A/D or left stick X")
    parser.add_argument("--move-y", help="persist robot axis mapped to W/S or left stick Y")
    parser.add_argument("--max-splats", type=int, help="persist the live scene budget")
    parser.add_argument("--capture-dir", type=Path, default=Path("captures"))
    parser.add_argument("--metrics", type=Path, help="write display and latency metrics as JSONL")
    parser.add_argument("--frames", type=int, default=0, help="exit after N display frames")
    arguments(parser)
    args = parser.parse_args(argv)
    if args.sim and args.address:
        parser.error("--sim replaces the robot address")
    configure_logging()
    overrides = {
        key: getattr(args, key)
        for key in Settings.model_fields
        if getattr(args, key, None) is not None
    }
    # Without an address or --sim, the pilot picks a robot on screen and can come back to it.
    choosing = args.address is None and not args.sim
    choice = connect.Choice(args.address, args.mode == "xr")
    window = window_mode = error = None

    def open_window(mode):
        window_type = DesktopWindow
        options = {}
        if mode == "xr":
            from ito.xr import XRWindow

            window_type = XRWindow
            options["reference_space"] = args.reference_space
        return window_type(
            args.size,
            fps=args.fps,
            capture_dir=args.capture_dir,
            max_splats=args.max_splats,
            **options,
        )

    try:
        Audio(args.audio_source, args.audio_sink)  # Validate before any window opens.
        Settings.model_validate(Settings().model_dump() | overrides)
        if args.frames < 0 or not 1 <= args.cameras <= 16:
            raise ValueError("frames must be nonnegative and cameras between 1 and 16")
        if args.metrics:
            args.metrics.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics.open("w") if args.metrics else nullcontext() as metrics:
            while True:
                if choosing:
                    if window_mode != "desktop":
                        if window:
                            window.close()
                        window, window_mode = open_window("desktop"), "desktop"
                    choice = connect.choose(window, xr=choice.xr, error=error)
                    if choice is None:
                        return 0
                    error = None
                mode = "xr" if choice.xr else "desktop"
                if mode != window_mode:
                    if window:
                        window.close()
                        window = window_mode = None
                    try:
                        window, window_mode = open_window(mode), mode
                    except RuntimeError as exc:
                        if not choosing or mode == "desktop":
                            raise
                        log.error("%s", exc)
                        error = str(exc)
                        continue
                window.overlay.can_leave = choosing
                window.overlay.leave = False
                pilot_window(window, args, overrides, metrics, choice, on_frame)
                if not window.overlay.leave:
                    return 0
    except (OSError, ValueError, RuntimeError, pygame.error, moderngl.Error) as exc:
        log.error("%s", exc)
        alert(str(exc))
        return 1
    finally:
        if window:
            window.close()


def pilot_window(window, args, overrides, metrics, choice, on_frame):
    with (
        SimulatedRobot() if choice.address is None else nullcontext() as sim,
        Pilot(
            choice.address or sim.address,
            defaults=Settings(max_splats=window.max_splats),
            overrides=overrides,
            camera=args.camera,
            cameras=args.cameras,
            audio_source=args.audio_source,
            audio_sink=args.audio_sink,
            persist=sim is None,
        ) as pilot,
    ):
        window.input.translate = False
        revision = -1

        def input_frame(value):
            nonlocal revision
            if revision != pilot.settings_revision:
                revision = pilot.settings_revision
                selected = pilot.settings
                window.fov = math.radians(selected.fov)
                window.input.sensitivity = selected.sensitivity
                window.input.invert_y = selected.invert_y
                window.overlay.max_splats = selected.max_splats
                if selected.max_splats > window.splat_limit:
                    pilot.set_max_splats(window.splat_limit)
            if sim and (failure := sim.failure()):
                window.overlay.error = failure
            pilot.input(value)
            if on_frame:
                on_frame(pilot, window, value)

        window.run(
            pilot,
            state=lambda: pilot.state,
            on_input=input_frame,
            max_frames=args.frames,
            metrics=metrics,
            save_settings=lambda _: None,
        )


if __name__ == "__main__":
    raise SystemExit(main())

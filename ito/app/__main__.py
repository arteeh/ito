import argparse
import logging
import math
import os
import sys

import pygame

from ito import diagnostics
from ito.desktop import DesktopWindow
from ito.desktop.settings import load, settings_path
from ito.link.pairing import DIGITS, normalize

from . import connect, settings
from .pilot import Pilot, ShutdownStuck
from .settings import Settings

log = logging.getLogger(__name__)


def alert(message):
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


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ito", description="Pilot one Ito robot")
    parser.add_argument(
        "address", nargs="?", help="driver host:port or HTTP(S) URL (omit to choose on screen)"
    )
    parser.add_argument("--code", help="the robot's pairing code (needed once per robot)")
    parser.add_argument("--mode", choices=("desktop", "xr"), default="desktop")
    parser.add_argument("--reference-space", choices=("seated", "standing"), default="seated")
    parser.add_argument("--size", type=int, nargs=2, default=(1280, 720))
    parser.add_argument("--fps", type=int, default=90)
    parser.add_argument("--fov", type=float, help="persist vertical field of view, 30–120 degrees")
    parser.add_argument("--sensitivity", type=float, help="persist mouse radians per pixel")
    parser.add_argument("--invert-y", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--move-x", help="persist robot axis mapped to A/D or left stick X")
    parser.add_argument("--move-y", help="persist robot axis mapped to W/S or left stick Y")
    args = parser.parse_args(argv)
    code = args.code
    if code is not None and (code := normalize(code)) is None:
        parser.error(f"a pairing code is {DIGITS} digits")
    configure_logging()
    diagnostics.current()
    overrides = {
        key: getattr(args, key)
        for key in Settings.model_fields
        if getattr(args, key, None) is not None
    }
    choosing = args.address is None
    credential = settings.credential(args.address) if args.address else None
    choice = connect.Choice(args.address, args.mode == "xr", code, credential)
    window = window_mode = error = pairing = failure = None
    stuck = False

    def open_window(mode):
        window_type = DesktopWindow
        options = {}
        if mode == "xr":
            from ito.xr import XRWindow

            window_type = XRWindow
            options["reference_space"] = args.reference_space
        return window_type(args.size, fps=args.fps, **options)

    try:
        Settings.model_validate(Settings().model_dump() | overrides)
        while True:
            if choosing:
                menu_mode = "xr" if choice.xr else "desktop"
                if window_mode != menu_mode:
                    if window:
                        window.close()
                    window = window_mode = None
                    try:
                        window, window_mode = open_window(menu_mode), menu_mode
                    except RuntimeError as exc:
                        if menu_mode == "desktop":
                            raise
                        error = str(exc)
                        choice = connect.Choice(None)
                        continue
                choice = connect.choose(window, xr=choice.xr, error=error, pairing=pairing)
                if choice is None:
                    return 0
                if choice.menu:
                    continue
                error = pairing = None
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
            refusal = pilot_window(window, args, overrides, choice)
            if refusal:
                choosing, pairing, error = True, choice.address, refusal
                continue
            if not window.overlay.leave:
                return 0
    except (OSError, ValueError, RuntimeError, pygame.error) as exc:
        log.error("%s", exc)
        failure = str(exc)
        stuck = isinstance(exc, ShutdownStuck)
        return 1
    finally:
        try:
            if window:
                with diagnostics.stage("window"):
                    window.close()
        finally:
            diagnostics.close()
            if failure:
                alert(failure)
            if stuck:
                # Python joins every live non-daemon thread and worker process before it
                # exits, with no timeout: aiortc's decoder threads in a stuck link, or a
                # worker that would not die, would keep the closed app running for good.
                log.error("Ito exits without waiting for what did not shut down")
                logging.shutdown()
                os._exit(1)


def pilot_window(window, args, overrides, choice):
    with Pilot(
        choice.address,
        defaults=Settings(**load().model_dump()),
        overrides=overrides,
        code=choice.code,
        credential=choice.credential,
    ) as pilot:
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
            if pilot.refusal:
                window.overlay.leave = True

        window.run(pilot, state=lambda: pilot.state, on_input=input_frame, on_sample=pilot.input)
    return pilot.refusal


if __name__ == "__main__":
    raise SystemExit(main())

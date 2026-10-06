import argparse
import logging
import math
from contextlib import nullcontext
from pathlib import Path

import moderngl
import pygame

from ito.desktop import DesktopWindow

from .pilot import Pilot
from .settings import Settings


def main(argv=None, *, on_frame=None):
    parser = argparse.ArgumentParser(description="Pilot one Ito robot")
    parser.add_argument("address", help="driver host:port or HTTP(S) URL")
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
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    overrides = {
        key: getattr(args, key)
        for key in Settings.model_fields
        if getattr(args, key, None) is not None
    }
    try:
        window_type = DesktopWindow
        window_options = {}
        if args.mode == "xr":
            from ito.xr import XRWindow

            window_type = XRWindow
            window_options["reference_space"] = args.reference_space
        Settings.model_validate(Settings().model_dump() | overrides)
        if args.frames < 0 or not 1 <= args.cameras <= 16:
            raise ValueError("frames must be nonnegative and cameras between 1 and 16")
        if args.metrics:
            args.metrics.parent.mkdir(parents=True, exist_ok=True)
        with (
            args.metrics.open("w") if args.metrics else nullcontext() as metrics,
            window_type(
                args.size,
                fps=args.fps,
                capture_dir=args.capture_dir,
                max_splats=args.max_splats,
                **window_options,
            ) as window,
            Pilot(
                args.address,
                defaults=Settings(max_splats=window.max_splats),
                overrides=overrides,
                camera=args.camera,
                cameras=args.cameras,
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
    except (OSError, ValueError, RuntimeError, pygame.error, moderngl.Error) as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

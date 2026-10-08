"""In a headset, both eyes see one flat camera feed panel straight ahead of the pilot.

Start the XR runtime, then: uv run python e2e/openxr_flat.py
Linux with Monado: XRT_COMPOSITOR_NULL=1 monado-service, then
DISPLAY=:97 LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=osmesa uv run python e2e/openxr_flat.py

The real XR app pilots the bundled MuJoCo robot with the flat camera feed. Every eye image
is read back from its swapchain and must show the panel. Each eye is also drawn again from
its runtime pose and frustum at 1024 pixels wide; there the panel must sit where a panel 2 m
ahead of the point between the eyes projects: the left eye sees it a little right of its own
forward, the right eye a little left, by half the eye separation over 2 m, and both see it
the same size. A panel drawn per eye (no disparity) or with the eye offset reversed fails.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from ito import clock

OUT = Path("e2e/out/xr-flat")
FRAMES = 30  # Measured stereo frames after the live feed arrives.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.child:
        # A runtime waiting for a worn headset must not hang the run.
        with (OUT / "pilot.log").open("w") as log:
            child = subprocess.Popen(
                [sys.executable, __file__, "--child"],
                stdout=log,
                stderr=log,
                start_new_session=sys.platform != "win32",
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
            )
            try:
                result = child.wait(timeout=120)
            except BaseException:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], check=False)
                else:
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                raise
        if result:
            raise SystemExit((OUT / "pilot.log").read_text())
        print((OUT / "summary.json").read_text())
        print("PASS: both eyes see one head-locked flat feed panel with correct disparity")
        return
    run()


def run():
    import numpy as np
    import pygame
    from OpenGL import GL

    from ito.app.__main__ import main as pilot_main
    from ito.driver import pairing
    from ito.render.video import DISTANCE, VideoPanel

    os.environ["XDG_CONFIG_HOME"] = str(OUT / "config")
    with socket.socket() as port:
        port.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{port.getsockname()[1]}"
    code_file = OUT / "pairing-code"
    code = pairing.rotate(code_file)
    driver_log = (OUT / "driver.log").open("w")
    robot = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "drivers.mujoco.cli",
            "--port",
            address.split(":")[1],
            "--gl",
            "glfw" if sys.platform == "win32" else "osmesa",
            "--pairing-file",
            str(code_file),
        ],
        stdout=driver_log,
        stderr=driver_log,
    )
    # A bright frame with a dark marker top left: edges are exact and mirroring shows.
    rgb = np.full((240, 320, 3), (240, 200, 40), dtype=np.uint8)
    rgb[20:100, 20:120] = (20, 60, 220)
    frames = []  # Per display frame, per eye: what it showed and how it was drawn.
    began = clock.now()
    installed = False

    def panel(pixels):
        yellow = (pixels[..., 0] > 150) & (pixels[..., 1] > 120) & (pixels[..., 2] < 140)
        blue = (pixels[..., 2] > 150) & (pixels[..., 0] < 120)
        rows, columns = np.nonzero(yellow | blue)
        if not len(rows):
            return None
        marker_rows, marker_columns = np.nonzero(blue)
        assert len(marker_rows), "panel marker missing"
        assert marker_rows.mean() < rows.mean() and marker_columns.mean() < columns.mean(), (
            "panel is mirrored or rotated"
        )
        return columns.min(), columns.max() + 1, rows.min(), rows.max() + 1

    def read(size):
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        data = GL.glReadPixels(0, 0, *size, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        return np.frombuffer(data, np.uint8).reshape(size[1], size[0], 3)[::-1]

    def measure(window):
        draw = window.draw_view
        # The runtime's eye images may be small; the same eye pose and frustum drawn
        # again at this width resolves the few pixels of disparity.
        fine = None

        def measured(current, eye, projection, target, viewport, *args, center=None, **kwargs):
            nonlocal fine
            live = current.flat_video and current.video is not None
            if live:
                # The live feed proved the flat path is up; draw a known frame to measure.
                current = replace(current, video=rgb, video_time=clock.now())
            draw(current, eye, projection, target, viewport, *args, center=center, **kwargs)
            if not live or center is None:
                return
            seen = panel(read(viewport[2:]))
            size = (1024, round(1024 * viewport[3] / viewport[2]))
            if fine is None:
                fine = window.context.simple_framebuffer(size)
            draw(current, eye, projection, fine, (0, 0, *size), *args, center=center, **kwargs)
            entry = dict(
                eye=eye.copy(),
                projection=projection.copy(),
                center=center.copy(),
                fov=current.video_fov,
                seen=seen,
                size=size,
                box=panel(read(size)),
            )
            if frames and len(frames[-1]) == 1:
                frames[-1].append(entry)
            else:
                frames.append([entry])
            if sum(len(f) == 2 for f in frames) >= FRAMES:
                pygame.event.post(pygame.event.Event(pygame.QUIT))

        window.draw_view = measured

    def drive(app, window, value):
        nonlocal installed
        assert clock.now() - began < 90, (app.state.status, app.reconstruction_status)
        if not installed:
            measure(window)
            installed = True

    try:
        result = pilot_main(
            [
                address,
                "--code",
                code,
                "--mode",
                "xr",
                "--reconstruction",
                "video",
                "--size",
                "768",
                "440",
            ],
            on_frame=drive,
        )
        assert result == 0, result
    finally:
        if robot.poll() is None:
            robot.terminate()
            robot.wait(timeout=10)
        driver_log.close()

    stereo = [f for f in frames if len(f) == 2]
    assert len(stereo) >= FRAMES, f"only {len(stereo)} stereo flat-feed frames"

    def tangents(entry):
        """Panel centre and width as tangents in this eye's frame: measured and expected."""
        width, height = entry["size"]
        left, right, top, bottom = entry["box"]
        p = entry["projection"]
        ndc_x = (left + right) / width - 1
        ndc_y = 1 - (top + bottom) / height
        measured = (
            (ndc_x + p[0, 2]) / p[0, 0],
            (ndc_y + p[1, 2]) / p[1, 1],
            (right - left) / width * 2 / p[0, 0],
        )
        ahead = np.linalg.inv(entry["eye"]) @ entry["center"] @ (0.0, 0.0, -DISTANCE, 1.0)
        half, _ = VideoPanel.half_size(p, rgb.shape[1] / rgb.shape[0], entry["fov"])
        expected = (ahead[0] / -ahead[2], ahead[1] / -ahead[2], 2 * half / -ahead[2])
        return measured, expected, 2 / (p[0, 0] * width)

    disparities = []
    for pair in stereo:
        assert all(e["seen"] and e["box"] for e in pair), "panel missing from an eye"
        assert all(e["fov"] is not None for e in pair), "camera field of view never arrived"
        results = []
        for entry in pair:
            measured, expected, pixel = tangents(entry)
            for got, want in zip(measured, expected, strict=True):
                assert abs(got - want) <= 2 * pixel, (measured, expected, entry["box"])
            results.append(measured)
        (left_x, _, left_width), (right_x, _, right_width) = results
        separation = np.linalg.norm(pair[0]["eye"][:3, 3] - pair[1]["eye"][:3, 3])
        # Left eye sees it right of its forward, right eye left; together half the eye
        # separation over the distance, and the same size in both.
        assert left_x > 0 > right_x, (left_x, right_x)
        disparity = left_x - right_x
        assert abs(disparity - separation / DISTANCE) < 4 * pixel, (disparity, separation)
        assert abs(left_width - right_width) < 2 * pixel, (left_width, right_width)
        disparities.append(float(disparity))
    report = {
        "stereo_frames": len(stereo),
        "eye_panel_box_px": [int(v) for v in stereo[0][0]["seen"]],
        "eye_separation_m": float(separation),
        "disparity_tangent_median": float(np.median(disparities)),
        "expected_disparity_tangent": float(separation / DISTANCE),
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

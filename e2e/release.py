"""Unzip the Windows release on a clean environment and pilot it like a user would.

From a checkout on the Windows PC, inside the logged-in desktop session (the windows
must be real), with output redirected to a file:

    uv run python e2e/release.py [--zip dist\\ito-windows-x64.zip] [--target DIR] > log 2>&1

No Python, uv, venv or CUDA Toolkit is visible to the app. It checks: ito.exe opens the
connect screen without a console; "Try simulated robot" shows a live RGB-D room; an
RGB-only bundled robot is reconstructed by MASt3R-SLAM from the bundled models; and
without models\\model.safetensors the pilot gets one plain error and the flat camera feed.
Screenshots, metrics and logs go to e2e/out/release.
"""

import argparse
import ctypes
import json
import os
import shutil
import socket
import subprocess
import time
import zipfile
from ctypes import wintypes
from pathlib import Path

import numpy as np
import psutil
import pygame

from ito import clock

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "e2e/out/release"
user32, gdi32, kernel32 = ctypes.windll.user32, ctypes.windll.gdi32, ctypes.windll.kernel32
HANDLE = wintypes.HANDLE
for function, arguments, result in (
    (user32.GetDC, [HANDLE], HANDLE),
    (user32.ReleaseDC, [HANDLE, HANDLE], ctypes.c_int),
    (user32.GetClientRect, [HANDLE, ctypes.POINTER(wintypes.RECT)], wintypes.BOOL),
    (user32.PrintWindow, [HANDLE, HANDLE, wintypes.UINT], wintypes.BOOL),
    (user32.ClientToScreen, [HANDLE, ctypes.POINTER(wintypes.POINT)], wintypes.BOOL),
    (user32.PostMessageW, [HANDLE, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM], wintypes.BOOL),
    (gdi32.CreateCompatibleDC, [HANDLE], HANDLE),
    (gdi32.CreateCompatibleBitmap, [HANDLE, ctypes.c_int, ctypes.c_int], HANDLE),
    (gdi32.SelectObject, [HANDLE, HANDLE], HANDLE),
    (
        gdi32.GetDIBits,
        [HANDLE, HANDLE] + [wintypes.UINT] * 2 + [ctypes.c_void_p] * 2 + [wintypes.UINT],
        ctypes.c_int,
    ),
    (gdi32.DeleteObject, [HANDLE], wintypes.BOOL),
    (gdi32.DeleteDC, [HANDLE], wintypes.BOOL),
):
    function.argtypes, function.restype = arguments, result
WM_CLOSE, WM_MOUSEMOVE, WM_LBUTTONDOWN, WM_LBUTTONUP = 0x10, 0x200, 0x201, 0x202


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def clean_environment(test):
    """What a pilot's PC has: Windows and the NVIDIA driver, nothing from development."""
    system = os.environ["SystemRoot"]
    hidden = ("PYTHON", "UV_", "VIRTUAL_ENV", "CONDA", "CUDA_PATH", "NVTOOLSEXT")
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(hidden)}
    env["PATH"] = ";".join(
        (rf"{system}\System32", system, rf"{system}\System32\Wbem", rf"{system}\System32\OpenSSH")
    )
    # Keep everything the app writes inside the test folder.
    env |= {
        "APPDATA": str(test / "appdata"),
        "CUPY_CACHE_DIR": str(test / "cupy-cache"),
        "HF_HOME": str(test / "huggingface"),
    }
    return env


def windows(pids):
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _):
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        title = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, title, 256)
        if pid.value in pids and user32.IsWindowVisible(hwnd) and title.value.startswith("Ito"):
            found.append(hwnd)
        return True

    user32.EnumWindows(visit, 0)
    return found


class App:
    running = []

    def __init__(self, folder, env, name, *args):
        self.name = name
        self.metrics = OUT / f"{name}.jsonl"
        self.metrics.unlink(missing_ok=True)
        log(f"{name}: ito.exe {' '.join(args)}")
        self.process = subprocess.Popen(
            [str(folder / "ito.exe"), *args, "--metrics", str(self.metrics)],
            cwd=folder,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        App.running.append(self.process)
        self.window = None
        deadline = clock.now() + 120
        while self.window is None:
            self.alive()
            assert clock.now() < deadline, f"{name}: no Ito window"
            found = windows({p.pid for p in self.tree()})
            self.window = found[0] if found else None
            time.sleep(0.5)

    def tree(self):
        try:
            return [
                psutil.Process(self.process.pid),
                *psutil.Process(self.process.pid).children(True),
            ]
        except psutil.NoSuchProcess:
            return []

    def alive(self):
        assert self.process.poll() is None, (
            f"{self.name}: ito.exe exited ({self.process.returncode})"
        )

    def rows(self):
        try:
            text = self.metrics.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        return [json.loads(line) for line in text.splitlines() if line.endswith("}")]

    def wait(self, label, condition, timeout, *, failed=lambda row: False):
        deadline = clock.now() + timeout
        while True:
            self.alive()
            rows = self.rows()
            assert not (rows and failed(rows[-1])), f"{self.name}: {rows[-1]['reconstruction']}"
            if rows and condition(rows):
                log(f"{self.name}: {label}")
                return rows
            if clock.now() > deadline:
                last = rows[-1] if rows else None
                raise AssertionError(f"{self.name}: timed out waiting for {label}; last {last}")
            time.sleep(1)

    def screenshot(self, name):
        """What the pilot sees: the window's own pixels, composited by Windows."""
        rect = wintypes.RECT()
        user32.GetClientRect(self.window, ctypes.byref(rect))
        width, height = rect.right, rect.bottom
        screen = user32.GetDC(None)
        memory = gdi32.CreateCompatibleDC(screen)
        bitmap = gdi32.CreateCompatibleBitmap(screen, width, height)
        gdi32.SelectObject(memory, bitmap)
        user32.PrintWindow(self.window, memory, 3)  # PW_CLIENTONLY | PW_RENDERFULLCONTENT
        header = (ctypes.c_uint32 * 10)(40, width, -height, 1 | (32 << 16), 0, 0, 0, 0, 0, 0)
        pixels = ctypes.create_string_buffer(width * height * 4)
        gdi32.GetDIBits(memory, bitmap, 0, height, pixels, header, 0)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(memory)
        user32.ReleaseDC(None, screen)
        image = np.frombuffer(pixels.raw, np.uint8).reshape(height, width, 4)[:, :, 2::-1]
        surface = pygame.image.frombuffer(
            np.ascontiguousarray(image).tobytes(), (width, height), "RGB"
        )
        pygame.image.save(surface, OUT / f"{name}.png")
        return image.astype(float)

    def click(self, x, y):
        # The real cursor has to be there too, or SDL sees the pointer leave the window.
        point = wintypes.POINT(int(x), int(y))
        user32.ClientToScreen(self.window, ctypes.byref(point))
        user32.SetCursorPos(point.x, point.y)
        time.sleep(0.2)
        position = (int(y) << 16) | int(x)
        for message, buttons in ((WM_MOUSEMOVE, 0), (WM_LBUTTONDOWN, 1), (WM_LBUTTONUP, 0)):
            user32.PostMessageW(self.window, message, buttons, position)
            time.sleep(0.15)

    def close(self):
        """Close the window like the pilot does; the app must exit cleanly with its children."""
        user32.PostMessageW(self.window, WM_CLOSE, 0, 0)
        try:
            code = self.process.wait(30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            raise AssertionError(f"{self.name}: did not exit after closing its window") from None
        assert code == 0, f"{self.name}: exit code {code}"
        log(f"{self.name}: closed, exit 0")


def simulated_robot_button(image):
    """The lowest button on the connect panel, found by colour so DPI and fonts do not matter."""
    column = image[:, image.shape[1] // 2]
    blue = (column[:, 2] > 90) & (column[:, 2] > column[:, 0] + 40)
    rows = np.flatnonzero(blue)
    assert len(rows), "No connect panel buttons visible"
    bottom = rows[-1]
    top = bottom
    while top - 1 in rows:
        top -= 1
    return image.shape[1] // 2, (top + bottom) / 2


def live(rows, *, flat, after=0.0):
    recent = [r for r in rows if r["time"] >= after and r["link"] == "CONNECTED"]
    return len(recent) >= 6 and all(r["flat_video"] == flat for r in recent[-6:])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", type=Path, default=ROOT / "dist/ito-windows-x64.zip")
    parser.add_argument("--target", type=Path, default=OUT / "unzipped")
    parser.add_argument("--slam-timeout", type=float, default=600)
    args = parser.parse_args()
    # A double-clicked ito.exe has no console to attach to; neither will ours.
    kernel32.FreeConsole()
    if OUT.exists():
        shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.target.exists():
        shutil.rmtree(args.target)
    log(f"Unzipping {args.zip} into {args.target}")
    with zipfile.ZipFile(args.zip) as archive:
        archive.extractall(args.target)
    folder = args.target / "ito"
    assert {p.name for p in folder.iterdir()} == {
        "ito.exe",
        "runtime",
        "models",
        "THIRD_PARTY_NOTICES.txt",
    }
    notices = (folder / "THIRD_PARTY_NOTICES.txt").read_text(encoding="utf-8")
    for shipped in ("\nav ", "\ntorch ", "pygame/docs/generated/LGPL.txt", "MASt3R"):
        assert shipped in notices, f"THIRD_PARTY_NOTICES.txt lacks {shipped.strip()}"
    env = clean_environment(args.target)
    try:
        pilot(folder, args, env)
    finally:
        # ito.exe holds its app, worker and simulated robot in a job that dies with it.
        for process in App.running:
            if process.poll() is None:
                process.kill()


def pilot(folder, args, env):
    results = {}

    # 1. Double-click: the connect screen, then "Try simulated robot" (posed RGB-D).
    app = App(folder, env, "simulated")
    time.sleep(3)
    connect = app.screenshot("1-connect")
    assert connect.std() > 5, "Blank connect screen"
    names = {p.name().lower() for p in app.tree()}
    assert "python.exe" not in names and "pythonw.exe" in names, names
    app.click(*simulated_robot_button(connect))
    rows = app.wait(
        "live RGB-D room",
        lambda rows: (
            live(rows, flat=False)
            and rows[-1]["gaussians"] > 1000
            and rows[-1]["reconstruction"].startswith("Posed RGB-D")
        ),
        180,
    )
    started = rows[-1]["scene_capture_time"]
    time.sleep(3)
    rows = app.rows()
    assert rows[-1]["scene_capture_time"] > started, "Scene stopped updating"
    assert rows[-1]["gaussians"] > 1000 and not rows[-1]["flat_video"]
    room = app.screenshot("2-simulated-rgbd")
    assert room.std() > 15, "Simulated room is blank"
    sims = [p for p in app.tree() if "drivers.mujoco.cli" in " ".join(p.cmdline())]
    assert sims and all(p.name().lower() == "pythonw.exe" for p in sims), sims
    results["rgbd"] = {
        "gaussians": rows[-1]["gaussians"],
        "capture_to_splat_visible_ms": rows[-1]["capture_to_splat_visible_ms"],
        "fps": rows[-1]["fps"],
        "renderer": rows[-1]["renderer"],
    }
    app.close()
    assert not any(p.is_running() for p in sims), "Simulated robot outlived the app"
    assert not (args.target / "appdata/ito/recent.json").exists(), "Simulated robot remembered"

    # 2. An RGB-only robot: MASt3R-SLAM from the bundled models, then without them.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    code_file = args.target / "pairing-code"
    code_file.write_text("135790\n")
    with (OUT / "rgb-only-robot.log").open("w") as robot_log:
        robot = subprocess.Popen(
            [str(folder / "runtime/python.exe"), "-I", "-m", "drivers.mujoco.cli"]
            + ["--rgb-only", "--port", str(port), "--pairing-file", str(code_file)],
            env=env,
            stdout=robot_log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    try:
        app = App(folder, env, "slam", f"127.0.0.1:{port}", "--code", "135790")
        rows = app.wait(
            "MASt3R-SLAM splats",
            lambda rows: live(rows, flat=False) and rows[-1]["gaussians"] > 500,
            args.slam_timeout,
            failed=lambda row: (
                row["reconstruction"].endswith("showing flat camera feed")
                and "paused" not in row["reconstruction"]
            ),
        )
        slam = app.screenshot("3-slam")
        assert slam.std() > 15, "SLAM scene is blank"
        results["slam"] = {
            "gaussians": rows[-1]["gaussians"],
            "status": rows[-1]["reconstruction"],
            "first_live_after_s": rows[-1]["time"] - rows[0]["time"],
        }
        app.close()
        recent = json.loads((args.target / "appdata/ito/recent.json").read_text())
        assert recent[0]["address"] == f"127.0.0.1:{port}", recent

        (folder / "models/model.safetensors").unlink()
        app = App(folder, env, "missing-model", f"127.0.0.1:{port}")
        expected = (
            f"MASt3R model missing from {folder / 'models/model.safetensors'}"
            "; showing flat camera feed"
        )
        rows = app.wait(
            "plain error with the flat camera feed",
            lambda rows: live(rows, flat=True) and rows[-1]["reconstruction"] == expected,
            180,
        )
        shown = rows[-1]["video_capture_time"]
        time.sleep(3)
        rows = app.rows()
        assert rows[-1]["video_capture_time"] > shown, "Flat camera feed is frozen"
        assert {r["reconstruction"] for r in rows[-5:]} == {expected}
        feed = app.screenshot("4-missing-model")
        assert feed.std() > 15, "Flat camera feed is blank"
        results["missing_model"] = {"status": expected}
        app.close()
    finally:
        robot.terminate()
        robot.wait(10)
    log(json.dumps(results, indent=2))
    (OUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    log("PASS: unzipped release, connect screen, simulated RGB-D room, bundled SLAM, missing model")


if __name__ == "__main__":
    main()

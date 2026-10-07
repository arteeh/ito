"""Live shared-memory producer + SDL gamepad: xvfb-run -a uv run python e2e/stream.py."""

import ctypes
import json
import multiprocessing as mp
import time
from contextlib import closing
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np
import pygame
from sample_scene import write_scene

from ito import clock
from ito.desktop import DesktopState, DesktopWindow, PilotStatus
from ito.render import GaussianBuffer, GaussianFrame, load_ply, pose

OUTPUT = Path("e2e/out/stream")


def produce(name, shape, path, connection):
    with closing(SharedMemory(name=name)) as memory:
        shared = np.ndarray(shape, dtype=np.float32, buffer=memory.buf)
        original = load_ply(path).records
        for revision in range(2):
            shared[:] = original
            shared[:, 0, 0] += revision * 0.3
            connection.send((revision, clock.now()))
            assert connection.poll(20), "Display did not acknowledge scene upload"
            connection.recv()
            time.sleep(1.5)  # Deliberately slower than the display and pilot input.
        connection.close()


class SharedSource:
    def __init__(self, memory, shape, connection):
        self.records = np.ndarray(shape, dtype=np.float32, buffer=memory.buf)
        self.connection = connection
        self.ack = False
        self.revisions = []

    def poll(self):
        if self.connection is None:
            return None
        if self.ack:
            self.connection.send("uploaded")
            self.ack = False
        if self.connection.poll():
            try:
                revision, captured_at = self.connection.recv()
            except EOFError:
                self.connection.close()
                self.connection = None
                return None
            self.revisions.append(revision)
            self.ack = True
            return GaussianFrame(GaussianBuffer(self.records), revision, captured_at)
        return None


class VirtualPad:
    def __init__(self):
        # Load the same SDL instance as pygame, so its real controller API sees the device.
        libraries = Path(pygame.__file__).parent.parent / "pygame_ce.libs"
        self.sdl = ctypes.CDLL(str(next(libraries.glob("libSDL2-2-*"))))
        for name, args, result in (
            ("SDL_JoystickAttachVirtual", [ctypes.c_int] * 4, ctypes.c_int),
            ("SDL_JoystickOpen", [ctypes.c_int], ctypes.c_void_p),
            (
                "SDL_JoystickSetVirtualAxis",
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_int16],
                ctypes.c_int,
            ),
            (
                "SDL_JoystickSetVirtualButton",
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint8],
                ctypes.c_int,
            ),
            ("SDL_JoystickClose", [ctypes.c_void_p], None),
            ("SDL_JoystickDetachVirtual", [ctypes.c_int], ctypes.c_int),
        ):
            function = getattr(self.sdl, name)
            function.argtypes, function.restype = args, result
        self.index = self.sdl.SDL_JoystickAttachVirtual(1, 6, 15, 0)
        assert self.index >= 0
        self.handle = self.sdl.SDL_JoystickOpen(self.index)
        assert self.handle

    def axis(self, axis, value):
        assert self.sdl.SDL_JoystickSetVirtualAxis(self.handle, axis, value) == 0

    def button(self, button, value):
        assert self.sdl.SDL_JoystickSetVirtualButton(self.handle, button, value) == 0

    def close(self):
        if self.handle:
            self.sdl.SDL_JoystickClose(self.handle)
            assert self.sdl.SDL_JoystickDetachVirtual(self.index) == 0
            self.handle = None


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scene = OUTPUT / "scene.ply"
    write_scene(scene)
    shape = load_ply(scene).records.shape
    context = mp.get_context("spawn")
    reader, writer = context.Pipe()
    memory = SharedMemory(create=True, size=int(np.prod(shape)) * 4)
    process = context.Process(target=produce, args=(memory.name, shape, scene, writer))
    process.start()
    writer.close()
    source = SharedSource(memory, shape, reader)
    history = []
    pad = None
    estop = False
    ticks = 0

    def receive(pilot):
        nonlocal pad, estop, ticks
        history.append(pilot)
        ticks += 1
        for command in pilot.commands:
            if command == "e_stop":
                estop = True
            elif command == "resume":
                estop = False
        if ticks == 15:
            pad = VirtualPad()
        elif ticks == 25:
            pad.axis(1, -20000)
            pad.axis(2, 12000)
            pad.button(10, 1)
        elif ticks == 45:
            pad.button(1, 1)
        elif ticks == 55:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F12))
        elif ticks == 56:
            pygame.event.post(pygame.event.Event(pygame.KEYUP, key=pygame.K_F12))
        elif ticks == 60:
            pad.button(1, 0)
        elif ticks == 65:
            pad.button(0, 1)
        elif ticks == 75:
            pad.button(0, 0)
            pad.axis(2, 0)
            pad.button(10, 0)
        elif ticks == 85:
            pygame.event.post(pygame.event.Event(pygame.WINDOWFOCUSLOST))
        elif ticks == 100:
            pygame.event.post(pygame.event.Event(pygame.WINDOWFOCUSGAINED))
        elif ticks == 120:
            pad.close()
        elif ticks == 175:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F12))
        elif ticks == 176:
            pygame.event.post(pygame.event.Event(pygame.KEYUP, key=pygame.K_F12))

    def state():
        return DesktopState(
            pose((0.15 * np.sin(ticks / 40), 0, 0)),
            PilotStatus("CONNECTED", 18, "E2E driver", estop),
        )

    try:
        with (
            (OUTPUT / "metrics.jsonl").open("w") as metrics,
            DesktopWindow((800, 600), fps=60, capture_dir=OUTPUT) as window,
        ):
            try:
                window.run(source, state=state, on_input=receive, max_frames=210, metrics=metrics)
            finally:
                if pad is not None:
                    pad.close()
        process.join(timeout=5)
        assert process.exitcode == 0, process.exitcode
        assert source.revisions == [0, 1], source.revisions
        commands = [command for pilot in history for command in pilot.commands]
        assert commands.count("e_stop") == 1 and commands.count("resume") == 1, commands
        assert commands.count("stop") >= 2, commands
        assert history[50].head[2, 3] < -0.1 and history[50].head[1, 3] > 0.1
        assert abs(history[50].head[0, 2]) > 0.1
        assert all(pilot.movement == (0, 0, 0) and not pilot.active for pilot in history[86:99])
        assert all(pilot.movement == (0, 0, 0) for pilot in history[122:])
        rows = [json.loads(line) for line in (OUTPUT / "metrics.jsonl").read_text().splitlines()]
        assert sum(row["revision"] == 0 for row in rows) >= 2, (
            "Did not render during producer stall"
        )
        assert any(row["revision"] == 1 for row in rows)
        print(
            "PASS: shared-memory producer stalls/updates, fresh anchor, "
            "virtual gamepad axes/buttons, focus loss, hot unplug"
        )
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        reader.close()
        memory.close()
        memory.unlink()


if __name__ == "__main__":
    main()

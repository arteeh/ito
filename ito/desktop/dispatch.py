"""SDL input stays on the window thread; GL work owns the context on a worker."""

import ctypes
import sys
import threading
from dataclasses import replace
from pathlib import Path

import pygame
from imgui_bundle import imgui

from ito import clock


class DisplayDispatch:
    def __init__(self, window):
        self.window = window
        library = (
            str(Path(pygame.__file__).parent / "SDL2.dll")
            if sys.platform == "win32"
            else pygame.base.__file__
        )
        self.sdl = ctypes.CDLL(library)
        self.sdl.SDL_GL_GetCurrentWindow.restype = ctypes.c_void_p
        self.sdl.SDL_GL_GetCurrentContext.restype = ctypes.c_void_p
        self.sdl.SDL_GL_MakeCurrent.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        self.handle = self.sdl.SDL_GL_GetCurrentWindow()
        self.context = self.sdl.SDL_GL_GetCurrentContext()
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.events = []
        self.value = None
        self.commands = []
        self.screenshot = self.quit = False
        self.keyboard_ui = self.capture_requested = False
        self.ui_commands = []
        self.failure = None

    def bind(self, context):
        if self.sdl.SDL_GL_MakeCurrent(self.handle, context) != 0:
            raise RuntimeError("Could not transfer the SDL OpenGL context")

    def sample(self, dt, on_sample):
        events = pygame.event.get()
        with self.lock:
            keyboard_ui = self.keyboard_ui
            capture, self.capture_requested = self.capture_requested, False
            commands, self.ui_commands = self.ui_commands, []
        if capture and self.window.input.active:
            self.window.input.capture(True)
            if self.window.input.refocus:
                # Clicking back into the scene asks to undo a stop that only focus loss caused.
                self.window.input.refocus = False
                commands.append("rearm")
        value = self.poll(dt, events, keyboard_ui, commands)
        if on_sample:
            on_sample(value)
        with self.lock:
            # A wedged display must not grow an unbounded event/command queue.
            if len(self.events) + len(events) > 4096 or len(self.commands) > 256:
                raise RuntimeError("Display stopped consuming pilot input")
            self.events.extend(events)
            self.value = value
            self.commands.extend(value.commands)
            self.screenshot |= value.screenshot
            self.quit |= value.quit

    def poll(self, dt, events, keyboard_ui, commands):
        value = self.window.input.poll(
            dt,
            events,
            # ImGui must classify uncaptured clicks before entering mouse-look.
            # Already captured motion and safety keys still poll independently.
            mouse_ui=True,
            keyboard_ui=keyboard_ui and not self.window.input.captured,
        )
        return replace(value, commands=value.commands + tuple(commands))

    def frame(self):
        with self.lock:
            events, self.events = self.events, []
            value = replace(
                self.value,
                commands=tuple(self.commands),
                screenshot=self.screenshot,
                quit=self.quit,
            )
            self.commands.clear()
            self.screenshot = False
        return events, value

    def ui(self, io, commands, events=()):
        with self.lock:
            self.keyboard_ui = io.want_capture_keyboard
            self.ui_commands.extend(commands)
            if not io.want_capture_mouse and not self.window.input.captured:
                self.capture_requested |= any(
                    event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 for event in events
                )

    def run(self, draw, on_sample):
        self.sample(0, on_sample)
        self.bind(None)

        def render():
            try:
                self.bind(self.context)
                imgui.set_current_context(self.window.overlay.imgui)
                draw()
            except BaseException as exc:
                self.failure = exc
            finally:
                try:
                    self.bind(None)
                finally:
                    self.done.set()

        thread = threading.Thread(target=render, name="ito-display")
        previous = clock.now()
        thread.start()
        try:
            while not self.done.wait(1 / 90):
                now = clock.now()
                self.sample(now - previous, on_sample)
                previous = now
        finally:
            with self.lock:
                self.quit = True
            # Closing input always disarms, including exceptions during polling.
            if on_sample:
                on_sample(replace(self.value, active=False, commands=("stop",)))
            thread.join()
            self.bind(self.context)
            imgui.set_current_context(self.window.overlay.imgui)
        if self.failure:
            raise self.failure

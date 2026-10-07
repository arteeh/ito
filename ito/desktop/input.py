"""Desktop tracking and input, using SDL's standardized controller mappings."""

import math
from dataclasses import dataclass, field

import numpy as np
import pygame
from pygame._sdl2 import controller

from ito import clock, diagnostics
from ito.protocol import Pose
from ito.render import pose
from ito.render.scene import FloatArray


@dataclass(frozen=True)
class PilotInput:
    timestamp: float
    head: FloatArray
    movement: tuple[float, float, float]  # right, up, forward; normalized [-1, 1]
    look: tuple[float, float]  # right/up stick
    buttons: frozenset[str]
    commands: tuple[str, ...]  # stop, e_stop, resume; edge triggered
    active: bool
    quit: bool = False
    screenshot: bool = False
    hands: dict[str, Pose] = field(default_factory=dict)
    trackers: dict[str, Pose] = field(default_factory=dict)
    axes: dict[str, float] = field(default_factory=dict)


class DesktopInput:
    def __init__(self, speed: float = 1.5):
        self.speed = speed
        self.position = np.zeros(3, dtype=np.float32)
        self.yaw = self.pitch = 0.0
        self.captured = False
        self.active = True
        self.translate = True
        self.sensitivity = 0.0025
        self.invert_y = False
        self.keys: set[int] = set()
        self.pad = None
        self.pad_buttons: set[str] = set()
        controller.init()
        self._connect_pad()

    def _connect_pad(self) -> None:
        if self.pad is not None and self.pad.attached():
            return
        if self.pad is not None:
            self.pad.quit()
        self.pad = None
        for index in range(controller.get_count()):
            if controller.is_controller(index):
                self.pad = controller.Controller(index)
                break

    def capture(self, enabled: bool) -> None:
        if enabled != self.captured:
            diagnostics.event("input_capture", captured=enabled)
        self.captured = enabled
        pygame.event.set_grab(enabled)
        pygame.mouse.set_visible(not enabled)
        pygame.mouse.set_relative_mode(enabled)
        pygame.mouse.get_rel()

    @staticmethod
    def _stick(x: int, y: int) -> tuple[float, float]:
        vector = np.array((x, y), dtype=float) / 32768
        length = float(np.linalg.norm(vector))
        if length <= 0.15:
            return 0.0, 0.0
        vector *= min(1, (length - 0.15) / 0.85) / length
        return float(vector[0]), float(vector[1])

    def poll(self, dt: float, events=None, *, mouse_ui=False, keyboard_ui=False) -> PilotInput:
        commands: list[str] = []
        quit_requested = screenshot = False
        if keyboard_ui:
            self.keys.clear()
        for event in pygame.event.get() if events is None else events:
            if keyboard_ui and event.type == pygame.KEYDOWN and event.key != pygame.K_e:
                continue
            if (
                mouse_ui
                and not self.captured
                and event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION)
            ):
                continue
            if event.type == pygame.QUIT:
                quit_requested = True
                commands.append("stop")
            elif event.type == pygame.WINDOWFOCUSLOST:
                self.active = False
                self.keys.clear()
                self.pad_buttons.clear()
                self.capture(False)
                commands.append("stop")
            elif event.type == pygame.WINDOWFOCUSGAINED:
                self.active = True
            elif event.type == pygame.KEYDOWN and self.active:
                fresh = event.key not in self.keys
                self.keys.add(event.key)
                if fresh:
                    if event.key == pygame.K_ESCAPE:
                        self.capture(False)
                    elif event.key == pygame.K_TAB:
                        self.capture(not self.captured)
                    elif event.key == pygame.K_F12:
                        screenshot = True
                    elif event.key in (pygame.K_SPACE, pygame.K_e, pygame.K_r):
                        commands.append(
                            {pygame.K_SPACE: "stop", pygame.K_e: "e_stop", pygame.K_r: "resume"}[
                                event.key
                            ]
                        )
                    elif event.key == pygame.K_HOME:
                        self.position[:] = 0
                        self.yaw = self.pitch = 0.0
            elif event.type == pygame.KEYUP:
                self.keys.discard(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.active:
                self.capture(True)
            elif event.type == pygame.MOUSEMOTION and self.captured and self.active:
                self.yaw -= event.rel[0] * self.sensitivity
                self.pitch -= event.rel[1] * self.sensitivity * (-1 if self.invert_y else 1)
            elif event.type in (pygame.CONTROLLERDEVICEADDED, pygame.CONTROLLERDEVICEREMOVED):
                if self.pad is not None and not self.pad.attached():
                    commands.append("stop")
                    self.pad_buttons.clear()
                self._connect_pad()

        movement = np.zeros(3, dtype=float)
        look = (0.0, 0.0)
        buttons: set[str] = set()
        if self.active:
            movement[:] = (
                int(pygame.K_d in self.keys) - int(pygame.K_a in self.keys),
                int(pygame.K_PAGEUP in self.keys) - int(pygame.K_PAGEDOWN in self.keys),
                int(pygame.K_w in self.keys) - int(pygame.K_s in self.keys),
            )
            if self.pad is not None and self.pad.attached():
                left = self._stick(
                    self.pad.get_axis(pygame.CONTROLLER_AXIS_LEFTX),
                    self.pad.get_axis(pygame.CONTROLLER_AXIS_LEFTY),
                )
                look = self._stick(
                    self.pad.get_axis(pygame.CONTROLLER_AXIS_RIGHTX),
                    -self.pad.get_axis(pygame.CONTROLLER_AXIS_RIGHTY),
                )
                movement += (
                    left[0],
                    int(self.pad.get_button(pygame.CONTROLLER_BUTTON_RIGHTSHOULDER))
                    - int(self.pad.get_button(pygame.CONTROLLER_BUTTON_LEFTSHOULDER)),
                    -left[1],
                )
                for name, button in (
                    ("a", pygame.CONTROLLER_BUTTON_A),
                    ("b", pygame.CONTROLLER_BUTTON_B),
                    ("x", pygame.CONTROLLER_BUTTON_X),
                    ("y", pygame.CONTROLLER_BUTTON_Y),
                    ("start", pygame.CONTROLLER_BUTTON_START),
                    ("back", pygame.CONTROLLER_BUTTON_BACK),
                ):
                    if self.pad.get_button(button):
                        buttons.add(name)
                for button, command in (("a", "resume"), ("b", "e_stop"), ("x", "stop")):
                    if button in buttons - self.pad_buttons:
                        commands.append(command)
            self.pad_buttons = buttons.copy()
        movement /= max(1, float(np.linalg.norm(movement)))
        dt = min(max(dt, 0), 0.05)  # A window stall must never cause a camera teleport.
        self.yaw -= look[0] * 1.8 * dt
        self.pitch = float(
            np.clip(self.pitch + look[1] * 1.8 * dt, -math.pi * 0.49, math.pi * 0.49)
        )
        self.yaw = math.remainder(self.yaw, 2 * math.pi)
        rotation = pose(yaw=self.yaw)[:3, :3]
        local = np.array((movement[0], movement[1], -movement[2]))
        if self.translate:
            self.position += rotation @ local * self.speed * dt
        buttons.update(pygame.key.name(key) for key in self.keys)
        return PilotInput(
            clock.now(),
            pose(self.position, self.yaw, self.pitch),
            tuple(map(float, movement)),
            look,
            frozenset(buttons),
            tuple(commands),
            self.active,
            quit_requested,
            screenshot,
        )

    def close(self) -> None:
        self.capture(False)
        if self.pad is not None:
            self.pad.quit()
        controller.quit()

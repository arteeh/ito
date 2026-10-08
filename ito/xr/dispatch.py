"""Sample XR on the session-creation thread, independently of GPU and frame waits."""

from dataclasses import replace

import numpy as np
import pygame

from ito import clock
from ito.desktop.dispatch import DisplayDispatch


class XRDispatch(DisplayDispatch):
    def __init__(self, window, state):
        super().__init__(window)
        self.state = state
        self.last_status = state().status
        self.was_active = False
        self.recenter = False
        self.pointer = (-10000, -10000, False)
        self.panel_triggers = set()  # Hands whose trigger press began on the panel.

    def poll(self, dt, events, keyboard_ui, commands):
        desktop = super().poll(dt, events, keyboard_ui, commands)
        window = self.window
        owner = window.xr
        timing = owner.clock
        if not owner.running or timing is None:
            return replace(desktop, active=False)
        # Advance the runtime's clock even when xrWaitFrame or GPU work stalls.
        # This is a new tracking query, never a re-timestamped cached input sample.
        at = timing[0] + clock.now_ns() - timing[1]
        changed_space = owner.space_changed and at >= owner.space_changed
        keyboard_center = any(e.type == pygame.KEYDOWN and e.key == pygame.K_HOME for e in events)
        if not window.centered or self.recenter or keyboard_center or changed_space:
            if owner.recenter(at):
                window.centered = True
                window.recenters += 1
                self.recenter = False
                owner.space_changed = False
                desktop = replace(desktop, commands=desktop.commands + ("stop",))
        value = window.actions.poll(at)
        commands = value.commands + desktop.commands
        if "recenter" in commands:
            self.recenter = True
            commands = tuple(c for c in commands if c not in ("recenter", "resume")) + ("stop",)
        if self.was_active and not value.active:
            commands += ("stop",)
        if "e_stop" in commands:
            commands = ("e_stop",)
        self.was_active = value.active
        status = self.state().status
        if (
            (status.e_stop and not self.last_status.e_stop)
            or (self.last_status.link == "CONNECTED" and status.link != "CONNECTED")
            or "e_stop" in commands
        ):
            window.actions.pulse()
        self.last_status = status
        self.pointer = window._pointer() if value.active else (-10000, -10000, False)
        # A trigger that clicks the panel is not a robot input (the Microduck's beak), and
        # neither is the rest of that press after the pointer leaves the panel.
        on_panel = self.pointer[0] >= 0
        axes, buttons = dict(value.axes), set(value.buttons)
        for hand in ("left", "right"):
            name = f"{hand}_trigger"
            if axes.get(name, 0.0) <= 0:
                self.panel_triggers.discard(hand)
            elif on_panel:
                self.panel_triggers.add(hand)
            if hand in self.panel_triggers or on_panel:
                axes[name] = 0.0
                buttons.discard(name)
        return replace(
            value,
            axes=axes,
            buttons=frozenset(buttons),
            commands=commands,
            movement=tuple(
                float(np.clip(a + b, -1, 1))
                for a, b in zip(value.movement, desktop.movement, strict=True)
            ),
            quit=desktop.quit,
            screenshot=desktop.screenshot,
        )

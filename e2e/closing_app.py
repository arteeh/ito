"""The real pilot app, closed with its window's X once a fault is in place; for lifecycle.py.

ITO_E2E_CLOSE=<scenario> ITO_E2E_MARK=<file> python e2e/closing_app.py <ito arguments>

Resumes and holds W, then prints "closed <clock>" the moment it closes the window. Scenarios:
- streaming: driving a live robot, splats arriving.
- sim: the same with the bundled simulated robot, which the app starts and stops.
- connecting: the robot accepts the connection but never answers.
- stuck_link: the link's event loop blocks in teardown joining a thread that never ends, as
  aiortc does joining its non-daemon video decoder thread when decoding has backed up.
- hung_teardown: aiortc's teardown awaits a DTLS transport that never answers, with the
  event loop idle, as seen on Windows after a long SLAM drive (#27).
- reconstruction: the worker process is inside one integration that never returns, as a
  SLAM step or CUDA call that never looks at the stop flag.

This file also runs as the spawned reconstruction worker's main module, so the worker fault
is injected at import.
"""

import os
import sys
import threading
import time
from pathlib import Path

import pygame
from pygame._sdl2 import Window

from ito import clock

SCENARIO = os.environ["ITO_E2E_CLOSE"]
MARK = Path(os.environ["ITO_E2E_MARK"])

if SCENARIO == "reconstruction":
    from ito.reconstruction.rgbd import RGBDBackend

    def integrate_forever(self, *args):
        MARK.touch()
        time.sleep(600)

    RGBDBackend.integrate = integrate_forever

if SCENARIO == "stuck_link":
    from ito.link.peer import Peer

    close = Peer.close

    async def stuck_close(self):
        decoder = threading.Thread(target=time.sleep, args=(600,), name="video-decoder")
        decoder.start()
        decoder.join()  # Blocks the event loop, as aiortc's synchronous decoder join does.
        await close(self)

    Peer.close = stuck_close

if SCENARIO == "hung_teardown":
    import asyncio

    from aiortc.rtcdtlstransport import RTCDtlsTransport

    stop = RTCDtlsTransport.stop

    async def never_stops(self):
        # Connection setup stops unused transports too; only the teardown hangs.
        if closed is None:
            return await stop(self)
        await asyncio.Event().wait()

    RTCDtlsTransport.stop = never_stops


def key(code, kind):
    pygame.event.post(pygame.event.Event(kind, key=code))


def ready(app):
    status = app.state.status
    if SCENARIO == "connecting":
        return status.link == "CONNECTING" and clock.now() - started > 1.5
    # Driving, with the worker, the link and every media task live.
    driving = status.armed and status.robot_state == "active"
    return driving and (MARK.exists() if SCENARIO == "reconstruction" else app.matched_frames > 20)


def close_when_ready(app, window, value):
    global closed, driving
    if not driving and app.matched_frames > 12 and app.telemetry:
        Window.from_display_module().focus()
        key(pygame.K_r, pygame.KEYDOWN)
        key(pygame.K_r, pygame.KEYUP)
        key(pygame.K_w, pygame.KEYDOWN)
        driving = True
    assert clock.now() - started < 80, app.state.status
    if closed is None and ready(app):
        closed = clock.now()
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        print("closed", closed, flush=True)


if __name__ == "__main__":
    from ito.app.__main__ import main

    started, closed, driving = clock.now(), None, False
    sys.exit(main(sys.argv[1:], on_frame=close_when_ready))

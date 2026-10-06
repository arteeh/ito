"""Drain decoded media continuously; a slow display never builds a frame backlog."""

import asyncio
import logging

from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError

log = logging.getLogger(__name__)


class LatestTrack(MediaStreamTrack):
    def __init__(self, source: MediaStreamTrack):
        super().__init__()
        self.kind = source.kind
        self.source = source
        self._frames: asyncio.Queue = asyncio.Queue(maxsize=2)
        self._task = asyncio.create_task(self._drain())

    @property
    def id(self) -> str:
        return self.source.id

    def _put(self, frame) -> None:
        if self._frames.full():
            self._frames.get_nowait()
        self._frames.put_nowait(frame)

    async def _drain(self) -> None:
        try:
            while self.readyState == "live":
                self._put(await self.source.recv())
        except (MediaStreamError, asyncio.CancelledError):
            pass
        except Exception:
            log.exception("media track failed")
        finally:
            self._put(None)

    async def recv(self):
        if self.readyState != "live":
            raise MediaStreamError
        frame = await self._frames.get()
        if frame is None:
            self.stop()
            raise MediaStreamError
        return frame

    def stop(self) -> None:
        super().stop()
        self.source.stop()
        if self._task is not asyncio.current_task():
            self._task.cancel()
        self._put(None)

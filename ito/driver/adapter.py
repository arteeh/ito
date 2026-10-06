from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

from aiortc import MediaStreamTrack

from ito.protocol import FrameMetadata, PilotState, RobotDescription


class Adapter(ABC):
    """Command methods must enqueue hardware I/O without blocking or awaiting it.

    The robot must also enforce its own hardware deadman. media_tracks returns fresh
    tracks per connection (use aiortc's MediaRelay for a persistent camera source).
    Camera track IDs must match description; publish metadata at camera capture time.
    """

    frame_sink: Callable[[FrameMetadata], bool] | None = None

    def publish_frame(self, metadata: FrameMetadata) -> bool:
        return self.frame_sink(metadata) if self.frame_sink else False

    @property
    @abstractmethod
    def description(self) -> RobotDescription: ...

    @abstractmethod
    def media_tracks(self) -> Sequence[MediaStreamTrack]: ...

    @abstractmethod
    def apply(self, state: PilotState) -> None: ...

    @abstractmethod
    def neutral(self) -> None: ...

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    def telemetry(self) -> dict[str, float | bool | str]:
        return {}

    def incoming_audio(self, track: MediaStreamTrack) -> None:
        track.stop()

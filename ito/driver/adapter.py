from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

from aiortc import MediaStreamTrack

from ito.protocol import FrameMetadata, Motion, PilotState, RobotDescription, Sensors


class Adapter(ABC):
    """Command methods must enqueue hardware I/O without blocking or awaiting it.

    The robot must also enforce its own hardware deadman. media_tracks returns fresh
    tracks per connection (use aiortc's MediaRelay for a persistent camera source).
    Camera track IDs must match description; publish metadata at camera capture time.
    A robot that knows its camera's motion publishes it as it is measured, stamped at the
    source, and describes its noise in RobotDescription.motion_noise. Sensors on a camera
    (an IMU, a time-of-flight grid) are published as they arrive; a robot without them, or
    whose sensors fail, publishes none and Ito works from the camera and motion alone.
    """

    frame_sink: Callable[[FrameMetadata], bool] | None = None
    motion_sink: Callable[[Motion], bool] | None = None
    sensor_sink: Callable[[Sensors], bool] | None = None

    def publish_frame(self, metadata: FrameMetadata) -> bool:
        return self.frame_sink(metadata) if self.frame_sink else False

    def publish_motion(self, motion: Motion) -> bool:
        """False when nothing was sent: the caller keeps the samples for its next batch."""
        return self.motion_sink(motion) if self.motion_sink else False

    def publish_sensors(self, sensors: Sensors) -> bool:
        """False when nothing was sent: the caller keeps the samples for its next batch."""
        return self.sensor_sink(sensors) if self.sensor_sink else False

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

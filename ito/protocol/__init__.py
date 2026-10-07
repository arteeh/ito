"""Wire v1: metres, xyzw rotations, and local monotonic seconds for timestamps."""

import base64
import zlib
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

VERSION = 1
MAX_MESSAGE_BYTES = 1_500_000
Number = Annotated[float, Field(allow_inf_nan=False)]
Time = Annotated[Number, Field(ge=0)]
Sequence = Annotated[int, Field(ge=0, le=2**53 - 1)]
Name = Annotated[str, Field(min_length=1, max_length=128)]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Pose(Model):
    """Right-handed: +X right, +Y up, -Z forward; xyzw rotation."""

    position: tuple[Number, Number, Number] = (0.0, 0.0, 0.0)
    orientation: tuple[Number, Number, Number, Number] = (0.0, 0.0, 0.0, 1.0)

    @model_validator(mode="after")
    def normalized(self) -> Self:
        if abs(sum(v * v for v in self.orientation) - 1) > 0.01:
            raise ValueError("orientation must be a unit quaternion")
        return self


class Intrinsics(Model):
    width: int = Field(ge=1, le=8192)
    height: int = Field(ge=1, le=8192)
    fx: Number = Field(gt=0)
    fy: Number = Field(gt=0)
    cx: Number = Field(ge=0)
    cy: Number = Field(ge=0)

    @model_validator(mode="after")
    def image_bounds(self) -> Self:
        if self.cx >= self.width or self.cy >= self.height:
            raise ValueError("principal point must be inside the image")
        return self


class Camera(Model):
    name: Name
    track_id: Name
    intrinsics: Intrinsics
    extrinsics: Pose = Field(default_factory=Pose, description="Camera-to-robot transform")


class DegreeOfFreedom(Model):
    name: Name
    unit: Literal["radians", "metres"]
    minimum: Number
    maximum: Number

    @model_validator(mode="after")
    def bounds(self) -> Self:
        if self.minimum >= self.maximum:
            raise ValueError("degree of freedom requires minimum < maximum")
        return self


class Message(Model):
    version: Literal[1] = VERSION

    @model_validator(mode="after")
    def wire_version(self, info: ValidationInfo) -> Self:
        if info.context and info.context.get("wire") and "version" not in self.model_fields_set:
            raise ValueError("wire message requires an explicit version")
        return self

    @field_validator("version", mode="before")
    @classmethod
    def integer_version(cls, value):
        if type(value) is not int:
            raise ValueError("version must be an integer")
        return value


class RobotDescription(Message):
    type: Literal["robot"] = "robot"
    name: Name
    cameras: tuple[Camera, ...] = Field(default=(), max_length=16)
    capabilities: tuple[Name, ...] = Field(default=(), max_length=64)
    degrees_of_freedom: tuple[DegreeOfFreedom, ...] = Field(default=(), max_length=256)

    @model_validator(mode="after")
    def unique_names(self) -> Self:
        for names in (
            [c.name for c in self.cameras],
            [c.track_id for c in self.cameras],
            list(self.capabilities),
            [d.name for d in self.degrees_of_freedom],
        ):
            if len(set(names)) != len(names):
                raise ValueError("robot identifiers must be unique")
        return self


class Depth(Model):
    """Little-endian uint16 millimetres; zero means unknown. Bounded zlib on the wire."""

    width: int = Field(ge=1, le=2048)
    height: int = Field(ge=1, le=2048)
    encoding: Literal["zlib-u16-mm"] = "zlib-u16-mm"
    data: str = Field(max_length=1_400_000)

    def to_bytes(self) -> bytes:
        try:
            compressed = base64.b64decode(self.data, validate=True)
            decoder = zlib.decompressobj()
            expected = self.width * self.height * 2
            raw = decoder.decompress(compressed, expected + 1)
            if len(raw) != expected or not decoder.eof or decoder.unused_data:
                raise ValueError("depth dimensions or compressed length mismatch")
            return raw
        except (zlib.error, ValueError) as exc:
            raise ValueError("invalid depth payload") from exc

    @model_validator(mode="after")
    def valid_payload(self) -> Self:
        self.to_bytes()
        return self

    @classmethod
    def from_bytes(cls, width: int, height: int, raw: bytes) -> Self:
        return cls(
            width=width, height=height, data=base64.b64encode(zlib.compress(raw)).decode("ascii")
        )


class FrameMetadata(Message):
    type: Literal["frame"] = "frame"
    camera: Name
    sequence: Sequence
    capture_time: Time
    video_pts: Sequence | None = Field(
        default=None,
        description="Video presentation timestamp in 90 kHz ticks, zero at the track's first frame",
    )
    camera_pose: Pose | None = Field(
        default=None, description="Camera-to-world pose; world anchored at robot startup"
    )
    depth: Depth | None = None
    body_yaw: Number = Field(default=0, description="Startup-relative body yaw at exposure")
    head_angles: tuple[Number, Number] | None = Field(
        default=None, description="Measured pan and tilt in radians from this exposure"
    )


class PilotState(Message):
    type: Literal["pilot"] = "pilot"
    sequence: Sequence
    capture_time: Time
    deadman: bool = False
    head: Pose | None = None
    hands: dict[Literal["left", "right"], Pose] = Field(default_factory=dict, max_length=2)
    trackers: dict[Name, Pose] = Field(default_factory=dict, max_length=32)
    buttons: dict[Name, bool] = Field(default_factory=dict, max_length=128)
    axes: dict[Name, Annotated[Number, Field(ge=-1, le=1)]] = Field(
        default_factory=dict, max_length=64
    )


class Command(Message):
    type: Literal["command"] = "command"
    sequence: Sequence
    action: Literal["stop", "e-stop", "resume"]


class Status(Message):
    type: Literal["status"] = "status"
    state: Literal["neutral", "active", "stopped", "e-stopped", "fault"]
    reason: str = Field(max_length=256)
    command_sequence: Sequence | None = None
    rejected_messages: int = Field(default=0, ge=0)
    telemetry: dict[Name, Number | bool | str] = Field(default_factory=dict, max_length=128)

    @model_validator(mode="after")
    def bounded_strings(self) -> Self:
        if any(isinstance(v, str) and len(v) > 256 for v in self.telemetry.values()):
            raise ValueError("telemetry string too long")
        return self


class Ping(Message):
    type: Literal["ping"] = "ping"
    sequence: Sequence
    sent: Time


class Pong(Message):
    type: Literal["pong"] = "pong"
    sequence: Sequence
    sent: Time
    received: Time
    replied: Time

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.replied < self.received:
            raise ValueError("reply predates receipt")
        return self


WireMessage = Annotated[
    RobotDescription | FrameMetadata | PilotState | Command | Status | Ping | Pong,
    Field(discriminator="type"),
]
_adapter = TypeAdapter(WireMessage)


class ProtocolError(ValueError):
    pass


def decode(data: str | bytes) -> WireMessage:
    try:
        if not isinstance(data, str | bytes):
            raise ValueError("message must be JSON text or UTF-8 bytes")
        if len(data) > MAX_MESSAGE_BYTES:
            raise ValueError("message too large")
        if isinstance(data, str) and len(data.encode("utf-8")) > MAX_MESSAGE_BYTES:
            raise ValueError("message too large")
        return _adapter.validate_json(data, context={"wire": True})
    except (ValidationError, ValueError, RecursionError) as exc:
        raise ProtocolError("invalid or unsupported Ito message") from exc


def encode(message: WireMessage) -> str:
    # Revalidate instances: nested dictionaries may have been mutated by an adapter.
    data = message.model_dump_json()
    decode(data)
    return data

"""Wire v4: metres, xyzw rotations, and local monotonic seconds for timestamps."""

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

VERSION = 4
MAX_MESSAGE_BYTES = 1_500_000
Number = Annotated[float, Field(allow_inf_nan=False)]
Time = Annotated[Number, Field(ge=0)]
Sequence = Annotated[int, Field(ge=0, le=2**53 - 1)]
Name = Annotated[str, Field(min_length=1, max_length=128)]
Token = Annotated[str, Field(pattern="^[0-9a-f]{32}$")]


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
    capture_delay: Number = Field(
        default=0,
        ge=0,
        le=1,
        description="How long before its capture_time a frame was exposed, as the driver "
        "estimates it; robot-first tracking refines it against the robot's motion",
    )
    exposure: Number | None = Field(
        default=None,
        ge=0,
        le=1,
        description="How long the sensor integrates a frame, as the driver knows it; None if "
        "it cannot say. Robot-first tracking measures nothing visually in a frame the camera "
        "turned too far during",
    )


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
    version: Literal[4] = VERSION

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


class MotionNoise(Model):
    """How far the robot's own camera pose can be trusted, as one standard deviation.

    Between two samples dt seconds apart that moved the camera d metres, the relative rotation
    is off by about hypot(rotation_deg, rotation_deg_per_s * dt) and the relative translation by
    translation_m + translation_fraction * d. Pitch and roll against gravity are off by at most
    gravity_deg at any time. A robot that is held or has fallen knows far less (MotionSample).
    A sample's pose may hold at a time up to about time_s from its stamp, against the
    exposures of the camera's frames (sensor read-out, a control loop's tick): a camera turning
    at w is then off by a further w * time_s at either end.
    """

    rotation_deg: Number = Field(ge=0, le=90)
    rotation_deg_per_s: Number = Field(ge=0, le=90)
    translation_m: Number = Field(ge=0, le=10)
    translation_fraction: Number = Field(ge=0, le=10)
    gravity_deg: Number = Field(gt=0, le=90)
    time_s: Number = Field(default=0, ge=0, le=0.1)


class RobotDescription(Message):
    type: Literal["robot"] = "robot"
    name: Name
    cameras: tuple[Camera, ...] = Field(default=(), max_length=16)
    capabilities: tuple[Name, ...] = Field(default=(), max_length=64)
    degrees_of_freedom: tuple[DegreeOfFreedom, ...] = Field(default=(), max_length=256)
    motion_noise: MotionNoise | None = Field(
        default=None, description="Set by a robot that streams Motion for its first camera"
    )

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


class MotionSample(Model):
    time: Time = Field(description="Driver clock at the sample's source, not its arrival")
    camera_pose: Pose = Field(
        description=(
            "First camera's camera-to-world; world is the robot's odometry at startup, its y the"
            " height above the floor the robot stands on (it gives reconstruction metric scale)"
        )
    )
    angular_velocity: tuple[Number, Number, Number] = Field(
        description="Camera frame, rad/s; a gyro where the robot has one"
    )
    state: Literal["standing", "walking", "turning", "held", "fallen"]


class Motion(Message):
    """The robot's own camera motion, batched: every sample once, oldest first."""

    type: Literal["motion"] = "motion"
    samples: tuple[MotionSample, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if any(b.time <= a.time for a, b in zip(self.samples, self.samples[1:], strict=False)):
            raise ValueError("motion samples must be in time order")
        return self


class ImuSample(Model):
    """One inertial reading, in the sensor's own right-handed axes: rad/s and m/s² of specific
    force (at rest it reads +g, pointing up)."""

    time: Time = Field(description="Driver clock when the sensor sampled, not its arrival")
    angular_velocity: tuple[Number, Number, Number]
    specific_force: tuple[Number, Number, Number]
    orientation: tuple[Number, Number, Number, Number] | None = Field(
        default=None,
        description="The sensor's own fusion, sensor-to-world xyzw in a +Y-up world whose yaw "
        "is arbitrary (no magnetometer); absent if the chip does not fuse",
    )

    @model_validator(mode="after")
    def unit(self) -> Self:
        if self.orientation is not None and abs(sum(v * v for v in self.orientation) - 1) > 0.01:
            raise ValueError("IMU orientation must be a unit quaternion")
        return self


class Imu(Model):
    """A camera-mounted IMU's samples since the last batch, oldest first."""

    sensor: Name
    mount: Pose = Field(description="Sensor-to-camera transform")
    gyro_sigma: Number = Field(gt=0, description="One sigma of each rate, rad/s, bias included")
    accel_sigma: Number = Field(gt=0, description="One sigma of each axis, m/s², bias included")
    samples: tuple[ImuSample, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if any(b.time <= a.time for a, b in zip(self.samples, self.samples[1:], strict=False)):
            raise ValueError("IMU samples must be in time order")
        return self


class RangeGrid(Model):
    """One frame of a multizone time-of-flight sensor: rows x cols zones evenly dividing its
    field of view, row 0 at the top and column 0 at the left as the sensor looks out along its
    -Z (+X right, +Y up, like a camera). Each distance is the range along its zone's centre
    ray, in metres, or None where the zone measured nothing usable."""

    sensor: Name
    time: Time = Field(description="Driver clock when the sensor ranged, not its arrival")
    mount: Pose = Field(description="Sensor-to-camera transform")
    rows: int = Field(ge=1, le=64)
    cols: int = Field(ge=1, le=64)
    field_of_view: tuple[Number, Number] = Field(description="Full width and height, radians")
    distances: tuple[Annotated[Number, Field(gt=0, le=100)] | None, ...]
    sigma: tuple[Number, Number] = Field(
        description="Range noise, one sigma: metres plus a fraction of the distance"
    )

    @model_validator(mode="after")
    def shape(self) -> Self:
        if len(self.distances) != self.rows * self.cols:
            raise ValueError("a range grid has one distance per zone")
        if any(not 0 < a < 3.1416 for a in self.field_of_view):
            raise ValueError("range field of view must be within (0, pi)")
        if self.sigma[0] < 0 or self.sigma[1] < 0:
            raise ValueError("range noise must be nonnegative")
        return self


class Sensors(Message):
    """What a robot's camera-mounted sensors measured, beside its motion: optional, so a
    robot without them sends none and Ito works from the camera and motion alone."""

    type: Literal["sensors"] = "sensors"
    camera: Name = Field(description="The described camera the mounts are relative to")
    imu: tuple[Imu, ...] = Field(default=(), max_length=4)
    ranges: tuple[RangeGrid, ...] = Field(default=(), max_length=8)


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


class Credential(Message):
    """The driver's answer to a correct pairing code: a 128-bit secret for this pilot alone.

    It only travels over the DTLS-encrypted data channel. Later offers prove the secret
    instead of the guessable six-digit code.
    """

    type: Literal["credential"] = "credential"
    pilot: Token
    secret: Token


class Paired(Message):
    """The pilot stored its credential; the driver may retire the pairing code."""

    type: Literal["paired"] = "paired"
    pilot: Token


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
    RobotDescription
    | FrameMetadata
    | Motion
    | Sensors
    | PilotState
    | Command
    | Credential
    | Paired
    | Status
    | Ping
    | Pong,
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

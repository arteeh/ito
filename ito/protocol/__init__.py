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
    calibration: Name = "pinhole"
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
    gravity_deg at any time. The driver scales uncertainty when its own estimate is less
    reliable (MotionSample).
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
        default=None,
        description="Uncertainty model for the camera explicitly named by motion_camera",
    )

    motion_camera: Name | None = None

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
        if (self.motion_noise is None) != (self.motion_camera is None):
            raise ValueError("motion_camera and motion_noise must be supplied together")
        if self.motion_camera and self.motion_camera not in {c.name for c in self.cameras}:
            raise ValueError("motion_camera must name a described camera")
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
    calibration: Name = "pinhole"
    exposure: Number | None = Field(default=None, ge=0, le=1)
    time_sigma: Number = Field(default=0, ge=0, le=1)
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
            "Named camera-to-world in a startup-anchored frame; +Y is up, with an arbitrary "
            "origin. Position never implies a floor or an observed metric scale."
        )
    )
    angular_velocity: tuple[Number, Number, Number] = Field(
        description="Camera frame, rad/s; a gyro where the robot has one"
    )
    uncertainty_scale: Number = Field(
        default=1, ge=1, le=1000, description="Driver inflation of the motion model for this sample"
    )


class Motion(Message):
    """The robot's own camera motion, batched: every sample once, oldest first."""

    type: Literal["motion"] = "motion"
    camera: Name
    samples: tuple[MotionSample, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if any(b.time <= a.time for a, b in zip(self.samples, self.samples[1:], strict=False)):
            raise ValueError("motion samples must be in time order")
        return self


class ImuSample(Model):
    """Right-handed chip axes, rad/s and m/s² specific force (rest reads +g).

    mount maps chip vectors/positions into the named Ito camera (+X right, +Y up,
    -Z forward). orientation maps chip axes into a +Y-up fusion world; yaw is
    arbitrary and never an odometry heading. All times are driver monotonic seconds.
    """

    sensor: Name
    time: Time
    mount: Pose
    angular_velocity: tuple[Number, Number, Number]
    specific_force: tuple[Number, Number, Number]
    orientation: tuple[Number, Number, Number, Number] | None = None
    temperature_c: Number | None = None
    gyro_sigma: Number = Field(gt=0, description="One-sigma angular-rate uncertainty, rad/s")
    accel_sigma: Number = Field(gt=0, description="One-sigma specific-force uncertainty, m/s²")
    time_sigma: Number = Field(ge=0, le=1)

    @model_validator(mode="after")
    def rotation(self) -> Self:
        if self.orientation is not None and abs(sum(v * v for v in self.orientation) - 1) > 0.01:
            raise ValueError("IMU orientation must be a unit quaternion")
        return self


class RangeZone(Model):
    """Slant range along a unit ray, with the zone's full angular width/height in radians.

    A zone is a footprint, never a dense-depth pixel. No-return and invalid readings
    constrain no surface; distance is absent in both cases.
    """

    direction: tuple[Number, Number, Number]
    angular_size: tuple[Number, Number] = Field(description="Full width and height, radians")
    distance: Number | None = Field(default=None, gt=0, le=10000)
    quality: Literal["valid", "no-return", "invalid"]
    sigma: Number = Field(gt=0, description="Range uncertainty in metres")

    @model_validator(mode="after")
    def geometry(self) -> Self:
        if abs(sum(v * v for v in self.direction) - 1) > 0.001:
            raise ValueError("range direction must be a unit vector")
        if any(not 0 < angle <= 3.141592653589793 for angle in self.angular_size):
            raise ValueError("range angular footprint must be in (0, pi]")
        if (self.distance is not None) != (self.quality == "valid"):
            raise ValueError("only a valid zone carries a distance")
        return self


class RangeSample(Model):
    sensor: Name
    time: Time
    mount: Pose = Field(description="Sensor-to-named-camera mount in metres and Ito axes")
    time_sigma: Number = Field(ge=0, le=1)
    exposure: Number | None = Field(default=None, ge=0, le=1)
    zones: tuple[RangeZone, ...] = Field(min_length=1, max_length=1024)


class SensorObservations(Message):
    """Optional head sensors belonging to one explicitly named camera.

    An empty tuple means unavailable. It never means zero motion, zero acceleration,
    or a measured range of zero. The newest bounded history survives a busy link.
    """

    type: Literal["sensors"] = "sensors"
    camera: Name
    imu: tuple[ImuSample, ...] = Field(default=(), max_length=64)
    ranges: tuple[RangeSample, ...] = Field(default=(), max_length=4)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if not self.imu and not self.ranges:
            raise ValueError("sensor observations require at least one sample")
        for samples in (self.imu, self.ranges):
            last = {}
            for sample in samples:
                if sample.time <= last.get(sample.sensor, -1):
                    raise ValueError("sensor samples must be in source-time order")
                last[sample.sensor] = sample.time
        return self


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
    | SensorObservations
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

import math

from ito.protocol import Imu, ImuSample, Pose, RangeGrid

from . import frames

# One sigma of a reading, bias included: the datasheets' zero-rate and zero-g offsets dominate
# white noise at these rates. LSM6DSV16X: ST DS13510 rev 4, table 3 (+-1 dps, +-12 mg).
# BMI088: Bosch BST-BMI088-DS001 rev 1.9, tables 4-6 (+-1 dps, +-20 mg).
IMU_NOISE = {
    "LSM6DSV16X": (math.radians(1.0), 0.012 * 9.81 + 0.05),
    "BMI088": (math.radians(1.0), 0.020 * 9.81 + 0.05),
}
UNKNOWN_IMU = (math.radians(2.0), 0.5)
RANGE_SIGMA = (0.003, 0.005)
# ST's status codes for a usable range: 5 valid, 9 valid with a large pulse (tof/src/lib.rs).
VALID = {5, 9}
# robotd's tof frame (+x along the optical axis, +y left, +z up) to a camera-like one (+x right,
# +y up, looking down -z), which RangeGrid's rows and columns are stated in.
GRID_IN_TOF = (0.5, 0.5, -0.5, -0.5)
# OpenCV camera axes (robotd's frames.camera) to Ito's.
OPENCV_TO_ITO = (0.0, 1.0, 0.0, 0.0)


def mount(sensor, camera, axes=(1.0, 0.0, 0.0, 0.0)):
    to_camera = frames.multiply(
        frames.conjugate(OPENCV_TO_ITO), frames.conjugate(frames.normalized(camera["quat"]))
    )
    q = frames.normalized(
        frames.multiply(to_camera, frames.multiply(frames.normalized(sensor["quat"]), axes))
    )
    offset = tuple(a - b for a, b in zip(sensor["pos"], camera["pos"], strict=True))
    return Pose(position=frames.rotate(to_camera, offset), orientation=(*q[1:], q[0]))


def imu(chip, samples, sensor, camera):
    gyro_sigma, accel_sigma = IMU_NOISE.get(chip, UNKNOWN_IMU)
    readings = []
    for when, data in samples:
        q = data.get("quat")
        orientation = None
        if q and any(q):
            # The chip's fusion: z up, yaw arbitrary. Ito's world is +Y up.
            w, x, y, z = frames.normalized(frames.multiply(frames.TO_ITO, q))
            orientation = (x, y, z, w)
        readings.append(
            ImuSample(
                time=when,
                angular_velocity=tuple(data["gyro"]),
                specific_force=tuple(data["accel"]),
                orientation=orientation,
            )
        )
    return Imu(
        sensor=chip.lower(),
        mount=mount(sensor, camera),
        gyro_sigma=gyro_sigma,
        accel_sigma=accel_sigma,
        samples=tuple(readings),
    )


def ranges(chip, when, data, sensor, camera, fov):
    rows, cols = data["rows"], data["cols"]
    distances = data["distance_mm"]
    if rows * cols != len(distances) or len(data["status"]) != len(distances):
        raise ValueError("Microduck ToF frame does not match its zones")
    return RangeGrid(
        sensor=chip.lower(),
        time=when,
        mount=mount(sensor, camera, GRID_IN_TOF),
        rows=rows,
        cols=cols,
        field_of_view=(fov, fov),
        distances=tuple(
            mm / 1000 if status in VALID and 0 < mm <= 100_000 else None
            for mm, status in zip(distances, data["status"], strict=True)
        ),
        sigma=RANGE_SIGMA,
    )

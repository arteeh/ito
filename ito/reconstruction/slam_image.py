import math
from functools import cache

import numpy as np

LONG_SIDE = {
    percent: math.ceil(512 * math.sqrt(percent / 100) / 16) * 16 for percent in (25, 50, 75, 100)
}


def input_size(width, height, percent):
    long = LONG_SIDE[percent]
    short = max(16, round(min(width, height) / max(width, height) * long / 16) * 16)
    return (long, short) if width >= height else (short, long)


@cache
def sensor_frame_type():
    import torch
    from mast3r_slam.frame import Frame

    class SensorFrame(Frame):
        def update_pointmap(self, points, confidence):
            super().update_pointmap(
                torch.where(self.sensor_pixels, points, 0),
                torch.where(self.sensor_pixels, confidence, 0),
            )
            # Weighted filtering (ITO_SLAM=full) divides by the summed confidence, zero here.
            self.X_canon = torch.where(self.sensor_pixels, self.X_canon, 0)

    return SensorFrame


class SLAMImage:
    def __init__(self, intrinsics, percent):
        self.width, self.height = input_size(intrinsics.width, intrinsics.height, percent)
        scale = min(self.width / intrinsics.width, self.height / intrinsics.height)
        self.scaled = (
            max(1, round(intrinsics.width * scale)),
            max(1, round(intrinsics.height * scale)),
        )
        self.sensor_pixels = None
        self.left = (self.width - self.scaled[0]) // 2
        self.top = (self.height - self.scaled[1]) // 2
        sx, sy = self.scaled[0] / intrinsics.width, self.scaled[1] / intrinsics.height
        self.calibration = np.array(
            [
                [intrinsics.fx * sx, 0, intrinsics.cx * sx + self.left],
                [0, intrinsics.fy * sy, intrinsics.cy * sy + self.top],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

    def frame(self, index, rgb, transform):
        import cv2
        import torch
        from mast3r_slam.frame import Frame

        image = np.zeros((self.height, self.width, 3), np.uint8)
        width, height = self.scaled
        image[self.top : self.top + height, self.left : self.left + width] = cv2.resize(
            rgb,
            self.scaled,
            interpolation=cv2.INTER_AREA,
        )
        pixels = torch.from_numpy(image)
        normalized = (
            pixels.to(device="cuda", dtype=torch.float32).permute(2, 0, 1)[None] / 127.5 - 1
        )
        shape = torch.tensor([[self.height, self.width]], dtype=torch.int32)
        frame_type = Frame
        if self.scaled != (self.width, self.height):
            if self.sensor_pixels is None:
                sensor = np.zeros((self.height, self.width), bool)
                sensor[self.top : self.top + height, self.left : self.left + width] = True
                self.sensor_pixels = torch.as_tensor(sensor, device="cuda").reshape(-1, 1)
            frame_type = sensor_frame_type()
        frame = frame_type(index, normalized, shape, shape.clone(), pixels.float() / 255, transform)
        if self.sensor_pixels is not None:
            frame.sensor_pixels = self.sensor_pixels
        return frame

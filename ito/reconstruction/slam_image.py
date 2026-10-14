"""One uncropped patch grid for MASt3R's image, points and camera calibration."""

import math

import numpy as np

LONG_SIDE = {
    percent: math.ceil(512 * math.sqrt(percent / 100) / 16) * 16 for percent in (25, 50, 75, 100)
}


def input_size(width, height, percent):
    long = LONG_SIDE[percent]
    short = max(16, round(min(width, height) / max(width, height) * long / 16) * 16)
    return (long, short) if width >= height else (short, long)


class SLAMImage:
    def __init__(self, intrinsics, percent):
        self.width, self.height = input_size(intrinsics.width, intrinsics.height, percent)
        scale = min(self.width / intrinsics.width, self.height / intrinsics.height)
        self.scaled = (
            max(1, round(intrinsics.width * scale)),
            max(1, round(intrinsics.height * scale)),
        )
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

        # Fit the whole sensor into the nearest patch grid; only sub-patch padding remains.
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
        shape = torch.tensor([[self.height, self.width]], device="cuda", dtype=torch.int32)
        return Frame(index, normalized, shape, shape.clone(), pixels.float() / 255, transform)

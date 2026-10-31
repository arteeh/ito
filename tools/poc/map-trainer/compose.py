"""Put four filmed runs (tools/tapes replay.py for Ito, run.py for MoonSplat; both write
path.avi, overview.avi and ticks.jsonl at the same tick times) into one 2x2 1920x1080 H.264 video,
cut to under 60 s, plus a contact sheet every 10 s for agents (OUT stem + -sheet.jpg):

    uv run --no-project --with opencv-python-headless --with imageio-ffmpeg --with numpy \
        python compose.py OUT.mp4 "LABEL|RUN|path" "LABEL|RUN|overview" ... (4 panels, row by row)
"""

import json
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

W, H, FPS, MAX_FRAMES = 960, 540, 15, 898  # 898 frames at 15 fps: 59.9 s


def label(img, text, colour=(80, 230, 255)):
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, colour, 1, cv2.LINE_AA)
    return img


def status(tick):
    if "status" in tick:  # Ito's worker
        s = tick["status"].replace("MASt3R-SLAM ", "")
        return s.split(", 0 relocalised")[0].replace(" keyframes", " kf")[:58]
    return f"{tick['keyframes']} kf, {tick['anchors'] // 1000}k anchors"


def main():
    out = Path(sys.argv[1])
    panels = [p.split("|") for p in sys.argv[2:6]]
    films, ticks = [], []
    for _, run, view in panels:
        films.append(cv2.VideoCapture(str(Path(run) / f"{view}.avi")))
        ticks.append([json.loads(x) for x in (Path(run) / "ticks.jsonl").read_text().splitlines()])
    writer = imageio_ffmpeg.write_frames(
        str(out),
        (2 * W, 2 * H),
        fps=FPS,
        codec="libx264",
        quality=None,
        pix_fmt_out="yuv420p",
        output_params=["-crf", "23", "-preset", "slow", "-movflags", "+faststart"],
    )
    writer.send(None)
    sheet = []
    n = min(MAX_FRAMES, *(len(t) for t in ticks))
    for i in range(n):
        tiles = []
        for (name, _, view), film, tick in zip(panels, films, ticks, strict=True):
            ok, img = film.read()
            img = cv2.resize(img, (W, H)) if ok else np.zeros((H, W, 3), np.uint8)
            tiles.append(
                label(img, f"{name} | {view} | t {tick[i]['t']:.0f} s | {status(tick[i])}")
            )
        frame = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        writer.send(np.ascontiguousarray(frame[:, :, ::-1]))
        if i % (FPS * 10) == FPS * 10 - 1 or i == n - 1:
            sheet.append(cv2.resize(frame, (960, 540)))
    writer.close()
    rows = [np.hstack(sheet[k : k + 2]) for k in range(0, len(sheet) - 1, 2)]
    if len(sheet) % 2:
        rows.append(np.hstack([sheet[-1], np.zeros_like(sheet[-1])]))
    cv2.imwrite(str(out.with_name(out.stem + "-sheet.jpg")), np.vstack(rows))
    print("wrote", out, n, "frames", round(n / FPS, 2), "s")


if __name__ == "__main__":
    main()

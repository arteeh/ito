"""Compose a replay.py run into the user's video and the agents' frame grids (WSL, uv):

    compose.py TAPE RUN [--ref REFRUN] [--drop NAME] [--grid-every 10]

Video RUN/<tape>__<candidate>.mp4, 1920x1080 H.264 (the imageio_ffmpeg encoder tools/session.py
uses), one frame per drawn tick:
    candidate on the path      | true-pose reference on the path (REFRUN)
    the robot's camera frame   | candidate from the fixed room overview
--drop also copies it to C:\\Users\\me\\Projects\\ito\\videos\\<drop>\\.
Frame grids RUN/grid-NN.jpg (agents only): the composed frame every --grid-every seconds,
12 per sheet, at half size, each stamped with its tape time."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data import Recording  # noqa: E402

W, H = 960, 540
VIDEOS = Path("/mnt/c/Users/me/Projects/ito/videos")


def label(img, text, colour=(80, 230, 255)):
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
    return img


def frames(video):
    capture = cv2.VideoCapture(str(video))
    while True:
        ok, image = capture.read()
        if not ok:
            return
        yield image


def blank(text):
    return label(np.full((H, W, 3), 16, np.uint8), text, (120, 120, 120))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tape")
    p.add_argument("run")
    p.add_argument("--ref")
    p.add_argument("--drop")
    p.add_argument("--grid-every", type=float, default=10.0)
    a = p.parse_args()
    run = Path(a.run)
    tape = Path(a.tape).name
    name = run.name
    candidate = name.split("__", 1)[1] if "__" in name else name
    env = (run / "env.txt").read_text().strip() if (run / "env.txt").exists() else ""
    code = (run / "code.txt").read_text().strip() if (run / "code.txt").exists() else ""
    title = f"{candidate}" + (f"  {env}" if env else "")
    ticks = [json.loads(line) for line in (run / "ticks.jsonl").read_text().splitlines()]
    fps = (len(ticks) - 1) / max(ticks[-1]["t"], 1e-6) if len(ticks) > 1 else 15
    rec = Recording(a.tape)
    by_seq = {f["seq"]: f for f in rec.frames}
    ref = Path(a.ref) if a.ref else None
    ref_frames = frames(ref / "path.avi") if ref and (ref / "path.avi").exists() else None
    ref_name = ref.name.split("__", 1)[1] if ref else "reference"
    out = run / f"{name}.mp4"
    writer = imageio_ffmpeg.write_frames(
        str(out),
        (2 * W, 2 * H),
        fps=round(fps, 3),
        codec="libx264",
        pix_fmt_out="yuv420p",
        quality=None,
        macro_block_size=8,
        output_params=["-crf", "18", "-preset", "fast", "-movflags", "+faststart"],
        ffmpeg_log_level="error",
    )
    writer.send(None)
    grid, sheets, next_grid = [], 0, 0.0
    camera_cache = (None, None)
    for tick, path, overview in zip(ticks, frames(run / "path.avi"), frames(run / "overview.avi")):
        t = tick["t"]
        left = label(path.copy(), f"{title} | path | {tick['splats']:,} splats")
        reference = next(ref_frames, None) if ref_frames is not None else None
        right = (
            label(reference.copy(), f"{ref_name} (true poses) | path")
            if reference is not None
            else blank("true-pose reference: not rendered yet")
        )
        seq = tick["frame"]
        if camera_cache[0] != seq:
            f = by_seq.get(seq)
            image = rec.rgb(f, (W, H))[..., ::-1] if f is not None else np.zeros((H, W, 3), np.uint8)
            camera_cache = (seq, image)
        camera = label(camera_cache[1].copy(), f"robot camera | {tape} t={t:6.1f} s | frame {seq}")
        over = label(overview.copy(), f"{candidate} | room overview")
        frame = np.concatenate(
            [np.concatenate([left, right], 1), np.concatenate([camera, over], 1)], 0
        )
        writer.send(np.ascontiguousarray(frame[..., ::-1]))
        if t >= next_grid:
            next_grid += a.grid_every
            small = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            grid.append(small)
            if len(grid) == 12:
                sheets += 1
                write_grid(run / f"grid-{sheets:02d}.jpg", grid)
                grid = []
    writer.close()
    if grid:
        sheets += 1
        write_grid(run / f"grid-{sheets:02d}.jpg", grid)
    print(out, f"{len(ticks)} frames at {fps:.2f} fps, {sheets} grids", code)
    if a.drop:
        drop = VIDEOS / a.drop
        drop.mkdir(parents=True, exist_ok=True)
        shutil.copy(out, drop / out.name)
        print("copied to", drop / out.name)


def write_grid(path, cells):
    cells = cells + [np.zeros_like(cells[0])] * (-len(cells) % 3)
    rows = [np.concatenate(cells[i : i + 3], 1) for i in range(0, len(cells), 3)]
    cv2.imwrite(str(path), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()

"""Eye check: what the pilot is shown next to what the camera saw, at chosen moments.

e2e/slam_replay.py RECORDING --truth T --pose RUNG --eye LABEL --out OUT writes into OUT, for 8
evenly spaced moments of the drive and its 4 fastest camera turns, one PNG per moment:

  LEFT    the camera frame: the simulator's render from the true camera pose
  MIDDLE  Ito's scene as published when that frame arrived (before it is integrated), drawn
          by the pilot's own OpenGL renderer from the SAME true pose, at the camera's size and
          intrinsics (truth placed in Ito's world by slam_replay's first-frame alignment)
  RIGHT   (rungs L2-L5) the same scene from Ito's estimate of that frame's pose: where the
          pilot's eye really is. On L1 the estimate is the truth, so the panel is left out.

Holes are black. Each panel says its full-frame PSNR against the camera frame (holes count)
and the share of pixels the scene covers; the header names the label, rung, moment and time.
contact.jpg puts every moment in one image (under 1.5 MB), eye.json the numbers.

With another revision's ito first on PYTHONPATH (see slam_replay's docstring) the scene and
the renderer are that revision's: compare revisions moment by moment, same recording.
"""

import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

EVEN = 8  # evenly spaced moments
TURNS = 4  # fastest camera turns
TURN_GAP = 1.5  # seconds between two turn moments
SETTLE = 1.0  # seconds at the start without moments: the scene is empty
THUMB = 200  # contact sheet panel width
COLUMNS = 2  # moments side by side on the contact sheet
SHEET_BYTES = 1_500_000


def moments(times, speed):
    """[(time, kind)]: EVEN evenly spaced capture times after SETTLE, and the TURNS frames
    with the fastest camera turn (speed: rad/s per frame), TURN_GAP apart."""
    first, last = times[0] + SETTLE, times[-1] - 0.5
    chosen = [(float(t), "even") for t in np.linspace(first, last, EVEN)]
    turns = []
    for i in np.argsort(-speed):
        if len(turns) == TURNS:
            break
        if first <= times[i] <= last and all(abs(times[i] - t) >= TURN_GAP for t, _ in turns):
            turns.append((float(times[i]), f"turn {math.degrees(speed[i]):.0f} deg/s"))
    return sorted(chosen + turns)


def turn_speed(times, samples, delay, gazes):
    """The camera's turn rate at each frame's exposure: the robot's measured angular velocity
    where it sent Motion, else the telemetry gaze's rate."""
    if len(samples):
        rate = np.linalg.norm(samples[:, 17:20], axis=1)
        return np.interp(times - delay, samples[:, 0], rate)
    return np.abs(np.gradient(gazes, times))


def rotation_degrees(a, b):
    cosine = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2
    return math.degrees(math.acos(float(np.clip(cosine, -1, 1))))


class Eye:
    def __init__(self, out, intrinsics, label, rung):
        import moderngl
        import pygame

        from ito.render.renderer import GaussianRenderer

        self.out, self.label, self.rung = out, label, rung
        out.mkdir(parents=True, exist_ok=True)
        pygame.display.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
        )
        pygame.display.set_mode((64, 64), pygame.OPENGL | pygame.DOUBLEBUF | pygame.HIDDEN)
        self.context = moderngl.create_context(require=430)
        self.renderer = GaussianRenderer(self.context)
        i = intrinsics
        self.size = (i.width, i.height)
        self.target = self.context.simple_framebuffer(self.size, components=4, dtype="f4")
        near, far = 0.01, 1000.0
        p = np.zeros((4, 4), np.float32)
        p[0, 0] = 2 * i.fx / i.width
        p[1, 1] = 2 * i.fy / i.height
        p[0, 2] = 1 - 2 * i.cx / i.width
        p[1, 2] = 2 * i.cy / i.height - 1
        p[2, 2] = -(far + near) / (far - near)
        p[2, 3] = -2 * far * near / (far - near)
        p[3, 2] = -1
        self.projection = p
        self.shots = []

    def upload(self, records, antialiased):
        from ito.render.scene import GaussianBuffer

        records = np.ascontiguousarray(records, np.float32)
        options = {}
        if "antialiased" in GaussianBuffer.__dataclass_fields__:
            options["antialiased"] = antialiased
        self.renderer.upload(GaussianBuffer(records, **options))
        self.count = len(records)

    def draw(self, pose):
        """The uploaded scene from a world-from-camera pose: colour over black, coverage."""
        self.renderer.draw(
            np.asarray(pose, np.float32),
            np.eye(4, dtype=np.float32),
            self.projection,
            self.target,
            clear=(0, 0, 0, 0),
        )
        self.context.finish()
        w, h = self.size
        rgba = np.frombuffer(self.target.read(components=4, dtype="f4"), np.float32)
        rgba = rgba.reshape(h, w, 4)[::-1]
        return rgba[..., :3].clip(0, 1), rgba[..., 3] > 0.5

    def shot(self, t, kind, frame, records, antialiased, true_pose, estimate):
        """One moment: frame (uint8 camera image), records (the published scene), the frame's
        true pose in Ito's world and Ito's estimate of it (None: same as the truth)."""
        self.upload(records, antialiased)
        truth = frame.astype(np.float32) / 255
        panels = [(truth, "camera (sim render, true pose)")]
        row = dict(number=len(self.shots), t=round(t, 2), kind=kind, splats=self.count)
        views = [("true", true_pose, "Ito scene, true pose")]
        if estimate is not None:
            rotation = rotation_degrees(estimate, true_pose)
            shift = float(np.linalg.norm(estimate[:3, 3] - true_pose[:3, 3]))
            row.update(pose_error_deg=round(rotation, 2), pose_error_m=round(shift, 3))
            views.append(
                (
                    "ito",
                    estimate,
                    f"Ito scene, Ito pose ({rotation:.1f} deg {shift * 100:.0f} cm off)",
                )
            )
        for name, pose, title in views:
            seen, covered = self.draw(pose)
            psnr = float(-10 * np.log10(max(((seen - truth) ** 2).mean(), 1e-10)))
            row[f"{name}_psnr"] = round(psnr, 2)
            row[f"{name}_coverage"] = round(float(covered.mean()), 3)
            panels.append((seen, f"{title}  {psnr:.1f} dB  covers {covered.mean():.0%}"))
        image = Image.fromarray(
            (np.concatenate([p for p, _ in panels], axis=1) * 255).round().astype(np.uint8)
        )
        pen = ImageDraw.Draw(image)
        w = self.size[0]
        header = f"{self.label}  {self.rung}  #{row['number']}  t={t:.1f}s  {kind}"
        header += f"  {self.count:,} splats"
        pen.text((6, 4), header, fill="yellow", stroke_width=2, stroke_fill="black")
        for k, (_, title) in enumerate(panels):
            pen.text((k * w + 6, 20), title, fill="white", stroke_width=2, stroke_fill="black")
        name = f"moment-{row['number']:02d}-t{t:05.1f}.png"
        image.save(self.out / name)
        row["png"] = name
        self.shots.append((row, image))

    def finish(self):
        """contact.jpg with every moment, and eye.json."""
        import ito

        rows = [row for row, _ in self.shots]
        source = str(Path(ito.__file__).parent.parent)
        (self.out / "eye.json").write_text(
            json.dumps(dict(label=self.label, rung=self.rung, ito=source, moments=rows), indent=2)
        )
        if not self.shots:
            return
        panels = self.shots[0][1].width // self.size[0]
        width = THUMB * panels
        height = round(self.size[1] * THUMB / self.size[0])
        down = math.ceil(len(self.shots) / COLUMNS)
        sheet = Image.new("RGB", (width * COLUMNS, height * down))
        for k, (row, image) in enumerate(self.shots):
            thumb = image.resize((width, height), Image.LANCZOS)
            caption = f"#{row['number']} t={row['t']:.1f}s {row['kind']}"
            caption += f"  {row['true_psnr']:.1f} dB" + (
                f" / {row['ito_psnr']:.1f} dB" if "ito_psnr" in row else ""
            )
            pen = ImageDraw.Draw(thumb)
            pen.rectangle((0, 0, width, 14), fill="black")
            pen.text((3, 1), caption, fill="yellow")
            sheet.paste(thumb, ((k // down) * width, (k % down) * height))
        for quality in (85, 75, 65, 55, 45):
            data = io.BytesIO()
            sheet.save(data, "JPEG", quality=quality)
            if data.tell() < SHEET_BYTES:
                break
        (self.out / "contact.jpg").write_bytes(data.getvalue())
        self.renderer.close()

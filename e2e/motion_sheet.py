"""Turn a motion recording into labelled frame sheets that a person or a vision model can judge.

    uv run python e2e/motion_sheet.py REC [--compare REC_B] [--step-ms 100] [--out DIR]

REC is a folder written by ITO_RECORD=<dir> (ito/desktop/record.py): pilot/ and camera/
streams, each JPEGs plus frames.jsonl ("t" in seconds). Any folder of such streams works.

Output (default REC/sheets/):
- SHEETS.md   how to read the sheets, run facts, a "look here first" list, one table per sheet
- sheet-NNN.jpg  one row per moment, step-ms apart, read top to bottom; columns are streams
- overview-NN.jpg  one pilot frame per second for the whole run
- sheets.json  every row's numbers, for scripts
- motion.mp4   camera | pilot side by side at 30 fps, for people (needs opencv)

Each cell shows the newest frame that existed at that moment (never a later one), so a
stall shows up as REPEAT rather than being hidden by picking the nearest frame.
The CHANGE column is |this pilot frame - the row above| x4: steady scenes stay dark,
popping, shimmer, smearing and jumps light up. Sheets are sized to stay readable after
the ~1.15 MP downscale vision models apply.
"""

import argparse
import bisect
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SHEET_W, SHEET_H = 1500, 1400
LABEL_W = 230
HEADER_H = 124
CAPTION_H = 22
PAD = 6
BG = (24, 24, 24)
FG = (235, 235, 235)
DIM = (150, 150, 150)
RED = (235, 60, 60)
AMBER = (240, 180, 40)


def font(size):
    for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


F_TITLE, F_BIG, F_TEXT, F_SMALL = font(22), font(20), font(15), font(13)


@dataclass
class Stream:
    name: str  # shown in column headings
    folder: Path
    rows: list = field(default_factory=list)
    times: list = field(default_factory=list)

    @classmethod
    def load(cls, name, folder):
        stream = cls(name, folder)
        index = folder / "frames.jsonl"
        if index.exists():
            for line in index.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    stream.rows.append(json.loads(line))
        stream.rows.sort(key=lambda row: row["t"])
        stream.times = [row["t"] for row in stream.rows]
        return stream

    def at(self, t):
        """Index of the newest frame at or before t, or None."""
        i = bisect.bisect_right(self.times, t) - 1
        return i if i >= 0 else None

    def image(self, i):
        return Image.open(self.folder / self.rows[i]["file"]).convert("RGB")


def fit(image, w, h):
    """Letterbox image into w x h, keeping its aspect."""
    scale = min(w / image.width, h / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    cell = Image.new("RGB", (w, h), (0, 0, 0))
    cell.paste(image.resize(size, Image.LANCZOS), ((w - size[0]) // 2, (h - size[1]) // 2))
    return cell


def change(a, b, w, h):
    """Heat image of |a - b| x4 and the mean absolute change in % of full scale."""
    a = np.asarray(fit(a, w, h), np.float32)
    b = np.asarray(fit(b, w, h), np.float32)
    diff = np.abs(a - b).mean(axis=2)
    score = float(diff.mean() / 255 * 100)
    v = np.clip(diff * 4, 0, 255)
    heat = np.stack((v, np.clip(v * 2 - 255, 0, 255), np.clip(v * 4 - 3 * 255, 0, 255)), axis=2)
    return Image.fromarray(heat.astype(np.uint8)), score


def keys(move, look):
    """Pilot input as words; move is (right, up, forward), look is (right, up)."""
    if not move and not look:
        return "-"
    words = []
    right, _up, forward = (list(move) + [0, 0, 0])[:3] if move else (0, 0, 0)
    if forward > 0.1:
        words.append("FWD")
    if forward < -0.1:
        words.append("BACK")
    if right > 0.1:
        words.append("STRAFE-R")
    if right < -0.1:
        words.append("STRAFE-L")
    if look and look[0] > 0.1:
        words.append("TURN-R")
    if look and look[0] < -0.1:
        words.append("TURN-L")
    return " ".join(words) or "idle"


def yaw_delta(a, b):
    if a is None or b is None:
        return None
    return (a - b + 180) % 360 - 180


class Run:
    def __init__(self, folder, label):
        self.folder, self.label = Path(folder), label
        self.pilot = Stream.load(f"{label} PILOT VIEW", self.folder / "pilot")
        self.camera = Stream.load(f"{label} ROBOT CAMERA", self.folder / "camera")
        if not self.pilot.rows:
            raise SystemExit(f"{self.folder}: no pilot frames (pilot/frames.jsonl)")
        self.t0 = self.pilot.times[0]


def sample(runs, step, start, end):
    """Rows of per-run picks at moments step seconds apart, relative to each run's start."""
    length = min(run.pilot.times[-1] - run.t0 for run in runs)
    end = length if end is None else min(end, length)
    rows, previous = [], [dict() for _ in runs]
    for k in range(int(math.floor((end - start) / step)) + 1):
        rel = start + k * step
        row = dict(index=k, rel=round(rel, 3), runs=[])
        for run, prev in zip(runs, previous, strict=True):
            t = run.t0 + rel
            pi, ci = run.pilot.at(t), run.camera.at(t)
            meta = run.pilot.rows[pi] if pi is not None else {}
            pick = dict(
                pilot=pi,
                camera=ci,
                pilot_age_ms=None if pi is None else round((t - run.pilot.times[pi]) * 1000),
                camera_age_ms=None if ci is None else round((t - run.camera.times[ci]) * 1000),
                pilot_repeat=pi is not None and pi == prev.get("pilot"),
                camera_repeat=ci is not None and ci == prev.get("camera"),
                keys=keys(meta.get("move"), meta.get("look")),
                yaw=meta.get("yaw"),
                head_yaw=meta.get("head_yaw"),
                yaw_step=yaw_delta(meta.get("yaw"), prev.get("yaw")),
                capture_age_ms=meta.get("capture_age_ms"),
                gaussians=meta.get("gaussians"),
                flat_video=meta.get("flat_video"),
                change=None,
                prev_pilot=prev.get("pilot"),
            )
            row["runs"].append(pick)
            prev.update(pilot=pi, camera=ci, yaw=meta.get("yaw"))
        rows.append(row)
    return rows


def caption(draw, x, y, text, colour=DIM):
    draw.text((x + 2, y + 3), text, font=F_SMALL, fill=colour)


def render_sheet(runs, rows, number, total, step_ms, path):
    columns = []  # (kind, run index, heading)
    if runs[0].camera.rows:
        columns.append(("camera", 0, "CAMERA"))
    for r, run in enumerate(runs):
        columns.append(("pilot", r, f"PILOT VIEW {run.label}".strip()))
    for r, run in enumerate(runs):
        columns.append(("change", r, f"CHANGE vs row above {run.label}".strip()))
    per_row = (SHEET_H - HEADER_H) // len(rows)
    cell_h = per_row - CAPTION_H - PAD
    aspect = 16 / 9
    if columns[0][0] == "camera":  # a portrait robot camera needs a narrower column
        first = runs[0].camera.image(0)
        aspect = min(aspect, first.width / first.height)
    wide = len(columns) - (columns[0][0] == "camera")
    narrow = aspect / (16 / 9) if columns[0][0] == "camera" else 0
    cell_w = min(round((SHEET_W - LABEL_W) / (wide + narrow)) - PAD, round(cell_h * 16 / 9))
    cell_h = min(cell_h, round(cell_w * 9 / 16))
    widths = [round(cell_h * aspect) if kind == "camera" else cell_w for kind, _r, _h in columns]
    lefts = [LABEL_W + PAD + sum(w + PAD for w in widths[:c]) for c in range(len(columns))]
    width = lefts[-1] + widths[-1] + PAD
    height = HEADER_H + len(rows) * (cell_h + CAPTION_H + PAD) + PAD
    sheet = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(sheet)
    first, last = rows[0]["rel"], rows[-1]["rel"]
    names = " vs ".join(run.label or run.folder.name for run in runs)
    draw.text(
        (PAD, 6),
        f"Ito motion sheet {number}/{total} - {names} - t = {first:.2f} to {last:.2f} s",
        font=F_TITLE,
        fill=FG,
    )
    draw.text(
        (PAD, 36),
        f"Each ROW is one moment, {step_ms:g} ms after the row above. Read top to bottom.",
        font=F_TEXT,
        fill=FG,
    )
    draw.text(
        (PAD, 56),
        "Cells: newest frame that existed then, captioned 'frame #, ms old'. RED border =\n"
        "REPEAT (no new frame since the row above).\n"
        "CHANGE: dark = steady, bright = pop/shimmer/jump.",
        font=F_SMALL,
        fill=DIM,
    )
    y = HEADER_H - 19
    for c, (_kind, _r, heading) in enumerate(columns):
        draw.text((lefts[c], y), heading, font=F_SMALL, fill=AMBER)
    for k, row in enumerate(rows):
        top = HEADER_H + k * (cell_h + CAPTION_H + PAD)
        lead = row["runs"][0]
        draw.text((PAD, top), f"ROW {k + 1}", font=F_BIG, fill=FG)
        draw.text((PAD, top + 24), f"t = {row['rel']:.2f} s", font=F_TEXT, fill=FG)
        lines = [f"input: {lead['keys']}"]
        if lead["yaw"] is not None:
            step = lead["yaw_step"]
            lines.append(f"yaw {lead['yaw']:.1f} deg" + ("" if step is None else f" ({step:+.1f})"))
        if lead["head_yaw"]:
            lines.append(f"pilot look {lead['head_yaw']:+.1f} deg")
        if lead["capture_age_ms"] is not None:
            lines.append(f"scene age {lead['capture_age_ms']:.0f} ms")
        if lead["gaussians"]:
            lines.append(f"{lead['gaussians']:,} splats")
        if lead["flat_video"]:
            lines.append("FLAT CAMERA FALLBACK")
        for i, line in enumerate(lines):
            draw.text((PAD, top + 46 + i * 17), line, font=F_SMALL, fill=DIM)
        for c, (kind, r, _heading) in enumerate(columns):
            x, cw = lefts[c], widths[c]
            run, pick = runs[r], row["runs"][r]
            box = (x - 2, top - 2, x + cw + 1, top + cell_h + 1)
            if kind == "camera":
                i, repeat, age = pick["camera"], pick["camera_repeat"], pick["camera_age_ms"]
                stream = run.camera
            else:
                i, repeat, age = pick["pilot"], pick["pilot_repeat"], pick["pilot_age_ms"]
                stream = run.pilot
            if i is None:
                draw.rectangle(box[:2] + (x + cw, top + cell_h), fill=(60, 60, 60))
                draw.text((x + 8, top + 8), "NO FRAME YET", font=F_TEXT, fill=FG)
                continue
            if kind == "change":
                prev = pick["prev_pilot"]
                if prev is None:
                    draw.text((x + 8, top + 8), "(first row)", font=F_TEXT, fill=DIM)
                    continue
                heat, score = change(stream.image(i), stream.image(prev), cell_w, cell_h)
                pick["change"] = round(score, 2)
                sheet.paste(heat, (x, top))
                caption(draw, x, top + cell_h, f"change {score:.1f}%", FG if score > 3 else DIM)
                continue
            sheet.paste(fit(stream.image(i), cw, cell_h), (x, top))
            if repeat:
                draw.rectangle(box, outline=RED, width=3)
            text = ("REPEAT " if repeat else "") + f"#{i}, {age} ms old"
            caption(draw, x, top + cell_h, text, RED if repeat else DIM)
    sheet.save(path, quality=90)


def render_overview(run, out, every=1.0, per_page=30, cols=6):
    length = run.pilot.times[-1] - run.t0
    picks = [(s, run.pilot.at(run.t0 + s)) for s in np.arange(0, length + 1e-6, every)]
    pages = []
    cell_w = (SHEET_W - PAD) // cols - PAD
    cell_h = round(cell_w * 9 / 16)
    for p in range(0, len(picks), per_page):
        chunk = picks[p : p + per_page]
        nrows = math.ceil(len(chunk) / cols)
        sheet = Image.new("RGB", (SHEET_W, 50 + nrows * (cell_h + CAPTION_H + PAD) + PAD), BG)
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (PAD, 8),
            f"Ito overview {len(pages) + 1} - {run.label or run.folder.name} - pilot view every "
            f"{every:g} s, read left to right, top to bottom",
            font=F_TITLE,
            fill=FG,
        )
        for k, (s, i) in enumerate(chunk):
            x = PAD + (k % cols) * (cell_w + PAD)
            y = 50 + (k // cols) * (cell_h + CAPTION_H + PAD)
            if i is not None:
                sheet.paste(fit(run.pilot.image(i), cell_w, cell_h), (x, y))
            caption(draw, x, y + cell_h, f"t = {s:.0f} s", FG)
        path = out / f"overview-{len(pages) + 1:02d}.jpg"
        sheet.save(path, quality=88)
        pages.append(path.name)
    return pages


def facts(run):
    times = np.array(run.pilot.times)
    gaps = np.diff(times) * 1000 if len(times) > 1 else np.array([0.0])
    length = times[-1] - times[0]
    cam = np.array(run.camera.times)
    return dict(
        run=run.label or run.folder.name,
        seconds=round(float(length), 2),
        pilot_frames=len(times),
        pilot_fps=round(len(times) / length, 1) if length else None,
        pilot_gap_p95_ms=round(float(np.percentile(gaps, 95)), 1),
        pilot_gap_max_ms=round(float(gaps.max()), 1),
        camera_frames=len(cam),
        camera_fps=round(len(cam) / length, 1) if length and len(cam) else 0,
        flat_fallback_frames=sum(1 for row in run.pilot.rows if row.get("flat_video")),
    )


def video(runs, out, fps=30):
    try:
        import cv2
    except ImportError:
        return None
    run = runs[0]
    length = run.pilot.times[-1] - run.t0
    size = (640, 360)
    path = out / "motion.mp4"
    writer = None
    for k in range(int(length * fps)):
        t = run.t0 + k / fps
        panels = [run.camera, *(r.pilot for r in runs)]
        frames = []
        for stream in panels:
            i = stream.at(t if stream is not run.camera else t)
            frames.append(
                np.asarray(fit(stream.image(i), *size))
                if i is not None
                else np.zeros((360, 640, 3), np.uint8)
            )
        frame = np.concatenate(frames, axis=1)
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if writer is not None:
        writer.release()
    return path.name


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("recording", type=Path)
    parser.add_argument("--compare", type=Path, help="a second recording of the same drive")
    parser.add_argument("--labels", default="", help="names for the runs, e.g. main,branch")
    parser.add_argument("--step-ms", type=float, default=100)
    parser.add_argument("--rows", type=int, default=6, help="moments per sheet")
    parser.add_argument("--start", type=float, default=0, help="seconds after the first frame")
    parser.add_argument("--end", type=float)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    folders = [args.recording] + ([args.compare] if args.compare else [])
    labels = (args.labels.split(",") + ["", ""])[: len(folders)]
    if len(folders) == 2:
        labels = [label or folder.name for label, folder in zip(labels, folders, strict=True)]
    runs = [Run(folder, label) for folder, label in zip(folders, labels, strict=True)]
    out = args.out or args.recording / "sheets"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("sheet-*.jpg"):
        old.unlink()
    step = args.step_ms / 1000
    rows = sample(runs, step, args.start, args.end)
    chunks = [rows[i : i + args.rows] for i in range(0, len(rows), args.rows)]
    for n, chunk in enumerate(chunks, 1):
        render_sheet(runs, chunk, n, len(chunks), args.step_ms, out / f"sheet-{n:03d}.jpg")
    overview = render_overview(runs[0], out)
    mp4 = None if args.no_video else video(runs, out)
    run_facts = [facts(run) for run in runs]
    (out / "sheets.json").write_text(
        json.dumps(dict(facts=run_facts, step_ms=args.step_ms, rows=rows), indent=1) + "\n"
    )
    write_guide(out, runs, rows, chunks, run_facts, overview, mp4, args)
    print(f"{len(chunks)} sheets, {len(overview)} overview pages -> {out / 'SHEETS.md'}")


def where(rows_per_sheet, k):
    return f"sheet {k // rows_per_sheet + 1:03d} row {k % rows_per_sheet + 1}"


def write_guide(out, runs, rows, chunks, run_facts, overview, mp4, args):
    per = args.rows
    lines = [
        f"# Motion sheets: {' vs '.join(f['run'] for f in run_facts)}",
        "",
        "## How to read",
        "",
        f"- `sheet-NNN.jpg`: {per} rows per sheet, one row per moment, {args.step_ms:g} ms apart,"
        " top to bottom; sheets continue each other.",
        "- Columns: ROBOT CAMERA (what the robot saw, Ito's input), PILOT VIEW (what Ito drew"
        " for the pilot)" + (", one per run," if len(runs) > 1 else ",") + " CHANGE (pilot view"
        " vs the row above, x4: dark = steady, bright = changed).",
        "- Each cell is the newest frame that existed at that moment. A red border and REPEAT"
        " mean no new frame since the row above (a stall, or a slow stream).",
        "- Left labels: time since the first frame, the pilot's input (FWD/BACK/TURN-L/R...),"
        " robot camera yaw and its change since the row above, pilot look (head yaw relative"
        " to the robot), scene age (ms since the camera frame behind the drawn scene).",
        "- `overview-NN.jpg`: one pilot frame per second, for the whole-run picture."
        + (f" `{mp4}`: side-by-side video for people." if mp4 else ""),
        "",
        "## What to judge (by eye, citing `sheet NNN row R`)",
        "",
        "- Sharpness: can you read edges, text, textures the robot camera shows?",
        "- Stability: while input is idle, CHANGE should stay dark. Bright speckle = shimmer;"
        " bright patches = popping objects; bright everywhere = a jump.",
        "- Motion: during TURN/FWD the view should move smoothly, by similar steps each row;"
        " not freeze (REPEAT), skip, smear or lag behind the camera column.",
        "- Holes, black areas, doubled walls, objects that suddenly come close.",
        "- Numbers below are hints for where to look, not verdicts. A run that scores well but"
        " looks worse than main on these sheets is worse.",
        "",
        "## Run facts",
        "",
    ]
    keys_ = list(run_facts[0])
    lines += ["| " + " | ".join(keys_) + " |", "|" + "---|" * len(keys_)]
    lines += ["| " + " | ".join(str(f[k]) for k in keys_) + " |" for f in run_facts]
    lines += ["", "## Look here first", ""]
    for r, run in enumerate(runs):
        scored = [(row["runs"][r]["change"] or 0, k) for k, row in enumerate(rows)]
        top = sorted(scored, reverse=True)[:5]
        stalls = [k for k, row in enumerate(rows) if row["runs"][r]["pilot_repeat"]]
        name = run.label or run.folder.name
        lines.append(
            f"- {name}: biggest CHANGE at "
            + ", ".join(f"{where(per, k)} ({s:.1f}%)" for s, k in top if s)
        )
        lines.append(
            f"- {name}: {len(stalls)} REPEAT rows"
            + (": " + ", ".join(where(per, k) for k in stalls[:12]) if stalls else "")
        )
    for n, chunk in enumerate(chunks, 1):
        lines += [
            "",
            f"## sheet-{n:03d}.jpg  (t = {chunk[0]['rel']:.2f} to {chunk[-1]['rel']:.2f} s)",
            "",
        ]
        head = ["row", "t s", "input", "yaw (step)"]
        for run in runs:
            tag = f" {run.label}" if len(runs) > 1 else ""
            head += [f"pilot frame{tag}", f"change %{tag}", f"scene age ms{tag}"]
        head += ["camera frame"]
        lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for k, row in enumerate(chunk):
            lead = row["runs"][0]
            yaw = "" if lead["yaw"] is None else f"{lead['yaw']:.1f}"
            if lead["yaw_step"] is not None:
                yaw += f" ({lead['yaw_step']:+.1f})"
            cells = [str(k + 1), f"{row['rel']:.2f}", lead["keys"], yaw]
            for pick in row["runs"]:
                frame = "-" if pick["pilot"] is None else str(pick["pilot"])
                if pick["pilot_repeat"]:
                    frame += " REPEAT"
                cells += [
                    frame,
                    "" if pick["change"] is None else f"{pick['change']:.1f}",
                    "" if pick["capture_age_ms"] is None else f"{pick['capture_age_ms']:.0f}",
                ]
            cam = lead["camera"]
            cells.append(
                "-" if cam is None else str(cam) + (" REPEAT" if lead["camera_repeat"] else "")
            )
            lines.append("| " + " | ".join(cells) + " |")
    (out / "SHEETS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""MoonSplat (TrickyGo/MoonSplat 86c7e15, SIGGRAPH 2026) on a sim tape, unchanged algorithm, its
own per-frame loop (main.py's MoonSplat.update). Runs in the ito-research/moonsplat image (cwd
/opt/moonsplat; its MASt3R-SLAM backend is a thread there, as Ito would need), started by run.sh:

    python run.py TAPE OUT --tools TAPES_TOOLS --start S --end S --cell ee|te|et|tt

Input: the tape's frames that carry the sim's exact depth (every 3rd: 10 per second), rectified to
the pinhole Ito is told (tools/tapes data.py), long side 512 as MASt3R takes them.

--cell is the 2x2 test (#40): first letter the poses the splat mapper gets, second its depth;
e = MoonSplat's own (MASt3R-SLAM keyframe poses / MASt3R depth), t = the sim's truth. Truth enters
in the units of the estimate: one scale c (metres per map unit) is fixed at the first mapped
keyframe from its MASt3R depth against its true depth (as Ito fixes its ToF scale once), and true
poses are given as translation / c, true depth as depth / c. Keyframe choice is MoonSplat's own in
every cell. With true poses, loop-closure map updates are off (the poses never change).

Filming, the same views as tools/tapes replay.py films Ito: after each frame is processed, every
tick (1/FPS of tape time) up to that frame's time is rendered with MoonSplat's own Scaffold-GS
renderer from the tape's smoothed true camera path and from the fixed room overview, 960x540,
into OUT/path.avi and OUT/overview.avi. The map is placed in the true world by the first fed
frame's true pose and c (est poses), or exactly (true poses). Processing is offline: the films show
the map as it stood when MoonSplat had taken every frame up to that tick, however long that took;
OUT/stats.json says how long it took (speed), OUT/frames.jsonl per frame."""

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

WIDTH, HEIGHT = 960, 540
BACKGROUND = (0.04, 0.04, 0.06)  # replay.py's, RGB
OVERVIEW = 6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tape")
    p.add_argument("out")
    p.add_argument("--tools", required=True)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=60.0)
    p.add_argument("--cell", default="ee", choices=("ee", "te", "et", "tt"))
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--refine", action="store_true", help="upstream's final colour refinement")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [a.tools, "/opt/moonsplat"]
    from data import AXES, Recording
    from path import camera_path
    from views import VIEWS, mujoco_camera, world_transform

    rec = Recording(a.tape)
    t0 = rec.frames[0]["received"]
    fed = [
        f for f in rec.truth_frames() if f.get("depth") and a.start <= f["received"] - t0 <= a.end
    ]
    K = rec.K_at((rec.width, rec.height))
    true_pose, true_depth = a.cell[0] == "t", a.cell[1] == "t"

    # MoonSplat's config: its own defaults (config.yaml) but no viz, our focal, our out dir.
    cfg = out / "moonsplat.yaml"
    cfg.write_text(
        json.dumps(
            dict(
                data_configs=dict(
                    frames_path="",
                    results_path=str(out / "upstream"),
                    frames_focal_normalized=float(K[0, 0] / rec.width),
                ),
                method_configs=dict(
                    keyframe_density_thresh=0.5,
                    loop_closure=True,
                    loop_closure_stride=10,
                    viz=False,
                    color_refinement_iter=100 if a.refine else 0,
                    mapping_itr_num=20,
                    window_size=4,
                ),
            )
        )
    )
    (out / "upstream" / "TPV").mkdir(parents=True, exist_ok=True)
    from tracking.config import load_config

    load_config(str(cfg))
    # Checkpoints memory-mapped instead of read whole: MASt3R's 2.7 GB and the retrieval model's
    # second copy of it otherwise peak the process at 6.7 GB RAM while loading (6 GB budget).
    torch_load = torch.load

    def load(f, *args, **kwargs):
        if isinstance(f, (str, Path)):
            kwargs.setdefault("mmap", True)
        return torch_load(f, *args, **kwargs)

    torch.load = load
    import main as M
    import mapping.mapping as mapping_module
    from mapping.gaussian_renderer import render
    from mapping.mapping_utils import anchor_in_frustum, getProjectionMatrix2
    from tracking.mast3r_utils import resize_img
    from utils import Camera

    class Quiet:
        """map() writes a debug PNG pair every keyframe whatever viz says: not written here."""

        def __getattr__(self, name):
            return getattr(cv2, name)

        @staticmethod
        def imwrite(*args, **kwargs):
            return True

    mapping_module.cv2 = Quiet()

    class Tape(torch.utils.data.Dataset):
        def __len__(self):
            return len(fed)

        def __getitem__(self, i):
            img = rec.rgb(fed[i]).astype(np.float32) / 255.0
            return i, resize_img(img, 512)

        def get_img_shape(self):
            img, _ = resize_img(
                np.zeros((rec.height, rec.width, 3), np.float32), 512, return_transformation=True
            )
            return img["img"][0].shape[1:], (rec.height, rec.width)

    dataset = Tape()
    M.dataset = dataset
    began = time.perf_counter()
    slam = M.MoonSplat(dataset)
    setup = time.perf_counter() - began
    h, w = dataset.height, dataset.width
    track = slam.moonsplat_tracking

    # Gauge: MASt3R-SLAM's world is its first keyframe's camera (the first fed frame).
    first = fed[0]["pose"] @ AXES
    state = dict(c=None, slam_kf_frames=[0], i=0)

    def to_map(pose):
        """A true camera (Ito axes, world-from-camera) as world-from-camera in MoonSplat's map."""
        cv = pose @ AXES
        out_pose = np.eye(4)
        c = state["c"]
        if true_pose:
            out_pose[:3, :3] = cv[:3, :3]
            out_pose[:3, 3] = cv[:3, 3] / c
        else:
            out_pose[:3, :3] = first[:3, :3].T @ cv[:3, :3]
            out_pose[:3, 3] = first[:3, :3].T @ (cv[:3, 3] - first[:3, 3]) / c
        return out_pose

    def truth_depth(i):
        return torch.tensor(rec.depth(fed[i], (w, h)), device="cuda") / state["c"]

    timing = dict(track=0.0, map=0.0)
    tracking_update = track.update

    def update(frame, frame_id):
        began = time.perf_counter()
        add, pred, loop = tracking_update(frame, frame_id)
        timing["track"] += time.perf_counter() - began
        if not add:
            return add, pred, loop
        i = state["i"]
        state["slam_kf_frames"].append(i)
        if state["c"] is None:
            d = rec.depth(fed[i], (w, h))
            e = pred.float().cpu().numpy()
            ok = (d > 0.05) & (e > 1e-4)
            state["c"] = float(np.median(d[ok] / e[ok]))
            print(f"scale: {state['c']:.4f} m per map unit (keyframe at frame {i})", flush=True)
        if true_depth:
            pred = truth_depth(i)
        if true_pose:
            track.w2c = np.linalg.inv(to_map(fed[i]["pose"]))
            loop = False
        elif loop and true_depth:
            frames = state["slam_kf_frames"]
            track.pred_depth_history = [truth_depth(frames[k]) for k in range(len(frames))]
        return add, pred, loop

    track.update = update
    backend_update = slam.backend.update

    def backend(msg):
        began = time.perf_counter()
        backend_update(msg)
        torch.cuda.synchronize()
        timing["map"] += time.perf_counter() - began

    slam.backend.update = backend

    # Films.
    path_times, path_poses = camera_path(rec)
    world = world_transform(rec.truth_frames())
    _, position, target = VIEWS[OVERVIEW]
    overview = world @ mujoco_camera(position, target)
    focal = WIDTH / 2 / math.tan(math.radians(90.0) / 2)
    K_over = np.array([[focal, 0, WIDTH / 2], [0, focal, HEIGHT / 2], [0, 0, 1]])
    K_path = rec.K_at((WIDTH, HEIGHT))
    blank = {"unnormalized_img": np.zeros((2, 2, 3), np.uint8)}
    background = torch.tensor(BACKGROUND, device="cuda", dtype=torch.float32)

    def shot(pose, Kv):
        fx, fy, cx, cy = Kv[0, 0], Kv[1, 1], Kv[0, 2], Kv[1, 2]
        proj = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=WIDTH, H=HEIGHT
        ).transpose(0, 1)
        cam = Camera(
            0,
            blank,
            None,
            proj,
            fx,
            fy,
            cx,
            cy,
            2 * math.atan(WIDTH / 2 / fx),
            2 * math.atan(HEIGHT / 2 / fy),
            HEIGHT,
            WIDTH,
            device="cuda",
        )
        w2c = np.linalg.inv(to_map(pose))
        cam.update_RT(
            torch.tensor(w2c[:3, :3], dtype=torch.float32, device="cuda"),
            torch.tensor(w2c[:3, 3], dtype=torch.float32, device="cuda"),
        )
        g = slam.backend.gaussians
        k = torch.tensor([fx, fy, cx, cy], device="cuda", dtype=torch.float32)
        mask = anchor_in_frustum(
            anchors=g.get_anchor,
            intrinsics=k,
            pose=torch.tensor(w2c, device="cuda").float(),
            cam_center=cam.camera_center,
            distance_lis=g.distance_lis,
            levels=g.get_level,
            h=HEIGHT,
            w=WIDTH,
        )
        with torch.no_grad():
            image = render(cam, g, slam.pipeline_params, background, visible_mask=mask)["render"]
        rgb = (image.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
        return np.ascontiguousarray(rgb[..., ::-1])

    def empty():
        return np.full((HEIGHT, WIDTH, 3), [int(255 * c) for c in BACKGROUND[::-1]], np.uint8)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    films = {
        name: cv2.VideoWriter(str(out / f"{name}.avi"), fourcc, a.fps, (WIDTH, HEIGHT))
        for name in ("path", "overview")
    }
    ticks = (out / "ticks.jsonl").open("w")
    frames_log = (out / "frames.jsonl").open("w")
    n_ticks = int((a.end - a.start) * a.fps) + 1
    tick = 0
    render_s = 0.0
    loop_updates = 0

    def film(upto):
        nonlocal tick, render_s
        while tick < n_ticks and a.start + tick / a.fps <= upto:
            when = a.start + tick / a.fps
            began = time.perf_counter()
            mapped = state["c"] is not None and len(slam.backend.viewpoints) > 0
            j = min(np.searchsorted(path_times, when), len(path_times) - 1)
            films["path"].write(shot(path_poses[j], K_path) if mapped else empty())
            films["overview"].write(shot(overview, K_over) if mapped else empty())
            render_s += time.perf_counter() - began
            ticks.write(
                json.dumps(
                    dict(
                        tick=tick,
                        t=round(when, 3),
                        frame=state["i"],
                        keyframes=len(slam.backend.viewpoints),
                        anchors=int(slam.backend.gaussians.get_anchor.shape[0]) if mapped else 0,
                    )
                )
                + "\n"
            )
            tick += 1

    torch.cuda.reset_peak_memory_stats()
    loop_began = time.perf_counter()
    for i in range(len(dataset)):
        state["i"] = i
        before = (timing["track"], timing["map"])
        began = time.perf_counter()
        timestamp, img = dataset[i]
        read = time.perf_counter() - began
        flag = getattr(track, "pose_history_old", None)
        slam.update(timestamp, img)
        torch.cuda.synchronize()
        spent = time.perf_counter() - began
        loop_updates += flag is not getattr(track, "pose_history_old", None)
        frames_log.write(
            json.dumps(
                dict(
                    i=i,
                    t=round(fed[i]["received"] - t0, 3),
                    ms=round(spent * 1e3, 1),
                    read_ms=round(read * 1e3, 1),
                    track_ms=round((timing["track"] - before[0]) * 1e3, 1),
                    map_ms=round((timing["map"] - before[1]) * 1e3, 1),
                    keyframes=len(slam.backend.viewpoints),
                )
            )
            + "\n"
        )
        film(fed[i]["received"] - t0)
    online = time.perf_counter() - loop_began - render_s
    film(a.end)
    track.slam_wrapper.shutdown()
    refine = 0.0
    if a.refine:
        began = time.perf_counter()
        slam.request_color_refinement()
        torch.cuda.synchronize()
        refine = time.perf_counter() - began
    for f in films.values():
        f.release()
    ticks.close()
    frames_log.close()
    # Finals: the overview and six path views of the last map.
    finals = []
    for when in np.linspace(a.start + 5, a.end, 6):
        j = min(np.searchsorted(path_times, when), len(path_times) - 1)
        finals.append(shot(path_poses[j], K_path))
    finals.append(shot(overview, K_over))
    for k, img in enumerate(finals):
        cv2.imwrite(str(out / f"final-{k}.jpg"), img)
    slam.backend.gaussians.save_ply(str(out / "scaffold_model.ply"))

    # Keyframe trajectory against truth (one similarity), numbers for the notes.
    err = None
    if not true_pose:
        frames = state["slam_kf_frames"][1:]
        est = []
        for _, cam in sorted(slam.backend.viewpoints.items()):
            w2c = np.eye(4)
            w2c[:3, :3] = cam.R.detach().cpu().numpy()
            w2c[:3, 3] = cam.T.detach().cpu().numpy()
            est.append(np.linalg.inv(w2c))
        gt = np.array([fed[frames[k]]["pose"] @ AXES for k in range(len(est))])
        est = np.array(est)
        Mx = np.einsum("nij,nkj->ik", gt[:, :3, :3], est[:, :3, :3])
        U, _, Vt = np.linalg.svd(Mx)
        R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
        pe = est[:, :3, 3] @ R.T
        de, dt = pe - pe.mean(0), gt[:, :3, 3] - gt[:, :3, 3].mean(0)
        s = float((de * dt).sum() / max((de * de).sum(), 1e-12))
        err = np.linalg.norm(s * de - dt, axis=1)
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20
    stats = dict(
        cell=a.cell,
        tape=Path(a.tape).name,
        start=a.start,
        end=a.end,
        frames=len(fed),
        keyframes=len(slam.backend.viewpoints),
        anchors=int(slam.backend.gaussians.get_anchor.shape[0]),
        scale_m_per_unit=state["c"],
        setup_s=round(setup, 1),
        online_s=round(online, 1),
        online_fps=round(len(fed) / online, 2),
        track_s=round(timing["track"], 1),
        map_s=round(timing["map"], 1),
        render_s=round(render_s, 1),
        refine_s=round(refine, 1),
        loop_updates=int(loop_updates),
        vram_peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
        vram_peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2),
        ram_peak_gib=round(usage, 2),
        ram_peak_children_gib=round(children, 2),
        kf_position_err_cm=None
        if err is None
        else dict(
            rms=round(float(np.sqrt((err**2).mean()) * 100), 2),
            max=round(float(err.max() * 100), 2),
        ),
    )
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    # MASt3R-SLAM's shared state lives in a multiprocessing Manager, a spawned process that
    # imports this file again.
    main()

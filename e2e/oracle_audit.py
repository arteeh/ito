"""Audit real runs for causal prefixes, input identity, Gaussian records and OOM recovery.

PYTHONPATH=. .venv/bin/python e2e/oracle_audit.py ~/Projects/ito/out/oracle-40
Timing and memory are diagnostics. This script makes no visual quality verdict.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def main():
    out = Path(sys.argv[1])
    result = {"visual": {}, "timing": {}, "native": {}, "inputs": {}}
    for sequence in ("sweep180", "tour"):
        source = out / sequence / "source"
        estimates = out / sequence / "causal-full"
        rows = read(estimates / "frames.json")
        settings = read(estimates / "settings.json")
        assert (
            settings["source_sha256"]
            == hashlib.sha256((source / "frames.jsonl").read_bytes()).hexdigest()
        )
        assert all(r["captured"] <= r["arrival"] <= r["started"] <= r["available"] for r in rows)
        assert all(a["available"] <= b["started"] for a, b in zip(rows, rows[1:], strict=False))
        plan = {r["source_frame_id"]: r["available"] - rows[0]["available"] for r in rows}
        result["inputs"][sequence] = dict(
            observations=len(rows),
            source_sha256=settings["source_sha256"],
            schedule_sha256=hashlib.sha256((estimates / "frames.json").read_bytes()).hexdigest(),
            settings=settings,
        )
        for arm in ("TT", "TE", "ET", "EE", "TF", "main-TT", "main-EE"):
            name = f"{sequence}-{arm}"
            folder = out / "visual" / name
            display = read(folder / "display.json")
            log = [json.loads(s) for s in (folder / "mapping.jsonl").read_text().splitlines()]
            updates = [r for r in log if r["event"] == "publication"]
            observations = [r for r in log if r["event"] == "observation"]
            if (folder / "worker.json").exists():
                work = read(folder / "worker.json")
                final_summary = True
            else:
                # A completed display/trace remains evidence if cancellation interrupts
                # the optional worker footer. Do not invent an unrecorded memory peak.
                assert arm.startswith("main-")
                work = dict(
                    admitted=[r["source_frame_id"] for r in observations],
                    oom=max(r["oom"] for r in observations),
                    torch_peak_reserved=None,
                )
                final_summary = False
            assert work["oom"] == 0
            assert all(
                plan[r["source_frame_id"]] <= r["started"] <= r["finished"] for r in observations
            )
            assert all(all(plan[n] <= r["at"] for n in r["observations"]) for r in updates)
            if arm == "TF":
                assert all(
                    r["tof_source"] is None or r["tof_source"] < r["source_frame_id"]
                    for r in observations
                )
            for path in folder.glob("snapshot-*.npz"):
                with np.load(path) as snap:
                    assert float(snap["actual"]) <= float(snap["deadline"])
                    records = snap["records"]
                    assert np.isfinite(records).all()
                    live = records[:, 0, 3] > 0
                    assert (records[live, 1, :3] > 0).all()
                    assert np.allclose(np.linalg.norm(records[live, 2], axis=1), 1, atol=1e-4)
            result["visual"][name] = dict(
                observations=len(work["admitted"]),
                final_steps=updates[-1]["steps"],
                final_count=updates[-1]["count"],
                min_free_mib=min(r["vram_free"] for r in updates) / 2**20,
                torch_peak_mib=work["torch_peak_reserved"] / 2**20
                if work["torch_peak_reserved"] is not None
                else None,
                final_worker_summary=final_summary,
                recording_display=display,
            )
    for folder in sorted((out / "timing").iterdir()):
        if (folder / "display.json").exists():
            display = read(folder / "display.json")
            assert not display["recording"]
            result["timing"][folder.name] = display
    for folder in sorted((out / "native").iterdir()):
        if (folder / "display.json").exists():
            display = read(folder / "display.json")
            assert "NVIDIA" in display["renderer"]
            assert display["settings"] == dict(
                precision="full", input_resolution=100, max_splats=1_000_000
            )
            if display["optimizer"]:
                work = read(folder / "load/worker.json")
                assert work["oom"] == 0 and len(work["admitted"]) > 1
            result["native"][folder.name] = display
    fault = out / "half-oom-rollback-fixed"
    if (fault / "worker.json").exists():
        work = read(fault / "worker.json")
        assert work["precision"] == "half" and work["loss_dtype"] in (
            "torch.bfloat16",
            "torch.float16",
        )
        assert work["oom"] == 1 and len(work["admitted"]) > 4
        result["half_fault"] = dict(
            oom=work["oom"],
            loss_dtype=work["loss_dtype"],
            observations_after_recovery=len(work["admitted"]) - 3,
        )
    (out / "audit.json").write_text(json.dumps(result, indent=2))
    print("Causal input, snapshot, record-format and fault audits passed")


if __name__ == "__main__":
    main()

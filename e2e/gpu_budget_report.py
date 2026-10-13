"""Freeze shared-GPU evidence and fail if any reference deadline/residency gate fails."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(path):
    result = json.loads((path / "display.json").read_text())
    for field in ("gpu_memory_samples", "ito_dedicated_samples", "ito_counter_coverage"):
        result.pop(field, None)
    publication = result.pop("publication")
    result["publication_ms"] = {
        key: {f"p{p:g}": float(np.percentile(values, p)) for p in (50, 99, 99.9, 100)}
        for key, values in publication.items()
        if values
    }
    for name in ("load", "training"):
        file = path / f"{name}.json"
        if file.exists():
            load = json.loads(file.read_text())
            values = load.pop("integration_ms", load.pop("step_ms", []))
            warm = load.get("warm_integrations", 10)
            active = values[warm:]
            load["active_ms"] = (
                {f"p{p:g}": float(np.percentile(active, p)) for p in (50, 95, 99, 100)}
                if active
                else None
            )
            result[name] = load
    result["frames_sha256"] = digest(path / "frames.npz")
    return result


def cuda_trace(path):
    trace = json.loads(path.read_text())
    kernels = [r for r in trace["activities"] if r["kind"] == "kernel"]
    copies = [r for r in trace["activities"] if r["kind"] == "copy"]
    longest = sorted(kernels, key=lambda r: r["end_ns"] - r["start_ns"], reverse=True)[:10]
    transfers = trace["host_scopes"]["cupy_transfers"]
    return dict(
        raw_sha256=digest(path),
        scope=trace["scope"],
        dropped=trace["dropped"],
        kernels=len(kernels),
        copies=len(copies),
        longest_kernels=[
            dict(name=r["name"], ms=(r["end_ns"] - r["start_ns"]) / 1e6) for r in longest
        ],
        copies_ms=sum((r["end_ns"] - r["start_ns"]) / 1e6 for r in copies),
        copies_bytes=sum(r["bytes"] for r in copies),
        map_phases_ms=trace["host_scopes"]["map_phases_ms"],
        cupy_transfer_host_ms={
            direction: sum(r["cpu_ms"] for r in transfers if r["direction"] == direction)
            for direction in ("H2D", "D2H")
        },
    )


def baseline(candidate, original):
    fields = ("psnr_db", "ssim", "lpips", "coverage", "previously_seen_holes")
    reports = []
    for repeat in (1, 2, 3):
        folder = candidate / f"repeat-{repeat}"
        reference = original / f"repeat-{repeat}"
        before = json.loads((reference / "metrics.json").read_text())["moments"]
        after = json.loads((folder / "metrics.json").read_text())["moments"]

        def key(r):
            return r["run"], r["frame"], r["view"]

        old = {key(r): r for r in before}
        assert set(old) == {key(r) for r in after}
        scores = {field: max(abs(r[field] - old[key(r)][field]) for r in after) for field in fields}
        images = sorted(folder.glob("*.png"))
        assert len(images) == 35
        reports.append(
            dict(
                repeat=repeat,
                comparisons=len(after),
                max_absolute_score_delta=scores,
                png_compared=len(images),
                png_identical=sum(digest(p) == digest(reference / p.name) for p in images),
                camera_moments=[
                    {field: r[field] for field in ("run", "frame", "prefix_frame", *fields)}
                    for r in after
                    if r["view"] == "camera"
                ],
            )
        )
    audit = json.loads((candidate / "audit.json").read_text())
    source = json.loads((candidate / "source/summary.json").read_text())
    return dict(
        comparisons=reports,
        audit=audit,
        source=source,
        product_identity=(candidate / "product-identity.txt").read_text().splitlines(),
        input_sha256=digest(candidate / "inputs.json"),
        packages_sha256=digest(candidate / "packages.txt"),
    )


def pooled(references, output):
    reports = {}
    for size in ((360, 640), (2160, 2160)):
        names = [k for k, v in references.items() if tuple(v["eye_size"]) == size]
        service, completion = [], []
        for name in names:
            with np.load(output / name / "frames.npz") as samples:
                service.extend(samples["duration_ms"])
                completion.extend(samples["completion_ms"])
        reports[f"{size[0]}x{size[1]}"] = dict(
            runs=names,
            frames=len(service),
            stereo_ms={f"p{p:g}": float(np.percentile(service, p)) for p in (50, 99, 99.9, 100)},
            deadline_ms={
                f"p{p:g}": float(np.percentile(completion, p)) for p in (50, 99, 99.9, 100)
            },
            misses=int(np.count_nonzero(np.array(completion) > 1000 / 90)),
            residency_upper_bound_bytes=max(
                references[n]["ito_residency_upper_bound_bytes"] for n in names
            ),
        )
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--original", type=Path, default=Path("../m1-out/verified"))
    args = parser.parse_args()
    cases = {p.parent.name: summarize(p.parent) for p in args.output.glob("*/display.json")}
    references = {
        key: value
        for key, value in cases.items()
        if any(key.startswith(s + "-") for s in ("tour", "sweep180"))
    }
    failures = []
    for name, result in references.items():
        if result["frames"] != 6000 or result["missed_fraction"] >= 0.001:
            failures.append(name + ": deadline fraction")
        if result["longest_miss_run"] > 1:
            failures.append(name + ": consecutive misses")
        if result["ito_residency_upper_bound_bytes"] >= 8_000_000_000:
            failures.append(name + ": residency")
        if result["load"].get("integrated", 0) < 1 or result["training"].get("steps", 0) <= 10:
            failures.append(name + ": no admitted work")
    if len(references) != 12:
        failures.append(f"Expected 12 reference runs, found {len(references)}")
    occupied_load = {
        key: value for key, value in cases.items() if key.startswith("loaded-occupied-")
    }
    if len(occupied_load) != 3:
        failures.append("Missing shared-load occupied-count sweep")
    for name, result in occupied_load.items():
        expected = 6000 if result["occupied"] == 524288 else 900
        if result["frames"] != expected or result["missed_fraction"] >= 0.001:
            failures.append(name + ": deadline fraction")
        if result["longest_miss_run"] > 1:
            failures.append(name + ": consecutive misses")
        if result["ito_residency_upper_bound_bytes"] >= 8_000_000_000:
            failures.append(name + ": residency")
        if result["load"].get("integrated", 0) < 1 or result["training"].get("steps", 0) <= 10:
            failures.append(name + ": no admitted work")
    visuals = {
        p.parent.name: json.loads(p.read_text()) for p in args.output.glob("visual-*/visual.json")
    }
    if len(visuals) != 2 or any(
        v["differing_frames"] or v.get("startup_frames") != 90 for v in visuals.values()
    ):
        failures.append("Frozen-image comparison missing or changed")
    fuzz = {
        name: json.loads((args.output / name / f"{name}.json").read_text())
        for name in ("matching-fuzz", "solver-fuzz")
    }
    if fuzz["matching-fuzz"]["exact_comparisons"] != 24 or fuzz["solver-fuzz"]["exact_solves"] < 1:
        failures.append("Native CUDA equivalence checks failed")
    quality = baseline(args.output / "baseline", args.original)
    for repeat in quality["comparisons"]:
        if repeat["png_identical"] != 35 or any(repeat["max_absolute_score_delta"].values()):
            failures.append(f"Baseline repeat {repeat['repeat']}: image/score change")
    trace = {
        name: cuda_trace(args.output / name / "cuda-trace.json")
        for name in ("trace", "trace-admitted")
    }
    if any(t["dropped"] or not t["kernels"] for t in trace.values()):
        failures.append("Missing or incomplete CUDA trace")
    fingerprints = {r["source_fingerprint"] for r in cases.values()}
    if len(fingerprints) != 1:
        failures.append("Mixed source fingerprints")
    report = dict(
        cases=cases,
        pooled=pooled(references, args.output),
        baseline=quality,
        cuda_trace=trace,
        native_fuzz=fuzz,
        visual=visuals,
        failures=failures,
        scope="offscreen production stereo GL, warmed shared load; no scanout/compositor",
    )
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(reference_runs=len(references), failures=failures), indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
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
    visuals = {
        p.parent.name: json.loads(p.read_text()) for p in args.output.glob("visual-*/visual.json")
    }
    if len(visuals) != 2 or any(v["differing_frames"] for v in visuals.values()):
        failures.append("Frozen-image comparison missing or changed")
    report = dict(
        cases=cases,
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

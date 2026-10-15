#!/usr/bin/env bash
# Frozen causal estimates and all pose/depth intervention arms, serialized on Ceres.
set -euo pipefail
cd "$(dirname "$0")/.."
output="${1:-$HOME/Projects/ito/out/estimator-41}"
recordings="${ITO_ESTIMATOR_RECORDINGS:-$HOME/Projects/ito/out/oracle-40}"
python="${ITO_ESTIMATOR_PYTHON:-/mnt/c/Users/me/Projects/ito/builds/feat-recon-40-41/ito/runtime/python.exe}"
for sequence in sweep180 tour; do
    duration=17
    if [[ "$sequence" == tour ]]; then duration=85; fi
    mkdir -p "$output/causal/$sequence" "$output/visual"
    if [[ "${2:-}" == --reuse-estimates && -f "$output/causal/$sequence/settings.json" ]]; then
        .venv/bin/python - "$output/causal/$sequence/settings.json" <<'VERIFY'
import hashlib, json, sys
from pathlib import Path
settings = json.loads(Path(sys.argv[1]).read_text())
for name, digest in settings["estimator_source_sha256"].items():
    if hashlib.sha256((Path("ito/reconstruction") / name).read_bytes()).hexdigest() != digest:
        raise SystemExit(f"Estimator source changed: {name}; run without --reuse-estimates")
VERIFY
    else
        flock /mnt/c/Users/me/Projects/ito/ceres.lock "$python" -u -I \
            "$(wslpath -w "$PWD/e2e/oracle_estimate.py")" "$(wslpath -w "$recordings/$sequence")" \
            --models 'C:\Users\me\Projects\ito\models-v5' \
            --out "$(wslpath -w "$output/causal/$sequence")" < /dev/null \
            > "$output/causal-$sequence.log" 2>&1
    fi
    for arm in EE ET TE main-EE; do
        options=()
        if [[ "$arm" == main-EE ]]; then options+=(--depth-support); fi
        bash e2e/oracle.sh e2e/oracle_run.py "$recordings/$sequence" \
            --estimates "$output/causal/$sequence" --arm "$arm" \
            --out "$output/visual/$sequence-$arm" --seconds "$duration" --record "${options[@]}" \
            > "$output/$sequence-$arm.log" 2>&1
    done
    .venv/bin/python e2e/estimator_report.py "$recordings/$sequence/causal-full" \
        "$output/causal/$sequence" --out "$output/$sequence.json"
    .venv/bin/python e2e/estimator_reprojection.py "$output/causal/$sequence" \
        --camera "$recordings/$sequence/source/camera.json" \
        --out "$output/reprojection-$sequence.json"
done

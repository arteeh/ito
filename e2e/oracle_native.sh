#!/usr/bin/env bash
# Native Windows NVIDIA GL + the actual WSL mapper; all children inside one Ceres lock.
set -euo pipefail
cd "$(dirname "$0")/.."
output="${1:-$HOME/Projects/ito/out/oracle-40}"
.venv/bin/python - "$output" <<'VERIFY'
import json,sys
from pathlib import Path
from ito.protocol import Intrinsics
from ito.reconstruction.slam_image import SLAMImage
out=Path(sys.argv[1])
for seq in ("sweep180", "tour"):
    k=Intrinsics.model_validate_json((out/seq/"source/camera.json").read_text())
    expected=list(SLAMImage(k,100).scaled)
    for arm in ("TT","TE","ET","EE","TF"):
        p=out/"visual"/f"{seq}-{arm}"/"worker.json"
        work=json.loads(p.read_text())
        if work.get("training_size") != expected:
            raise SystemExit(f"Re-run {seq}-{arm}: fitting grid must match main {expected}")
VERIFY
python=/mnt/c/Users/me/Projects/ito/builds/feat-recon-40-41/ito/runtime/python.exe
for repeat in 1 2 3; do
    for sequence in sweep180 tour; do
        for mode in main-TT main-EE TT TE ET EE TF; do
            name="$sequence-$mode-$repeat"
            arm="$mode"
            if [[ "$mode" == *-frozen ]]; then arm="${mode%-frozen}"; fi
            load=()
            if [[ "$mode" != main-* ]]; then load=(--load); fi
            echo "Native GPU countercheck: $name"
            flock /mnt/c/Users/me/Projects/ito/ceres.lock "$python" -I \
                "$(wslpath -w "$PWD/e2e/oracle_native.py")" \
                "$(wslpath -w "$output/$sequence")" \
                --out "$(wslpath -w "$output/native/$name")" \
                --linux-root "$PWD" --arm "$arm" "${load[@]}" --seconds 20 < /dev/null \
                > "$output/native-$name.log" 2>&1
        done
    done
done

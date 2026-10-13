#!/usr/bin/env bash
# Offline oracle/added-value gates only. No mono inference after a failed gate.
set -euo pipefail
cd "$(dirname "$0")/.."
root="$(realpath -m ../p48-out)"
work="$(realpath ..)"
mkdir -p "$root/logs" "$root/reference"
export TORCH_HOME="$root/cache/torch" LIBGL_ALWAYS_SOFTWARE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
python="$root/venv/bin/python"
"$python" e2e/measure.py --work "$work" --out "$root/reference" --prepare \
    > "$root/logs/reference.log" 2>&1
# Mesa is the portable rerun; the archived hardware captures used viewcache_gpu.sh.
"$python" e2e/viewcache_render.py --work "$work" --out "$root" \
    > "$root/logs/oracle-render.log" 2>&1
"$python" e2e/viewcache_score.py "$root/rendered/truth-oracle" \
    > "$root/logs/oracle-score.log" 2>&1
"$python" - "$root" "$work" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
root, work = map(Path, sys.argv[1:])
cache=json.loads((root/'rendered/truth-oracle/metrics.json').read_text())
main=json.loads((work/'m1-out/verified/repeat-1/metrics.json').read_text())
for run in ('sweep180','tour'):
    rows=[r for r in main['moments'] if r['run']==run and r['frame']!=526
          and r['view'] in ('left','right')]
    c=cache['means'][run]
    psnr=c['psnr_db']-np.mean([r['psnr_db'] for r in rows])
    lpips=1-c['lpips']/np.mean([r['lpips'] for r in rows])
    if psnr<2 and lpips<.2:
        raise SystemExit(f'STOP at stage 1: {run}, {psnr:.2f} dB / {lpips:.1%} LPIPS gain')
print('Stage 1 numeric gate passes; inspect camera/eyes sheets for sharpness')
PY
"$python" e2e/viewcache_clips.py --work "$work" --out "$root" \
    > "$root/logs/oracle-clips-render.log" 2>&1
"$python" e2e/viewcache_clip_score.py "$root/clips/truth-oracle" --work "$work" \
    > "$root/logs/oracle-clips-score.log" 2>&1
"$python" e2e/viewcache_audit.py --work "$work" --root "$root" \
    > "$root/logs/audit.log" 2>&1
bash e2e/viewcache_viewer.sh > "$root/logs/viewer-e2e.log" 2>&1
"$python" - "$root" <<'PY'
import json,sys
from pathlib import Path
r=json.loads((Path(sys.argv[1])/'results.json').read_text())
print(f"Stage 1: {r['stage1']}; stage 2: {r['stage2']}. {r['verdict']}")
PY

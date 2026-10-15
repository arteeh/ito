#!/usr/bin/env bash
# Existing beta oracle captures -> causal sensor arms and labelled oracle/point replays.
# bash e2e/observations_suite.sh RECORDINGS OUT
# Set ITO_OBSERVATION_PYTHON to the built Windows release's runtime/python.exe.
set -euo pipefail
cd "$(dirname "$0")/.."
recordings=$(realpath "$1")
output=$(realpath -m "$2")
runtime=${ITO_OBSERVATION_PYTHON:-/mnt/c/Users/me/Projects/ito/builds/feat-obs-42/ito/runtime/python.exe}
guard() {
    .venv/bin/python e2e/disk_guard.py --owned "$PWD/.venv" \
        --owned "$HOME/.local/share/ito-microduck" --owned "$recordings" --owned "$output" \
        --owned "$HOME/.cache/torch_extensions" \
        --windows-build "$(dirname "$(dirname "$(dirname "$runtime")")")"
}
for sequence in ${ITO_OBSERVATION_SEQUENCES:-sweep180 tour}; do
    duration=20
    if [[ "$sequence" == tour ]]; then duration=90; fi
    for arm in camera imu imu-tof prior; do
        estimates="$output/estimates/$sequence-$arm"
        mkdir -p "$estimates"
        guard
        flock "$HOME/Projects/ito/ceres.lock" "$runtime" -u -I \
            "$(wslpath -w "$PWD/e2e/oracle_estimate.py")" \
            "$(wslpath -w "$recordings/$sequence")" \
            --models 'C:\Users\me\Projects\ito\models-v5' \
            --out "$(wslpath -w "$estimates")" --observations "$arm" --seconds "$duration" \
            > "$output/estimate-$sequence-$arm.log" 2>&1
        guard
        .venv/bin/python e2e/estimator_report.py "$estimates" \
            --out "$output/report-$sequence-$arm.json" > "$output/report-$sequence-$arm.log"
        if [[ "$arm" == imu-tof ]]; then
            .venv/bin/python e2e/estimator_reprojection.py "$estimates" \
                --camera "$recordings/$sequence/source/camera.json" --matcher sift \
                --out "$output/reprojection-$sequence-$arm.json" \
                > "$output/reprojection-$sequence-$arm.log"
        fi
        mode=EE
        options=(--mapper-revision a7b8ea0)
        if [[ "$arm" == prior ]]; then mode=main-EE; options=(--depth-support); fi
        guard
        bash e2e/oracle.sh e2e/oracle_run.py "$recordings/$sequence" \
            --estimates "$estimates" --arm "$mode" --out "$output/visual/$sequence-$arm" \
            --seconds "$duration" --view-start 0 --record --no-snapshots "${options[@]}" \
            > "$output/visual-$sequence-$arm.log" 2>&1
        guard
        # Only the current arm owns depth stacks. JPEG motion evidence supports sheets.
        find "$estimates" -maxdepth 1 -type f -name '*.npz' -delete
        guard
    done
    moments=(5 12 18)
    if [[ "$sequence" == tour ]]; then moments=(12 50 76); fi
    for arm in imu imu-tof; do
        .venv/bin/python e2e/estimator_sheets.py "$recordings/$sequence" \
            --main "$output/visual/$sequence-camera" --branch "$output/visual/$sequence-$arm" \
            --labels "camera,$arm" --commits "$(git rev-parse --short HEAD),$(git rev-parse --short HEAD)" \
            --at "${moments[@]}" --out "$output/sheets/$sequence-$arm"
    done
done

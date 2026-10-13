# Motion sheets: judging Ito in motion

Numbers can pass while Ito looks bad. Motion sheets let people and vision models (Claude,
GPT) judge what the pilot actually saw, frame by frame, during a live or simulated run.

## Record

Set `ITO_RECORD=<folder>` before starting Ito (desktop pilot). Ito writes:

- `pilot/`: the drawn world view (before the overlay), up to `ITO_RECORD_FPS` (default 30)
- `camera/`: every robot camera frame Ito received
- `frames.jsonl` in each: one line per frame with time, pilot input, robot yaw, look, scene age

Reading pixels back costs a few ms per frame: don't record during timing measurements
(#39 GPU budget runs).

## Make sheets

    uv run python e2e/motion_sheet.py REC                      # one run
    uv run python e2e/motion_sheet.py REC_MAIN --compare REC_BRANCH --labels main,branch

Writes `REC/sheets/`: `SHEETS.md` (start here), `sheet-NNN.jpg`, `overview-NN.jpg`,
`sheets.json`, `motion.mp4` (needs opencv). Options: `--step-ms 50` for fast motion,
`--rows`, `--start/--end` seconds.

Sheet layout, built for vision models: one row per moment (default 100 ms apart), read top to
bottom; columns ROBOT CAMERA | PILOT VIEW (per run) | CHANGE (vs the row above, x4). Every
cell is the newest frame that existed at that moment, so stalls show as red REPEAT cells.
Each sheet stays under ~1.5 MP so labels survive model downscaling.

## Rule for verdicts

- Any change to what the pilot sees gets a recorded run and sheets, against main on the same
  drive (`--compare`).
- Verdicts cite cells (`sheet 004 row 3`). Numbers say where to look; the sheets decide.
- If the numbers pass but the sheets look worse than main, it fails.

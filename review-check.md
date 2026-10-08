# review-fixes check (origin/review-fixes at 61d9b06, on origin/microduck-scene)

Judged against AGENTS.md and review-9313372.md. No commit adds unit tests, a docs tree, dead
test hooks in product code or "for now" paths; README edits are usage lines only. Ruff check
and format pass. Not for merging: review notes only.

## Verdict per commit

- 9ecbf3e KEEP: ruff clean; the dropped `timeout` arg is never passed by aioice (it calls with component/addresses only).
- 98ea510 FIX: right fix for P2-1, but wire/input times still use `time.monotonic` in `drivers/microduck/camera.py:79` (capture_time on the wire), `drivers/microduck/adapter.py`, `ito/app/frames.py:29`, and after the merge `ito/desktop/overlay.py:187` (from 2d18b71). Switch them to `ito.clock.now` so "one clock" holds on Windows too.
- d5333f5 FIX: closes P1-3 replay/offline reuse and P2-6 lockout (pairing, webrtc, connect e2e pass). Fix: (a) a used code never pairs again and rotate/regeneration forget all pilots, so `.pilots` holds at most one pilot: `MAX_PILOTS` and the dict are unreachable; store one `{used, pilot, secret}` (one pilot is the product). (b) a bad *answer* proof raises `PairingError`, and `Pilot._run` then deletes the stored credential: anything answering at that address (spoof, DHCP reuse) wipes it. Only forget on the driver's 401/403. Residual, say it in the commit: the first pairing is still crackable in ~1.4-6 s, so an active MITM during that one exchange can take the secret; a PAKE is the full fix.
- 8e47702 FIX: separate unordered channel is right, but `e2e/depth_load.py` passed only 3 of 6 runs here (Status gaps 1079, 1103 and 2061 ms; one run got 1 depth frame in 10 s). All channels share one SCTP association and send queue, and `FRAME_BUFFER` (4 MB, about 0.5 s of depth) lets depth queue ahead of Status. Fix: in `Peer.send`, drop FrameMetadata while `frame_channel.bufferedAmount` exceeds about one frame (`MAX_MESSAGE_BYTES`), then require depth_load to pass 10/10. Expect a conflict with the clock-channel change in progress on microduck-scene (`channels`, `_bind`, `_opened`).
- 0c41e6b KEEP: openxr on Monado reports `sorts_per_frame_max: 1`; uploads run before the mid-eye sort, so the order is never stale.
- fdf0c11 KEEP: anchor_glide passes; jumps over 1 m and gaps over 0.6 s snap; rotation stays the newest.
- 2471800 KEEP: removes a test signal as a product default; audio e2e passes `tone:440` explicitly.
- 04284bc KEEP: psutil is in the dev group; slam (no CUDA) passes.
- 475b166 KEEP: strengthens safety (driver starts stopped and stops every new connection; e-stop latch untouched). Pairs with 2d18b71's stopped prompt: merged e2e/app.py passes "no auto-resume after stop/e-stop".
- 398313a KEEP: the done-callback latches fault, calls `adapter.neutral()` directly (does not need the dead loop) and exits 1; lifecycle injection passes. Nit: `main()` now turns every RuntimeError into a one-line exit.
- 5872a14 KEEP: simulated passes (killed app ends the driver). PDEATHSIG follows the spawning *thread*: the viewer is spawned on `ito-display`, which ends with the pilot window, when SimulatedRobot closes anyway. Fine today; don't move spawning to a shorter-lived thread.
- 9ffafd5 KEEP: ran `notices()` on this Linux venv: 46 packages, PyAV's x264/x265/FFmpeg listed. ~100 lines of build code; LICENSE, (L)GPL source offer and MASt3R NC stay owner decisions.
- 7bbd9de KEEP: app e2e passes (worker killed mid-drive, link kept, scene rebuilt). Nit: `restart_delay` never resets after a healthy run, so a late failure can wait 30 s.
- d578c9e KEEP (unverified): logic is sound; needs `e2e/slam.py --cuda`, which no machine here can run.
- 50d9262 FIX: fixes trigger leakage, but (a) "on panel" means anywhere on the 1.05 x 0.60 m quad 1.2 m ahead (about 47 x 28 degrees, `_pointer`), so a controller aimed roughly forward shows the panel and zeroes *both* triggers: the Microduck beak stops working whenever the pilot points ahead. Use ImGui's hover from the previous frame (`want_capture_mouse`), and zero only the pointing hand. (b) `panel_visible` ignores stopped/fault/not armed. Since 475b166 every connection starts stopped, and 2d18b71 puts "Robot stopped, press A to drive" in that panel, so in VR the pilot never sees why the robot won't move. Show it while `not status.armed` or the robot is stopped/faulted. (c) gaze no longer aims, so a runtime without an aim pose cannot reach the panel; accept that knowingly.
- 61d9b06 KEEP: correct; fold into the d5333f5 fix.

Not addressed by any commit: P1-1 (double head turn on Microduck), P1-2 (Microduck pts join
drops about half the frames), P1-5 (one refused RPC faults the Microduck driver), P1-4 (torch
cu124 vs Blackwell), VISION.md vs AGENTS.md. P1-1 and P1-2 matter most for comfort.

## e2e on review-fixes (Linux container, no GPU, Xvfb :97, Mesa llvmpipe, OSMesa)

Passed: webrtc, pairing, connect, lifecycle, mujoco_driver, anchor_glide, render, desktop,
walking, stream, view_motion, diagnostics, reconstruction_faults, app (63 s),
reconstruction (124 s), slam (no CUDA), simulated (after installing xdotool),
openxr (Monado 21, simulated HMD, null compositor, lavapipe), audio (JACK dummy).

Flaky: depth_load passed 3 of 6 runs (see 8e47702).

Skipped: release.py and bundle.py (Windows only / `slam` extra), slam.py --cuda and d578c9e's
path (no NVIDIA GPU), microduck*.py (need Pollen's upstream checkouts and policies). No real
headset or GPU, so the frame-time and comfort claims of 0c41e6b and fdf0c11 are unmeasured.
OSMesa was not at /opt/data/lib/osmesa; the system libosmesa6 was used.

## Merge conflicts (`git merge-tree --write-tree origin/microduck-scene origin/review-fixes`)

- e2e/app.py only: the `nonlocal` list, `time.monotonic` vs `clock.now`, and the run budget
  (65 base; microduck-scene went to 80, review-fixes to 75). Resolve as
  `nonlocal steady_start, steady_end, worker_restart_s`, `now = clock.now()`, budget 90.
  With that resolution the merged tree passes e2e/app.py, including the stopped prompt,
  focus-loss resume, no auto-resume and worker-restart stages.
- Clean auto-merges: pilot.py, desktop/{dispatch,input,window}.py, xr/window.py,
  e2e/stream.py. Semantic clashes git cannot see: overlay.py's `time.monotonic`
  (98ea510 FIX) and the hidden VR panel vs the stopped prompt (50d9262 FIX b).
- Expected: the in-progress link/peer.py clock-channel change on microduck-scene will
  conflict with 8e47702 (both touch channel setup and `_opened`).

## Recommended merge plan

1. Finish and push the peer.py clock-channel change on microduck-scene first, so review-fixes
   rebases its link work onto it once.
2. On review-fixes, add follow-up commits (do not rewrite the 16): the 8e47702 send-side
   depth drop, the 50d9262 hover and visibility fixes, the d5333f5 single-pilot store and
   answer-proof handling, and the 98ea510 clock stragglers.
3. Merge microduck-scene into review-fixes with the e2e/app.py resolution above, convert
   overlay.py to `clock.now`, then run the full e2e list (depth_load 10 times, openxr on
   Monado) on the merged tree.
4. Fast-forward microduck-scene to the result. Then take P1-1, P1-2 and P1-5 as the next
   Microduck branch, before any pilot uses Microduck in VR.

# Microduck held-head motion (#66)

The camera contains real movement. The apparent alternating frames in the original
report were a reduced-contact-sheet impression: `purge-3` frames 700–703 move
progressively, and the view becomes steadier around 850.

Instrumenting the live stack found two contributions: continued gaze corrections
from Ito while the policy acquires a fixed pilot target, and smaller physical
settling that continues with an exactly constant head intent. This change corrects
a command-coordinate bug found on that path. It does **not** make the simulated
robot perfectly motionless, and should not be treated as proof that #66 is fully
resolved.

## Trace and interventions

The baseline used Windows build `feat-merge-purge`, the beta sim install, and this
checkout at `f062dca`. Temporary instrumentation logged pilot inputs, outgoing RPCs,
robot.state (including head intent, motor targets, joints and camera FK), and each
camera snapshot's simulation time, qpos and motor controls. These probes only logged
values and were removed after the investigation. No image smoothing, stabilization,
frame dropping or control tuning was added.

- Before resume, Ito resends the same zero **offset** head command about every
  0.5 seconds. The policy stays active. There is small body/joint movement and
  passive head-yaw play despite an unchanged head intent.
- Resume starts a new world-space gaze target. In the original tour, identity pilot
  orientation can require a large turn: the driver's yaw origin was captured while
  the simulated robot was getting upright. Keeping the mouse still after resume
  does not cancel that target acquisition.
- During active piloting, `_commands` refreshes `_head_at` and `_hold_head` does not
  send. The pilot pose is constant in the first standing step, but the measured
  camera direction is not, so `TiltTrim.measured` continues changing pitch/pan trim.
  Near the pan limit, small elevation corrections require large pitch-joint changes
  through gaze IK. The default baseline's head pitch moves from about 20.5° at two
  seconds after resume to 7.3° at three seconds, then settles further. The camera
  snapshots contain that motion; it is not a mediad or recorder reorder.
- Disabling trim learning for diagnosis leaves a large aiming error and still does
  not stop the smaller drift. That intervention was discarded.
- In a separate live run, from two seconds after resume onward, Ito repeatedly sent
  one captured `robot.head` intent and an exactly zero `robot.move`. Neck/pitch/yaw
  intents stayed constant. From seconds 3–4 to 12–13 after resume, mean neck position
  changed 13.687° → 13.128°, head yaw −74.662° → −75.190°, and true body yaw
  95.613° → 95.105°. Passive pitch/yaw joints stayed against the same approximately
  +1° stops. Motor yaw targets changed −75.648° → −75.854°: the learned policy was
  still adjusting its output, and the physical joints/body were still settling.
  The diagnostic camera frames 700–880 show this smaller drift. Constant fresh head
  intent also prevents robotd's idle animation from becoming eligible.

These numbers locate the movement; the frame inspection supplies the visual
judgment. The constant-intent run establishes a policy/plant contribution rather
than proving that any particular real robot has the same amplitude or duration.

## Command-coordinate correction

The policy is trained on **head offsets from home**, not absolute joint angles:
`microduck_rl/src/mjlab_microduck/tasks/mdp.py`, `head_pose_tracking`, compares
`measured - default_joint_pos` with the command. Its bias reward uses the same
convention. `duck-control::DEFAULT_POSITION` has 0.3491 rad (20°) for both neck
pitch and head pitch.

In pinned microduck `175045d`, `robotd/src/main.rs` passes `HeadFk::look_at`'s
**absolute** joint solution directly to `intents.set_head`. That mistakes both
pitch angles for home-relative offsets. Forward kinematics takes absolute angles;
`Observation::build` passes the head command through without conversion.

Microduck commit `03e3fd0` subtracts the home joints before setting head intent and
returns those same offsets in `robot.look.result.head`. Resending that result with
`robot.head` still holds exactly the same command. `robot.look.neck_pitch` remains
an absolute posture angle. API 45 and the protocol/design documentation state the
units; a hardware installation needs the updated daemon too. The existing posture
assertion was adjusted to account for the returned offset; no new tests were added
or automated tests run.

Ito only bumps the Microduck pin and its description. Its adaptive trim is retained:
removing it to make the recording look quieter would sacrifice aiming accuracy.
The conversion fixes a real API/policy mismatch; it is not a sufficient explanation
or cure for all of the settling seen in the recordings.

## Validation and limits

The isolated `~/.local/share/ito-microduck-66` install was built with
`drivers/microduck/setup-sim.sh`. Shared beta and ordinary sim installs were not
modified. The before/after walking policy is identical (velstand SHA-256
`2fc7f8ee6acd7d0d3f66a99927b5d5a56589aeed02ec714de329b252d5ccf492`).

The retained comparison uses the same room-tour script with one explicit recenter
before resume in both runs. This removes the variable startup heading correction
from the comparison; it is a diagnostic script action, not a runtime behavior change.
The original default tour was also run with each daemon. Its large initial turn and
settling remain, so the recentered comparison must not be read as a fix for that turn.

The retained sessions under `C:\Users\me\Projects\ito\out\sessions` are
`shake-66-aligned-before` (beta daemon) and `shake-66-aligned-after` (API 45 daemon).
Their `script.json` files preserve the exact input sequence. Camera frame numbers:

| Action | Before | After |
| --- | --- | --- |
| First standing step, consecutive frames | 700–707 | 698–705 |
| Same step, two-second intervals | 700, 760, 820, 880 | 698, 758, 818, 878 |
| Walk forward, one-second intervals | 1051, 1081, 1111, 1141 | 1049, 1079, 1109, 1139 |
| Hold left, two-second intervals | 1392, 1452, 1512, 1572 | 1390, 1450, 1510, 1570 |
| Hold right, two-second intervals | 1752, 1812, 1872, 1932 | 1748, 1808, 1868, 1928 |
| Hold up, one-second intervals | 2097, 2127, 2157 | 2093, 2123, 2153 |
| Hold down, one-second intervals | 2307, 2337, 2367 | 2303, 2333, 2363 |

Inspection with `tools/frame.py` shows small progressive movement during the first
standing step in both runs. The corrected daemon reduces the large roll swings
seen while holding left, but does not eliminate settling. Walking and all four look
directions still respond. During the first stand, the pitch trim needed for a level
aim falls from roughly 20° to 10°; that is supporting telemetry, not a visual pass
criterion. These runs do not establish that every pose has improved.

Temporary bulk traces and other diagnostic recordings were deleted after producing
summaries and contact sheets. Only this before/after recording pair is retained;
the pre-existing user recording `purge-3` was left untouched.

Headset and physical hardware were not tested. Camera pose from servo FK excludes
passive gear play; the logged MuJoCo qpos snapshots were used to distinguish it.
Improving the residual settling needs work on the policy/servo response and hardware
validation, not freezing targets or hiding camera motion in Ito.

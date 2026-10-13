# Layered camera-view cache — #48

Separate `proto/view-cache` experiment from `07ca8a3`, 13 October 2026.
Main's default pipeline is unchanged. No model inference or new recording was needed.

**Oracle gate passes; the added-value gate is being evaluated.** Original camera
textures recover book labels and floor grain, but source selection also produces
brightness islands on the wall and floor. This is not a pilot-comfort acceptance.

Every sheet has **sim truth | frozen main | #47 | view-cache**, at 1:1 projection.
Look at [sweep](proto-view-cache/img/oracle-sweep.jpg),
[tour](proto-view-cache/img/oracle-tour.jpg),
[sweep eyes](proto-view-cache/img/oracle-sweep-eyes.jpg),
[tour eyes](proto-view-cache/img/oracle-tour-eyes.jpg), and
[hard turn](proto-view-cache/img/hard-526.jpg).

Six held-out eye images per sequence, configured 64 mm IPD; hard #526 is separate.
Full-frame scores include holes. [Oracle evidence](proto-view-cache/oracle.json)
contains all 35 views and input hashes.

| Representation, true depth/pose for both candidates | Sweep PSNR / SSIM / LPIPS | Tour PSNR / SSIM / LPIPS |
|---|---|---|
| Frozen main, its own fused baseline | 17.33 / .734 / .524 | 18.92 / .778 / .513 |
| #47 | 19.74 / .836 / .245 | 23.33 / .878 / .271 |
| View cache | 19.47 / .827 / .291 | 22.49 / .852 / .357 |

The requested ≥2 dB or ≥20% LPIPS oracle gate passes on both sequences: +2.13/+3.57 dB,
44.4%/30.4% LPIPS reduction over main. Against #47, image quality is weaker: especially
near the wall, where appearance selection produces a visibly mottled surface.

`ito/proto/viewcache` retains at most 64 source views by viewpoint redundancy,
sharpness and angular/spatial coverage. Twelve candidates rasterize the same 90×160
projective grids as #47. A midpoint depth/quality pass chooses a front source and an
observed background source per pixel, with reprojected ownership hysteresis. Both eyes
use those choices with independent depth tests. Original 360×640 RGB remains the texture;
no Gaussians, appearance training, persistent fused surface or permanent retirement mask.

RGB and depth have separate bounded publication calls. Without verified correspondence,
a pending RGB image supplies no scene geometry. This experiment makes no independent-RGB
freshness claim and does not paste new RGB onto unrelated old depth.

Inputs, robot-pose intervention and queued inference/publication times come read only
from #47. Current evidence is oracle only. The new temporal comparison loads #47's
renderer read only at `932c7d6`, with a pinned SHA256, to generate synchronized clips;
it does not rerun #47's inference/evaluation sweep. Every candidate gets the same frames,
poses, depths and publication times. These virtual times include #47's declared 10 ms
update budget and are not measurements of live publication. Legacy IDs/poses retain the
baseline's thumbnail matching and centred principal-point limitation.

The saved-cache inspector passed its actual Xvfb run: keyboard movement, Home recenter,
F12 captures and the Dear ImGui stereo checkbox. Fly an archived cache with:

```sh
../p48-out/venv/bin/python -m ito.proto.viewcache ../p48-out/rendered/truth-oracle/tour-world.npz
```

WASD, PgUp/PgDn, Tab mouse-look, Home recenter, ImGui 64 mm stereo and FOV control.
This is desktop inspection of a saved cache, not a live robot or VR path.

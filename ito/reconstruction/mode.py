"""How frames are placed, read once for every module.

Default: camera-first full MASt3R-SLAM, as upstream designed it. Every frame is tracked against a
keyframe by the images alone (vision_full.py), every keyframe joins one global Sim(3) graph solved
by upstream's dense calibrated Gauss-Newton with retrieval loop closure and relocalisation
(mast3r_global.py), and the map follows the keyframes. The robot only seeds each frame's turn,
places the first camera in its world, gives the map its metric scale (ToF and floor) and bridges
frames the images cannot place (dead reckoning).

Developer opt-outs, for comparison only:
- ITO_SLAM_LEAD=robot: full SLAM on fusion.py's robot-first window (frames tied by the robot's
  motion, per-frame ToF scale, gravity, height and odometry in the global graph).
- ITO_SLAM=vision: vision_track.py without the global graph (no loop closure).
- ITO_SLAM=window: the old robot-first 30-keyframe pose graph alone (no global graph)."""

import os

SLAM = os.environ.get("ITO_SLAM") or "full"
LEAD = os.environ.get("ITO_SLAM_LEAD") or "vision"
FULL = SLAM == "full"
VISION = SLAM == "vision"
VISION_FULL = FULL and LEAD == "vision"
VISION_LED = VISION or VISION_FULL

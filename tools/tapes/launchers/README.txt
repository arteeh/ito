Ito sprint 2: drive the virtual Microduck and every drive becomes a tape

A tape is everything the robot sent while you drove (camera frames, the sim's exact camera
poses and depth, robot motion, head IMU, ToF). The agents replay every candidate on the same
tapes and film them side by side, so drives with your own turns, head twists and shake are the
most useful tapes there are.

1. Run "0 Start sim + record (pauses agents).cmd". It waits until the agents' current GPU job
   ends, then opens the virtual Microduck (same pairing code as always). Recording starts with
   the sim; the window says where the tape goes.
2. Run "1 Ito (for recording).cmd" and drive as you normally would. This Ito build knows the
   sim's new capture timestamps (an older Ito build refuses this sim). Its reconstruction is the
   current research default, nothing new to judge here.
3. Close the sim window (or Ctrl+C in it) to stop. Each sim start makes a new tape:
   WSL ~/Projects/ito/research/data/user-MMDD-HHMM. About 1 GB per minute of driving.
4. Run "resume agents.cmd" when you're done (otherwise they resume by themselves in 12 h).

Tell Unii which drives to keep and what you did in them (e.g. "fast turns by the sofa, then
stared at the shelf"); the rest get deleted.

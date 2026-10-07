"""Keep the CAD body visible in Pollen's viewer without occluding its head camera."""

import sys

import mujoco
from mjlab_microduck.sim import body_server
from mjlab_microduck.sim.camera import Camera, to_uyvy


class HeadCamera(Camera):
    def render(self, world):
        option = mujoco.MjvOption()
        option.geomgroup[2] = 0
        with world.lock:
            self.renderer.update_scene(world.data, camera=self.camera, scene_option=option)
        packed = to_uyvy(self.renderer.render())
        with self.lock:
            self.latest = packed


if __name__ == "__main__":
    if "--headless" not in sys.argv:
        import mujoco.viewer

        launch = mujoco.viewer.launch_passive

        def visible_viewer(*args, **kwargs):
            try:
                return launch(*args, **kwargs)
            except Exception as exc:
                raise SystemExit(f"MuJoCo viewer unavailable: {exc}") from exc

        mujoco.viewer.launch_passive = visible_viewer
    body_server.Camera = HeadCamera
    body_server.main()

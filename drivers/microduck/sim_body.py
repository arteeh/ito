"""Keep the CAD body visible in Pollen's viewer without occluding its head camera."""

import os
import sys

import mujoco
from mjlab_microduck.sim import body_server
from mjlab_microduck.sim.camera import Camera, to_uyvy

from drivers.microduck import lens


class HeadCamera(Camera):
    def __init__(self, model, name):
        height = int(os.environ.get("ITO_MICRODUCK_CAMERA_HEIGHT", 360))
        width = lens.width_for(height)
        # MuJoCo renders no larger than the offscreen buffer, which the scene sizes at 640x480.
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        model.cam_fovy[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)] = (
            lens.vertical_fov_deg(width, height)
        )
        super().__init__(model, name, width, height)

    def render(self, world):
        option = mujoco.MjvOption()
        option.geomgroup[2] = 0
        option.geomgroup[5] = 1  # Room cutaway walls are hidden only in the overview viewer.
        with world.lock:
            self.renderer.update_scene(world.data, camera=self.camera, scene_option=option)
        # MuJoCo's shadow pass produces dark floor triangles at this low camera height.
        # Keep the head image clean without changing the viewer or the robot's physics.
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        packed = to_uyvy(self.renderer.render())
        with self.lock:
            self.latest = packed


if __name__ == "__main__":
    if "--headless" not in sys.argv:
        import mujoco.viewer

        launch = mujoco.viewer.launch_passive

        def visible_viewer(model, data, **kwargs):
            try:
                viewer = launch(model, data, **kwargs)
                with viewer.lock():
                    mujoco.mjv_defaultFreeCamera(model, viewer.cam)
                return viewer
            except Exception as exc:
                raise SystemExit(f"MuJoCo viewer unavailable: {exc}") from exc

        mujoco.viewer.launch_passive = visible_viewer
    body_server.Camera = HeadCamera
    body_server.main()

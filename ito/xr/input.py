"""Action bindings are shared across both hands; inactive tracking is never sent."""

import logging

import numpy as np
import xr
from xr.utils import Matrix4x4f

from ito import clock
from ito.desktop import PilotInput
from ito.protocol import Pose
from ito.render import pose

log = logging.getLogger(__name__)
VALID = xr.SpaceLocationFlags.POSITION_VALID_BIT | xr.SpaceLocationFlags.ORIENTATION_VALID_BIT
TRACKED = xr.SpaceLocationFlags.POSITION_TRACKED_BIT | xr.SpaceLocationFlags.ORIENTATION_TRACKED_BIT


def matrix(value):
    result = Matrix4x4f.create_from_quaternion(value.orientation).as_numpy()
    result[:3, 3] = value.position.as_numpy()
    return result


def protocol_pose(value):
    return Pose(
        position=tuple(map(float, value.position.as_numpy())),
        orientation=tuple(map(float, value.orientation.as_numpy())),
    )


class Actions:
    def __init__(self, owner):
        self.owner = owner
        self.instance, self.session = owner.instance, owner.session
        self.paths = {hand: self.path(f"/user/hand/{hand}") for hand in ("left", "right")}
        self.set = xr.create_action_set(
            self.instance,
            xr.ActionSetCreateInfo(action_set_name="pilot", localized_action_set_name="Ito pilot"),
        )
        owner.resources.callback(xr.destroy_action_set, self.set)
        self.actions = {}
        for name, kind in (
            ("grip", xr.ActionType.POSE_INPUT),
            ("aim", xr.ActionType.POSE_INPUT),
            ("stick", xr.ActionType.VECTOR2F_INPUT),
            ("trigger", xr.ActionType.FLOAT_INPUT),
            ("squeeze", xr.ActionType.FLOAT_INPUT),
            ("resume", xr.ActionType.BOOLEAN_INPUT),
            ("e_stop", xr.ActionType.BOOLEAN_INPUT),
            ("stop", xr.ActionType.BOOLEAN_INPUT),
            ("recenter", xr.ActionType.BOOLEAN_INPUT),
            ("panel", xr.ActionType.BOOLEAN_INPUT),
            ("haptic", xr.ActionType.VIBRATION_OUTPUT),
        ):
            self.actions[name] = self.create(name, kind, list(self.paths.values()))
        self.spaces = {
            (hand, name): self.action_space(self.actions[name], path)
            for hand, path in self.paths.items()
            for name in ("grip", "aim")
        }
        common = {"grip": "input/grip/pose", "aim": "input/aim/pose", "haptic": "output/haptic"}
        touch = {
            **common,
            "stick": "input/thumbstick",
            "trigger": "input/trigger/value",
            "squeeze": "input/squeeze/value",
        }
        face = {
            ("left", "stop"): "input/x/click",
            ("left", "recenter"): "input/y/click",
            ("right", "resume"): "input/a/click",
            ("right", "e_stop"): "input/b/click",
            ("left", "panel"): "input/thumbstick/click",
        }
        self.suggest("/interaction_profiles/oculus/touch_controller", touch, face)
        self.suggest(
            "/interaction_profiles/valve/index_controller",
            touch | {"squeeze": "input/squeeze/force"},
            {
                ("left", "stop"): "input/a/click",
                ("left", "recenter"): "input/b/click",
                ("right", "resume"): "input/a/click",
                ("right", "e_stop"): "input/b/click",
                ("left", "panel"): "input/thumbstick/click",
            },
        )
        self.suggest(
            "/interaction_profiles/htc/vive_controller",
            common
            | {
                "stick": "input/trackpad",
                "trigger": "input/trigger/value",
                "squeeze": "input/squeeze/click",
            },
            {("left", "recenter"): "input/menu/click", ("right", "e_stop"): "input/menu/click"},
        )
        self.suggest(
            "/interaction_profiles/microsoft/motion_controller",
            common
            | {
                "stick": "input/thumbstick",
                "trigger": "input/trigger/value",
                "squeeze": "input/squeeze/click",
            },
            {
                ("left", "recenter"): "input/menu/click",
                ("right", "e_stop"): "input/menu/click",
                ("left", "panel"): "input/thumbstick/click",
            },
        )
        self.suggest(
            "/interaction_profiles/khr/simple_controller",
            common
            | {
                "trigger": "input/select/click",
            },
            {("left", "recenter"): "input/menu/click", ("right", "e_stop"): "input/menu/click"},
        )
        self.trackers = {}
        if "XR_HTCX_vive_tracker_interaction" in owner.extensions:
            roles = (
                "handheld_object",
                "left_foot",
                "right_foot",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_knee",
                "right_knee",
                "waist",
                "chest",
                "camera",
                "keyboard",
            )
            paths = {role: self.path(f"/user/vive_tracker_htcx/role/{role}") for role in roles}
            action = self.create("trackers", xr.ActionType.POSE_INPUT, list(paths.values()))
            bindings = [
                xr.ActionSuggestedBinding(
                    action=action,
                    binding=self.path(f"/user/vive_tracker_htcx/role/{role}/input/grip/pose"),
                )
                for role in roles
            ]
            xr.suggest_interaction_profile_bindings(
                self.instance,
                xr.InteractionProfileSuggestedBinding(
                    interaction_profile=self.path("/interaction_profiles/htc/vive_tracker_htcx"),
                    suggested_bindings=bindings,
                ),
            )
            self.trackers = {
                role: (action, path, self.action_space(action, path))
                for role, path in paths.items()
            }
        xr.attach_session_action_sets(
            self.session, xr.SessionActionSetsAttachInfo(action_sets=[self.set])
        )
        self.previous = set()
        self.head_flags = 0
        self.focused = False
        self.aims = {}
        self.triggers = {}
        self.panel_held = False
        self.panel_pinned = False  # The left stick click keeps the panel in view.
        self.haptic_pulses = 0
        self.haptic_pending = False

    def path(self, name):
        return xr.string_to_path(self.instance, name)

    def create(self, name, kind, paths):
        return xr.create_action(
            self.set,
            xr.ActionCreateInfo(
                action_name=name,
                localized_action_name=name.replace("_", " ").title(),
                action_type=kind,
                subaction_paths=paths,
            ),
        )

    def action_space(self, action, path):
        space = xr.create_action_space(
            self.session,
            xr.ActionSpaceCreateInfo(
                action=action, subaction_path=path, pose_in_action_space=xr.Posef()
            ),
        )
        self.owner.resources.callback(xr.destroy_space, space)
        return space

    def suggest(self, profile, common, specific):
        bindings = []
        for hand in self.paths:
            pairs = common | {name: path for (side, name), path in specific.items() if side == hand}
            for name, suffix in pairs.items():
                bindings.append(
                    xr.ActionSuggestedBinding(
                        action=self.actions[name], binding=self.path(f"/user/hand/{hand}/{suffix}")
                    )
                )
        try:
            xr.suggest_interaction_profile_bindings(
                self.instance,
                xr.InteractionProfileSuggestedBinding(
                    interaction_profile=self.path(profile), suggested_bindings=bindings
                ),
            )
        except xr.PathUnsupportedError:
            log.info("OpenXR runtime does not support %s", profile)

    def located(self, action, path, space, at):
        active = xr.get_action_state_pose(
            self.session, xr.ActionStateGetInfo(action=action, subaction_path=path)
        )
        if active.is_active:
            location = xr.locate_space(space, self.owner.space, at)
            if location.location_flags & (VALID | TRACKED) == (VALID | TRACKED):
                return location.pose
        return None

    def poll(self, at):
        sampled_at = clock.now()
        focused = self.owner.state == xr.SessionState.FOCUSED
        location = xr.locate_space(self.owner.head, self.owner.space, at)
        self.head_flags = location.location_flags
        # A 3DoF HMD can supply a valid inferred position without position tracking.
        valid = location.location_flags & VALID == VALID
        tracked = bool(location.location_flags & xr.SpaceLocationFlags.ORIENTATION_TRACKED_BIT)
        head = matrix(location.pose) if valid else pose()
        hands, trackers, axes, buttons = {}, {}, {}, set()
        self.aims, self.triggers = {}, {}
        if focused:
            try:
                xr.sync_actions(
                    self.session,
                    xr.ActionsSyncInfo(active_action_sets=[xr.ActiveActionSet(self.set)]),
                )
            except xr.SessionNotFocused:
                focused = False  # Focus can change between polling events and syncing actions.
        panel = False
        if focused:
            if self.haptic_pending:
                self.pulse()
            for hand, path in self.paths.items():
                for name in ("grip", "aim"):
                    value = self.located(self.actions[name], path, self.spaces[hand, name], at)
                    if value is not None:
                        if name == "grip":
                            hands[hand] = protocol_pose(value)
                        else:
                            self.aims[hand] = matrix(value)
                for name, action in self.actions.items():
                    info = xr.ActionStateGetInfo(action=action, subaction_path=path)
                    if name == "panel":
                        state = xr.get_action_state_boolean(self.session, info)
                        panel |= bool(state.is_active and state.current_state)
                    elif name in ("resume", "e_stop", "stop", "recenter"):
                        state = xr.get_action_state_boolean(self.session, info)
                        if state.is_active and state.current_state:
                            buttons.add(f"{hand}_{name}")
                    elif name in ("trigger", "squeeze"):
                        state = xr.get_action_state_float(self.session, info)
                        value = (
                            float(np.clip(state.current_state, 0, 1)) if state.is_active else 0.0
                        )
                        axes[f"{hand}_{name}"] = value
                        if value > 0.7:
                            buttons.add(f"{hand}_{name}")
                        if name == "trigger":
                            self.triggers[hand] = value > 0.7
                    elif name == "stick":
                        state = xr.get_action_state_vector2f(self.session, info)
                        for axis in ("x", "y"):
                            value = (
                                float(np.clip(getattr(state.current_state, axis), -1, 1))
                                if state.is_active
                                else 0.0
                            )
                            axes[f"{hand}_stick_{axis}"] = value if abs(value) > 0.15 else 0.0
            for role, (action, path, space) in self.trackers.items():
                value = self.located(action, path, space, at)
                if value is not None:
                    trackers[role] = protocol_pose(value)
        commands = tuple(
            name
            for name in ("resume", "stop", "recenter", "e_stop")
            if (name != "resume" or self.focused)
            and any(f"{hand}_{name}" in buttons - self.previous for hand in self.paths)
        )
        self.previous = buttons
        self.focused = focused
        if panel and not self.panel_held:
            self.panel_pinned = not self.panel_pinned
        self.panel_held = panel
        return PilotInput(
            sampled_at,
            head,
            (axes.get("left_stick_x", 0.0), 0.0, axes.get("left_stick_y", 0.0)),
            (axes.get("right_stick_x", 0.0), axes.get("right_stick_y", 0.0)),
            frozenset(buttons),
            commands,
            focused and valid and tracked,
            hands=hands,
            trackers=trackers,
            axes=axes,
        )

    def pulse(self):
        self.haptic_pending = True
        if self.owner.state != xr.SessionState.FOCUSED:
            return
        try:
            for path in self.paths.values():
                xr.apply_haptic_feedback(
                    self.session,
                    xr.HapticActionInfo(action=self.actions["haptic"], subaction_path=path),
                    xr.HapticVibration(
                        duration=150_000_000, frequency=xr.FREQUENCY_UNSPECIFIED, amplitude=0.8
                    ),
                )
        except xr.SessionNotFocused:
            return
        self.haptic_pending = False
        self.haptic_pulses += 1
        log.info("OpenXR safety haptic pulse")

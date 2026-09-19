"""Third-person follow camera.

The released behaviour, re-measured per frame over window 0 rather than in aggregate: the
subject is held at a stable screen position AND at a stable size. On ex30 their subject-x
sd is 0.003 of frame width, subject-y 0.001, and the area coefficient of variation 0.021 --
a tight, steady follow for all 124 frames.

An earlier reading of the same takes concluded the boom "breathes by ~23%" and should
therefore be soft. That was an artefact of aggregating across takes with different boom
lengths; per frame within a take it does not breathe. The two axes still get separate time
constants, because the aim must be stiffer than the boom, but the boom's job is to HOLD the
stand-off, not to lag behind it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .render import OUT_W, Camera


@dataclass
class FollowCamera:
    """Critically damped boom plus a separate, stiffer aim.

    `boom_tau` and `aim_tau` are time constants in seconds: the smaller, the stiffer.
    `shoulder_offset` is metres to the camera's right, which puts the subject
    off-centre the way the released framing does.
    """

    distance: float = 1.55
    height: float = 1.83
    pitch_deg: float = -8.0
    focal_px: float = 206.0   # the measured released intrinsic; see StreetSpec
    shoulder_offset: float = 0.45
    # Where the subject sits across the frame, 0 = left edge, 0.5 = centre, 1 = right edge.
    # This rotates the optical axis; `shoulder_offset` moves the camera body and the aim
    # then cancels it, which is why it never changed the screen position.
    aim_screen_x: float = 0.5
    boom_tau: float = 0.55
    aim_tau: float = 0.12
    zeta: float = 0.55          # boom damping ratio; < 1 overshoots, 1 is critical
    aim_height: float = 1.20    # metres above the subject's feet that the camera looks at
    pitch_min: float = -35.0    # clamp on the optical axis; +ve is up, and it must be
    pitch_max: float = 35.0     # able to go up or a climbing subject leaves the frame
    # Vertical response to a climbing subject. At the defaults (rise_gain 1, pull_back 0)
    # the boom is rigidly attached and these are exactly the old geometry, so every take
    # that does not leave the ground is bit-identical.
    # Shot type, settable per frame. "follow" is the boom rig and the default, so a take
    # that never touches this is unchanged.
    #   hold   - the body stops where it is; the aim keeps tracking the subject, so the
    #            viewpoint is fixed but the framing is not ("a close, steady vantage while
    #            she rotates")
    #   lock   - locked off: body AND aim frozen, so the whole layout is preserved and the
    #            subject drifts across the frame ("preserving the exact initial layout")
    #   orbit  - the boom swings around the subject at `orbit_rate` deg/s
    #   push   - the boom shortens toward `push_to` at `push_rate` m/s
    #   pull   - the boom lengthens toward `push_to`
    mode: str = "follow"
    orbit_rate: float = 24.0    # degrees per second
    push_rate: float = 2.0      # metres per second of boom change
    push_to: float = 8.0        # boom length push/pull aims for
    _locked: object = field(default=None, repr=False)
    _orbit: float = field(default=0.0, repr=False)
    _dist: float | None = field(default=None, repr=False)
    ground_y: float = 0.0       # world y of the road (+Y is down, so the ground is 0)
    rise_gain: float = 0.30     # fraction of the subject's altitude the camera climbs
    pull_back: float = 2.50     # extra boom metres per metre of subject altitude
    # Deadband before either engages. A rider sits 1.16 m up in the car and 1.76 m up on
    # the camel, and those takes frame correctly today; without a knee above both, a gain
    # meant for a 14 m climb would pull the boom 3.5 m back for a man sitting on a camel.
    rise_knee: float = 2.50
    _pos: np.ndarray | None = field(default=None, repr=False)
    _vel: np.ndarray | None = field(default=None, repr=False)  # relative to the anchor
    _off: np.ndarray | None = field(default=None, repr=False)  # camera pos minus anchor
    _yaw: float | None = field(default=None, repr=False)

    def _maybe_lock(self, cam: Camera) -> Camera:
        """Remember the first pose taken while locked, so later frames can reuse it."""
        if self.mode == "lock" and self._locked is None:
            self._locked = cam
        return cam

    def _yaw_offset(self) -> float:
        """Degrees to rotate off the subject so it lands at `aim_screen_x`.

        u = cx/cz * f + W/2, and increasing yaw swings the subject LEFT in frame, so the
        sign is negative. `t_aim_screen_x` projects a subject back and asserts the fraction
        it actually lands at, which pins this rather than trusting the derivation.
        """
        if abs(self.aim_screen_x - 0.5) < 1e-9:
            return 0.0
        return -float(np.degrees(np.arctan(
            (self.aim_screen_x - 0.5) * OUT_W / self.focal_px)))

    def _desired(self, target: np.ndarray, heading_deg: float) -> np.ndarray:
        """Boom anchor: behind the subject along its heading, offset to one side.

        A boom rigidly attached in Y rises exactly as fast as the subject, which is why
        flight did not read as flight: the camera held the same 2.4 m stand-off and the
        same relative height all the way to 14 m, so the shot was a walk with the ground
        cropped out. Worse, once the axis was free to look up it looked at nothing --
        98% of the frame was empty sky and the conditioning went with it.

        So a climb does two things a walk does not: the camera keeps some of its altitude
        (rise_gain < 1, so the subject genuinely gains height in frame) and the boom
        lengthens (pull_back > 0, so the ground stays in shot underneath him). Both are
        driven by height above `ground_y`, so nothing changes while the subject walks.
        """
        # Orbit swings the boom's azimuth away from the subject's heading; push and pull
        # drive its length. Both are integrated in `update`, which owns dt.
        th = np.radians(heading_deg + self._orbit)
        back = np.array([-np.sin(th), 0.0, -np.cos(th)])       # opposite the facing
        right = np.array([np.cos(th), 0.0, -np.sin(th)])
        alt = max(0.0, (self.ground_y - float(target[1])) - self.rise_knee)
        base = self.distance if self._dist is None else self._dist
        dist = base + self.pull_back * alt
        want = target + back * dist + right * self.shoulder_offset
        # Below the knee this is exactly `target[1] - height`, the rigid boom.
        want[1] = float(target[1]) - self.height + (1.0 - self.rise_gain) * alt
        return want

    def update(self, target: np.ndarray, heading_deg: float, dt: float) -> Camera:
        target = np.asarray(target, float)
        # Locked off: once the first frame has fixed the pose, hand back exactly that pose.
        # Nothing about the subject can move it, which is what preserving a layout means.
        if self.mode == "lock" and self._locked is not None:
            c = self._locked
            return Camera(focal_px=c.focal_px, position=c.position.copy(),
                          pitch_deg=c.pitch_deg, yaw_deg=c.yaw_deg, roll_deg=c.roll_deg)
        if self._dist is None:
            self._dist = self.distance
        if self.mode == "orbit":
            self._orbit += self.orbit_rate * dt
        elif self.mode in ("push", "pull"):
            step = self.push_rate * dt
            self._dist += float(np.clip(self.push_to - self._dist, -step, step))
        want = self._desired(target, heading_deg)
        if self._pos is None:
            # Zero offset and zero relative velocity: frame 0 is already the steady state,
            # so there is nothing to spring in from.
            self._pos, self._vel = want.copy(), np.zeros(3)
            self._off = np.zeros(3)
            # Seed the aim at the direction that actually looks at the aim point, NOT at
            # the subject's heading. `shoulder_offset` puts the camera to one side of the
            # boom axis, so the two differ by atan(shoulder_offset / boom): 21.4 deg at a
            # 1.15 m boom, which is 57 px of a 336 px frame. `aim_tau` then eased that
            # error out over ~18 frames, sliding the subject sideways while nothing in the
            # world had moved. Shrinking aim_tau would only hide it faster, and would
            # stiffen tracking everywhere else.
            aim0 = target + np.array([0.0, -self.aim_height, 0.0]) - self._pos
            self._yaw = float(np.degrees(np.arctan2(aim0[0], aim0[2])))
        if self.mode == "hold":
            # Locked off: the rig keeps its pose and the subject moves through the frame.
            # The aim still tracks, which is what their "holds a steady vantage while she
            # rotates" shots do; a rig that also froze its yaw would lose the subject.
            aim0 = target + np.array([0.0, -self.aim_height, 0.0]) - self._pos
            yaw0 = np.degrees(np.arctan2(aim0[0], aim0[2]))
            d0 = (yaw0 - self._yaw + 180.0) % 360.0 - 180.0
            self._yaw += d0 * (1.0 - np.exp(-dt / max(self.aim_tau, 1e-6)))
            pitch0 = np.degrees(np.arctan2(-aim0[1], np.hypot(aim0[0], aim0[2])))
            return self._maybe_lock(Camera(
                focal_px=self.focal_px, position=self._pos.copy(),
                pitch_deg=float(np.clip(pitch0, self.pitch_min, self.pitch_max)),
                yaw_deg=float(self._yaw + self._yaw_offset())))
        # Second-order spring-damper, run in the frame that MOVES WITH the boom anchor.
        #
        # Damping the camera's ABSOLUTE velocity leaves a steady-state lag. For the camera
        # to travel at the subject's speed v the acceleration must vanish, which needs
        # w0^2 (want - pos) = 2 zeta w0 v, so the rig settles 2 zeta v / w0 metres behind
        # the anchor and stays there. At these constants that is 1.15 m behind a 1.9 m/s
        # walk, which doubles ex30's 1.15 m boom and quarters the subject's apparent area.
        # It also produced what looked like a startup transient: frame 0 is the only frame
        # framed correctly, because `_pos` is initialised at `want`, and every later frame
        # has sagged away from it.
        #
        # Measuring both the error and the velocity against the anchor makes constant-
        # velocity motion indistinguishable from standing still, which is what following
        # means. The spring still overshoots on genuine CHANGES of pose -- turns, a climb,
        # a push -- which is what zeta < 1 is for; it just no longer treats "the subject is
        # walking" as a change to be lagged. `_vel` is therefore a velocity RELATIVE to the
        # anchor, and zero is the correct value to initialise it to.
        # `_off` is CARRIED, not recomputed from `_pos - want`: recomputing it re-injects
        # the anchor's motion every frame and collapses back to the absolute-velocity form
        # algebraically, changing nothing. Integrating the offset itself is what makes
        # `_vel` a velocity relative to the anchor.
        w0 = 1.0 / max(self.boom_tau, 1e-6)
        acc = -w0 * w0 * self._off - 2.0 * self.zeta * w0 * self._vel
        self._vel = self._vel + acc * dt
        self._off = self._off + self._vel * dt
        self._pos = want + self._off

        # Aim height sets the pitch, and pitch decides whether any sky is in frame at all.
        # Aiming at 0.9 m from a 1.83 m camera only 1.9 m away pitches down ~26 degrees,
        # putting the whole frame below the horizon; released takes sit at -5 to -23.
        aim = target + np.array([0.0, -self.aim_height, 0.0]) - self._pos
        yaw = np.degrees(np.arctan2(aim[0], aim[2]))
        d = (yaw - self._yaw + 180.0) % 360.0 - 180.0
        self._yaw += d * (1.0 - np.exp(-dt / max(self.aim_tau, 1e-6)))
        # The aim decides the sign. Forcing `-abs(pitch)` here used to pin the axis
        # downward whenever the rest tilt was negative, so once the boom lagged a climb
        # and the subject rose ABOVE the camera, the rig kept staring at the ground: the
        # protagonist left the top of the frame at 425 and never came back (v = -1101 px
        # on a 192 px frame at 470). Elevation is what makes flight read, so the camera
        # has to be allowed to look up.
        pitch = np.degrees(np.arctan2(-aim[1], np.hypot(aim[0], aim[2])))

        return self._maybe_lock(Camera(
            focal_px=self.focal_px, position=self._pos.copy(),
            pitch_deg=float(np.clip(pitch, self.pitch_min, self.pitch_max)),
            yaw_deg=float(self._yaw + self._yaw_offset())))

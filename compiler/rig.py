"""The human proxy: one shared capsule rig, sized from the released examples.

Section 5.1 of out/SPEC.md measured the rig from depth spans, which are metric and
independent of focal length: head ellipsoid ~0.225 m across, standing height ~1.6 m,
torso ~0.30 m deep. All 15 float32 examples share it, so these are constants, not
per-character parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .render import Capsule, Sphere

HUMAN = 8

# Measured (SPEC 5.1). Radii are half the measured depth spans.
HEAD_D = 0.225
HEIGHT = 1.60
TORSO_R = 0.150
NECK_R = 0.058
# Upper-arm radius decides whether the arm falls inside the head band that the SPEC 5.1
# descriptor measures, and the response is a step, not a slope: measured through the real
# batch pipeline the descriptor reads 0.248 at both 0.076 and 0.055, then drops to 0.209
# at 0.040 as the arm clears the band. Only 0.040 lands inside the released band.
# human_frac is flat across all of these (0.023-0.024), so the thinner arm costs nothing
# elsewhere. Earlier sweeps that suggested otherwise measured a simplified scene, not this
# pipeline.
UPPER_ARM_R, FOREARM_R = 0.040, 0.045
THIGH_R, SHIN_R = 0.075, 0.060


def _env_override() -> None:
    """Allow a sweep to vary rig constants without editing this file.

    A tuning script that rewrote rig.py in a loop, run under a multiprocessing Pool that
    was missing its __main__ guard, spawned copies of itself until concurrent writes left
    the file empty. Sweeps set CWM_RIG_<NAME> instead; env vars also reach spawned
    workers for free, which file edits only appeared to do.
    """
    import os

    g = globals()
    for name in ("HEAD_D", "HEIGHT", "TORSO_R", "NECK_R", "UPPER_ARM_R", "FOREARM_R",
                 "THIGH_R", "SHIN_R", "SHOULDER_FRAC"):
        raw = os.environ.get(f"CWM_RIG_{name}")
        if raw is not None:
            g[name] = float(raw)


SHOULDER_FRAC = 0.80        # shoulder height as a fraction of standing height
_env_override()


def _rot(axis: str, deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@dataclass
class Pose:
    """Joint angles in degrees. Limb swing is about the body's left-right axis."""

    l_hip: float = 0.0
    r_hip: float = 0.0
    l_knee: float = 0.0
    r_knee: float = 0.0
    l_shoulder: float = 0.0
    r_shoulder: float = 0.0
    l_elbow: float = 0.0
    r_elbow: float = 0.0
    lean: float = 0.0


def walk_pose(phase: float, *, stride_deg: float = 32.0, run: bool = False) -> Pose:
    """A plain sinusoidal gait. The proxy only needs coarse, plausible articulation --
    the video model reconstructs real motion from the text prompt."""
    a = stride_deg * (1.6 if run else 1.0)
    s = np.sin(2 * np.pi * phase)
    c = np.cos(2 * np.pi * phase)
    return Pose(
        l_hip=a * s, r_hip=-a * s,
        l_knee=max(0.0, -35.0 * c) + (25.0 if run else 0.0),
        r_knee=max(0.0, 35.0 * c) + (25.0 if run else 0.0),
        l_shoulder=-0.7 * a * s, r_shoulder=0.7 * a * s,
        l_elbow=20.0 + (35.0 if run else 0.0), r_elbow=20.0 + (35.0 if run else 0.0),
        lean=8.0 if run else 2.0,
    )


# Where the rig's SILHOUETTE centroid sits, as a fraction of standing height. Measured by
# rendering the standing rig and taking the mean row of its class-8 mask, then solving for
# the aim height at which a screen-placement request lands on it: 0.96 m on a 1.75 m figure.
# It is above the geometric half because the torso, head and arms carry more silhouette
# area than the legs. `aim_screen_y` resolves against this, so that a request positions the
# FIGURE rather than whatever point the lens happens to be aimed at.
SILHOUETTE_CENTRE_FRAC = 0.549


# Idle amplitudes. Two sources, because the clause fixes the KIND of motion and their own
# conditions fix its SCALE.
#
# The kind comes from ex36's timeline: the subject "remains planted ... without stepping,
# feet fixed" while she "subtly turns her head and upper chest toward her right" and "her
# right hand swings a short distance beside her thigh". So it is a torso turn, an arm
# swing and a breathing sway, with the hips and knees pinned at zero -- that is what "feet
# fixed" means, and it is why this is a separate channel from the gait.
#
# The scale comes from the same clause, read at the landmark the words name. "A short
# distance beside her thigh" is a statement about the HAND, so it is checked at the
# forearm end against a full walking swing at that same point:
#
#     walking hand swing (walk_pose)   40.4 cm peak-to-peak
#     idle hand swing at  8 deg        16.8 cm  =  42% of it
#     idle hand swing at 18 deg        34.1 cm  =  84% of it
#
# 8 degrees reads as a short swing beside the thigh; 18 is four fifths of a full walking
# arm swing on a subject the prompt describes as planted, and does not.
#
# These were briefly set to 14/18, scaled to match the 47 px/frame of class-8 change their
# own planted subject shows. That was the one constant in the idle path fitted to a metric
# rather than derived, and it does not survive scrutiny: their 47 px is not a physical
# quantity -- unlike the stand-off, which is a distance recoverable from the depth channel,
# it composes amplitude, subject size, framing, resolution and the 2x2 pool. The check that
# was supposed to justify the fit was also computed off the wrong bone, quoting 10 cm for
# the hand when 10 cm is the ELBOW (hand 34.1 cm, elbow 20.4 cm at 18 degrees).
#
# So both constants come from the clause and ex36's shortfall is recorded rather than
# closed: it lands at 26 px/frame against their 47. That gap is real and is not something
# to spend amplitude on.
#
# Both constants matter. The torso turn dominates the rendered mask change -- 286 px/frame
# at full amplitude, 122 with the turn zeroed, 217 with the swing zeroed -- so a sweep that
# varies only the arm reports the same pixel count at every angle and concludes the
# amplitude has no effect. t_idle_amplitude_follows_the_clause asserts both.
#
IDLE_TURN_DEG = 6.0
IDLE_SWING_DEG = 8.0
IDLE_SWAY_DEG = 1.5


def idle_pose(phase: float) -> Pose:
    """Articulation for a subject that is standing still rather than walking.

    Gait is driven by distance covered, so a planted subject had no phase and no motion,
    and rendered as a single frame repeated for the whole window. This is the channel that
    runs instead -- upper body only, legs untouched.
    """
    s = np.sin(2 * np.pi * phase)
    c = np.cos(2 * np.pi * phase)
    return Pose(l_shoulder=-IDLE_SWING_DEG * s, r_shoulder=IDLE_SWING_DEG * s,
                l_elbow=12.0, r_elbow=12.0, lean=2.0 + IDLE_SWAY_DEG * c)


def idle_turn(phase: float) -> float:
    """Degrees the upper body turns off its heading. Paired with `idle_pose`."""
    return IDLE_TURN_DEG * float(np.sin(2 * np.pi * phase))


@dataclass
class Human:
    """A rig instance: ground position, facing, and a pose."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))   # feet on the ground
    facing_deg: float = 0.0
    pose: Pose = field(default_factory=Pose)
    scale: float = 1.0

    def primitives(self) -> list:
        h = HEIGHT * self.scale
        # Landmark heights as fractions of standing height. The shoulder sits at 0.80:
        # the head-band descriptor of SPEC 5.1 jumps between ~0.19 (arms clear of the
        # band) and ~0.24 (arms inside it), and released sits at 0.214, so the released
        # rig has the arms partly in the band.
        hip_y, shoulder_y, head_y = 0.53 * h, SHOULDER_FRAC * h, 0.93 * h
        hip_dx, sh_dx = 0.085 * h, 0.115 * h
        thigh, shin = 0.26 * h, 0.26 * h
        uarm, farm = 0.175 * h, 0.16 * h

        R = _rot("y", self.facing_deg) @ _rot("x", -self.pose.lean)
        base = np.asarray(self.position, float)

        def place(p):
            # Body frame is +Y up; world frame is +Y down.
            q = np.asarray(p, float) * np.array([1.0, -1.0, 1.0])
            return base + R @ q

        def limb(root, ang1, ang2, len1, len2, r1, r2):
            d1 = _rot("x", ang1) @ np.array([0.0, -1.0, 0.0])
            mid = root + d1 * len1
            d2 = _rot("x", ang1 + ang2) @ np.array([0.0, -1.0, 0.0])
            end = mid + d2 * len2
            return [Capsule(place(root), place(mid), r1 * self.scale, HUMAN),
                    Capsule(place(mid), place(end), r2 * self.scale, HUMAN)]

        p = self.pose
        prims = [
            Sphere(place([0, head_y, 0]), HEAD_D / 2 * self.scale, HUMAN,
                   scale=np.array([1.0, 1.25, 1.0])),
            # Neck. Without it the head is a detached sphere: depth-aware blob extraction
            # splits it off at close range, and the descriptor that is supposed to measure
            # head diameter ends up measuring the torso blob's shoulders instead.
            Capsule(place([0, shoulder_y, 0]), place([0, head_y, 0]), NECK_R * self.scale, HUMAN),
            Capsule(place([0, hip_y, 0]), place([0, shoulder_y, 0]), TORSO_R * self.scale, HUMAN),
        ]
        prims += limb(np.array([-hip_dx, hip_y, 0]), p.l_hip, p.l_knee, thigh, shin, THIGH_R, SHIN_R)
        prims += limb(np.array([hip_dx, hip_y, 0]), p.r_hip, p.r_knee, thigh, shin, THIGH_R, SHIN_R)
        prims += limb(np.array([-sh_dx, shoulder_y, 0]), p.l_shoulder, p.l_elbow, uarm, farm,
                      UPPER_ARM_R, FOREARM_R)
        prims += limb(np.array([sh_dx, shoulder_y, 0]), p.r_shoulder, p.r_elbow, uarm, farm,
                      UPPER_ARM_R, FOREARM_R)
        return prims

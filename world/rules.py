"""Reusable mechanisms the coding agent composes, revises, and invokes.

Rules run every tick for every entity, independent of what the camera sees; that is what
gives off-screen entities a continuous history. The agent intervenes sparsely -- to change
a goal, add a mechanism, or handle an exception -- while these advance the dense state.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .state import Entity, WorldState

Rule = Callable[[WorldState], None]

# The rules defined in this module, by the decorator, at import. Written once and never
# again: a take's own rules go in its RuleSet, not here.
#
# This used to be one mutable module-level dict that `agent.AddRule` wrote into, so it
# outlived the take that created it. A second simulate() in the same process started from
# `sorted(rules.names())` -- by then 14 entries rather than 4 -- and activated every rule
# the FIRST take had added, from frame 0, before the entities they reference existed. The
# flight rule fired immediately and the protagonist spent the whole take 14 m up.
_BUILTIN: dict[str, Rule] = {}
# Rules every take runs unless it says otherwise. The physics rules added later (blocking,
# falling, impact) are registered but NOT default: `names()` used to return every
# registered rule, and repro.py / test_actions.py activate `names()` wholesale, so
# registering `impact` as a builtin had the taxi in `board_and_drive` launch the courier it
# was meant to pick up, and 5 of 29 scenario tests failed. A take opts in by naming them.
_DEFAULT: list[str] = []


def rule(name: str, default: bool = True):
    def deco(fn: Rule) -> Rule:
        _BUILTIN[name] = fn
        if default:
            _DEFAULT.append(name)
        return fn
    return deco


class RuleSet:
    """The rules available to ONE take: the builtins, plus whatever its agent adds.

    Takes get one of these instead of sharing module state, so nothing a take does can
    reach the next one.
    """

    def __init__(self, base: dict[str, Rule] | None = None) -> None:
        # The defaults only: a take that activates `self.names()` wholesale must not pick
        # up the opt-in physics rules by being constructed after they were registered.
        self._r = dict({k: _BUILTIN[k] for k in _DEFAULT} if base is None else base)

    def add(self, name: str, fn: Rule) -> None:
        self._r[name] = fn

    def get(self, name: str) -> Rule:
        # Opt-in rules are reachable by name without being listed.
        return self._r[name] if name in self._r else _BUILTIN[name]

    def names(self) -> list[str]:
        return sorted(self._r)

    def step(self, s: WorldState, active: list[str]) -> None:
        """One tick: run the named rules in order, then advance time."""
        for name in active:
            self.get(name)(s)
        s.t = round(s.t + s.dt, 6)


def get(name: str) -> Rule:
    return _BUILTIN[name]


def names() -> list[str]:
    """The default rules. A take should ask its own RuleSet, not this. The opt-in physics
    rules are reachable by name through `get()` and `step()` but are not listed here."""
    return sorted(_DEFAULT)


def physics() -> list[str]:
    """The opt-in rules, in the order a take should run them: impact marks a launch before
    falling integrates it, blocking vetoes the step before locomotion takes it."""
    return ["impact", "falling", "blocking"]


@rule("locomotion")
def locomotion(s: WorldState) -> None:
    """Advance every moving entity along its heading. Runs whether or not it is on screen."""
    for e in s.entities.values():
        if e.alive and e.speed:
            e.pos = e.pos + e.forward() * e.speed * s.dt


@rule("waypoints")
def waypoints(s: WorldState) -> None:
    """Steer toward the current waypoint, turning at a bounded rate, and cycle on arrival."""
    turn_rate = s.params.get("turn_rate_deg_s", 90.0)
    for e in s.entities.values():
        wps = e.attrs.get("waypoints")
        if not e.alive or not wps:
            continue
        i = int(e.attrs.get("wp_index", 0)) % len(wps)
        target = np.asarray(wps[i], float)
        to = target - e.pos
        to[1] = 0.0
        if np.linalg.norm(to) < e.attrs.get("wp_radius", 1.5):
            e.attrs["wp_index"] = (i + 1) % len(wps)
            s.log("waypoint_reached", e.eid, index=i)
            continue
        want = float(np.degrees(np.arctan2(to[0], to[2])))
        d = (want - e.heading_deg + 180.0) % 360.0 - 180.0
        e.heading_deg += float(np.clip(d, -turn_rate * s.dt, turn_rate * s.dt))


def _solid_footprints(s: WorldState, mover) -> list:
    """Oriented XZ boxes of everything a mover cannot pass through: buildings (scenery)
    and other vehicles. Vegetation is not here -- a crown is walkable-under in this proxy
    and a trunk is not modelled. Water is a surface, not a wall."""
    out = []
    for o in s.entities.values():
        if o is mover or not o.alive or o.kind not in ("scenery", "vehicle"):
            continue
        th = np.radians(o.heading_deg)
        out.append((o, o.pos, np.array([np.cos(th), 0.0, -np.sin(th)]),
                    np.array([np.sin(th), 0.0, np.cos(th)]),
                    float(o.size[0]) / 2.0, float(o.size[2]) / 2.0))
    return out


@rule("blocking", default=False)
def blocking(s: WorldState) -> None:
    """Stop a mover whose next step would put its leading edge inside something solid.

    Runs BEFORE `locomotion` (rule order is the active list's order), so the step that
    would penetrate is never taken: speed is zeroed for this tick and the entity stays put.
    Scripted rules that set speed every tick (DRIVE_OFF) re-arm it, and this rule cancels
    it again next tick, so a blocked car stays blocked without teleporting back. Until this
    existed nothing in the world knew a wall was solid -- buildings and cars were render-
    only boxes and a walker went straight through them.
    """
    for e in s.entities.values():
        if not e.alive or not e.speed or e.kind not in ("human", "vehicle", "animal"):
            continue
        fwd = e.forward()
        th = np.radians(e.heading_deg)
        right = np.array([np.cos(th), 0.0, -np.sin(th)])
        # The two leading corners after the step: half the body length ahead of the centre
        # and half the width to each side. Testing corners rather than a disc around the
        # nose keeps a head-on stop at one step's length from the face (0.06 m walking,
        # 0.33 m at 8 m/s) instead of a body half-width short of it.
        lead = float(e.size[2]) / 2.0
        half_w = float(e.size[0]) / 2.0
        # Stand-off of one half-width: stopping with the nose 5 cm from a face left no room
        # for a body of that width to walk along the face (its near corner was inside the
        # wall in every lateral direction), so a pedestrian who sidestepped out of a dead
        # end was blocked again on the spot. Arm's length for a walker, ~1 m for a car.
        nose = e.pos + fwd * (e.speed * s.dt + lead + half_w)
        corners = (nose + right * half_w, nose - right * half_w)
        for o, c, ax, az, hx, hz in _solid_footprints(s, e):
            if any(abs(float((q - c) @ ax)) < hx and abs(float((q - c) @ az)) < hz
                   for q in corners):
                if not e.attrs.get("blocked_by"):
                    s.log("blocked", e.eid, by=o.eid, speed=round(float(e.speed), 2))
                e.attrs["blocked_by"] = o.eid
                e.speed = 0.0
                break
        else:
            e.attrs["blocked_by"] = ""


FALL_S = 0.5          # seconds from the hit to lying flat


G = 9.81
IMPACT_MIN_SPEED = 2.0    # m/s; slower than this a car just stops against a person (blocking)
TUMBLE_DEG_S = 540.0      # how fast a launched body spins about the vertical


@rule("falling", default=False)
def falling(s: WorldState) -> None:
    """Anyone with `attrs["hit"]` goes down: `down_frac` runs 0 -> 1 over FALL_S with a
    quadratic ease (a fall accelerates), speed is zero from the first frame, and once flat
    `attrs["down"]` is set and stays. The rig reads `down_frac` as a backward lean about the
    feet. One hit is enough; there is no punch geometry and no getting up.

    A body with `attrs["fly"]` (set by `impact`) is also ballistic: the velocity is
    integrated here, +Y down so altitude is -pos[1], it tumbles about the vertical while
    airborne, and lands where gravity puts it, flat."""
    for e in s.entities.values():
        if not e.alive or e.kind not in ("human", "animal") or not e.attrs.get("hit"):
            continue
        t = float(e.attrs.get("down_t", 0.0)) + s.dt
        e.attrs["down_t"] = t
        e.attrs["down_frac"] = min(1.0, (t / FALL_S) ** 2)
        e.speed = 0.0
        fly = e.attrs.get("fly")
        if fly:
            vx, vup, vz = (float(x) for x in fly)
            p = e.pos
            alt = -float(p[1]) + vup * s.dt
            vup -= G * s.dt
            p[0] += vx * s.dt
            p[2] += vz * s.dt
            e.heading_deg += TUMBLE_DEG_S * s.dt
            if alt <= 0.0 and vup < 0.0:
                p[1] = 0.0
                e.attrs["fly"] = None
                e.attrs["down_frac"] = 1.0
                s.log("landed", e.eid, x=round(float(p[0]), 2), z=round(float(p[2]), 2))
            else:
                p[1] = -alt
                e.attrs["fly"] = [vx, vup, vz]
            e.pos = p
        if e.attrs["down_frac"] >= 1.0 and not e.attrs.get("down") and not e.attrs.get("fly"):
            e.attrs["down"] = 1.0
            s.log("down", e.eid)


@rule("impact", default=False)
def impact(s: WorldState) -> None:
    """A vehicle moving faster than IMPACT_MIN_SPEED that reaches a standing person launches
    them: forward at most of the car's speed, up by a speed-dependent amount, and sideways
    away from the car's centre line, so a glancing hit throws to the side. The person is
    `hit` from that frame (see `falling`). Nothing happens to the car. Runs before
    `locomotion`, on where the car will be after this step."""
    for v in s.entities.values():
        if not v.alive or v.kind != "vehicle" or abs(v.speed) < IMPACT_MIN_SPEED:
            continue
        th = np.radians(v.heading_deg)
        fwd = np.array([np.sin(th), 0.0, np.cos(th)]) * (1.0 if v.speed >= 0 else -1.0)
        right = np.array([np.cos(th), 0.0, -np.sin(th)])
        hx, hz = float(v.size[0]) / 2.0, float(v.size[2]) / 2.0
        c = v.pos + fwd * abs(v.speed) * s.dt
        for e in s.entities.values():
            if not e.alive or e.kind not in ("human", "animal") or e.attrs.get("hit"):
                continue
            if e.attrs.get("riding") or float(e.pos[1]) < -0.5:
                continue
            d = e.pos - c
            lat, lon = float(d @ right), float(d @ fwd)
            if abs(lat) < hx + float(e.size[0]) / 2.0 and abs(lon) < hz + 0.3:
                side = 1.0 if lat >= 0.0 else -1.0
                sp = abs(v.speed)
                vel = fwd * sp * 0.85 + right * side * (1.0 + 0.15 * sp)
                e.attrs["hit"] = 1.0
                e.attrs["fly"] = [float(vel[0]), 2.0 + 0.3 * sp, float(vel[2])]
                e.speed = 0.0
                s.log("run_over", e.eid, by=v.eid, speed=round(sp, 2))


@rule("separation")
def separation(s: WorldState) -> None:
    """Keep movers apart. Cheap, dense, repetitive -- exactly what code should own."""
    movers = [e for e in s.entities.values() if e.alive and e.speed]
    radius = s.params.get("separation_m", 1.6)
    for i, a in enumerate(movers):
        for b in movers[i + 1:]:
            d = a.pos - b.pos
            d[1] = 0.0
            dist = float(np.linalg.norm(d))
            if 1e-6 < dist < radius:
                push = d / dist * (radius - dist) * 0.5
                a.pos = a.pos + push
                b.pos = b.pos - push
                if not a.attrs.get("_touching") and dist < radius * 0.6:
                    s.log("near_miss", a.eid, other=b.eid, distance=round(dist, 2))
                a.attrs["_touching"] = True
            else:
                a.attrs["_touching"] = False


@rule("fatigue")
def fatigue(s: WorldState) -> None:
    """A consequence that accumulates off-screen: running tires an actor, and it persists."""
    for e in s.of_kind("human"):
        run_speed = s.params.get("run_threshold", 2.0)
        f = float(e.attrs.get("fatigue", 0.0))
        f += (0.06 if e.speed > run_speed else -0.03) * s.dt
        e.attrs["fatigue"] = float(np.clip(f, 0.0, 1.0))
        cap = s.params.get("base_speed", 2.4) * (1.0 - 0.45 * e.attrs["fatigue"])
        if e.speed > cap:
            e.speed = cap
            if not e.attrs.get("_slowed"):
                s.log("slowed_by_fatigue", e.eid, fatigue=round(e.attrs["fatigue"], 3))
                e.attrs["_slowed"] = True
        elif e.attrs["fatigue"] < 0.2:
            e.attrs["_slowed"] = False


def step(s: WorldState, active: list[str]) -> None:
    """One tick over the BUILTIN rules, for scenes that never add any."""
    for name in active:
        _BUILTIN[name](s)
    s.t = round(s.t + s.dt, 6)


def run(s: WorldState, seconds: float, active: list[str]) -> None:
    for _ in range(int(round(seconds / s.dt))):
        step(s, active)

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


def rule(name: str):
    def deco(fn: Rule) -> Rule:
        _BUILTIN[name] = fn
        return fn
    return deco


class RuleSet:
    """The rules available to ONE take: the builtins, plus whatever its agent adds.

    Takes get one of these instead of sharing module state, so nothing a take does can
    reach the next one.
    """

    def __init__(self, base: dict[str, Rule] | None = None) -> None:
        self._r = dict(_BUILTIN if base is None else base)

    def add(self, name: str, fn: Rule) -> None:
        self._r[name] = fn

    def get(self, name: str) -> Rule:
        return self._r[name]

    def names(self) -> list[str]:
        return sorted(self._r)

    def step(self, s: WorldState, active: list[str]) -> None:
        """One tick: run the named rules in order, then advance time."""
        for name in active:
            self._r[name](s)
        s.t = round(s.t + s.dt, 6)


def get(name: str) -> Rule:
    return _BUILTIN[name]


def names() -> list[str]:
    """The builtin rules. A take should ask its own RuleSet, not this."""
    return sorted(_BUILTIN)


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

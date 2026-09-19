"""Per-frame invariants for a take, checked while it is generated.

Every defect found in the action demos so far was caught by eye, after rendering, from a
contact sheet: a broom above its rider instead of below, an aircraft at the right elevation
but behind the camera, a protagonist walking straight past the car he had just called. Each
was a violated invariant that nothing was checking.

These run inside the simulation loop, cost almost nothing, and fail loudly with the frame
number and the numbers involved.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from compiler.render import screen_bbox


@dataclass
class Mount:
    """A rider-to-mount relation: which body, how high the seat sits on it, and from which
    frame the rider is expected to be in it."""

    eid: str
    seat_frac: float = 0.78
    seated_from: int = 0


@dataclass
class Violation:
    frame: int
    rule: str
    detail: str


@dataclass
class Invariants:
    """Accumulates violations; `report()` summarises, `assert_clean()` raises."""

    violations: list[Violation] = field(default_factory=list)
    _prev: dict = field(default_factory=dict)
    # eid -> (first_frame, last_frame) over which the entity must be on screen
    visible_windows: dict = field(default_factory=dict)
    _seen_visible: dict = field(default_factory=dict)
    _active: dict = field(default_factory=dict)
    # eid -> its own on-screen threshold, overriding finish()'s default. The protagonist
    # needs a far tighter bound than a prop: the courier was off the top of the frame for
    # 55% of the flight and still cleared a 60% bar, because losing him for two seconds is
    # a small fraction of a 20 s take and a total failure of the shot.
    visible_min: dict = field(default_factory=dict)

    def fail(self, frame: int, rule: str, detail: str) -> None:
        self.violations.append(Violation(frame, rule, detail))

    # -- per-frame checks ---------------------------------------------------
    def check(self, s, cam, frame: int, *, mounts: dict | None = None,
              flying: set[str] | None = None) -> None:
        """Check one frame.

        `mounts` maps rider eid -> `Mount(eid, seat_frac, seated_from)`. The relation exempts
        the rider from `grounded` for the whole take -- climbing legitimately lifts it off the
        road -- while the geometric checks only start at `seated_from`, since a climb that
        takes a second cannot satisfy them on frame one. `seat_frac` is per mount: a camel's
        back and a car's seat are not at the same fraction of their height, and assuming one
        value for both reported a correctly seated driver as 0.5 m out of place.
        """
        mounts = {k: (v if isinstance(v, Mount) else Mount(v)) for k, v in (mounts or {}).items()}
        flying = flying or set()

        for eid, e in s.entities.items():
            if not e.alive:
                continue
            p = e.pos

            if not np.all(np.isfinite(p)):
                self.fail(frame, "finite", f"{eid} position {p}")
                continue

            # A body on the ground has y == 0. Anything else is either a rider, a flyer, or
            # a bug -- and it is the bug case this exists to catch.
            if e.kind in ("human", "animal", "vehicle") and eid not in flying and eid not in mounts:
                if abs(float(p[1])) > 1e-6:
                    self.fail(frame, "grounded", f"{eid} y={float(p[1]):+.3f} but is neither riding nor flying")

            # No teleports: a step may not exceed the speed it claims, with slack for rules
            # that drive position directly.
            prev = self._prev.get(eid)
            if prev is not None:
                step = float(np.linalg.norm(p - prev))
                # Free fall reaches ~sqrt(2*g*h) regardless of `speed`, so the budget has to
                # cover gravity for anything off the ground, not just its declared speed.
                fall = (2.0 * 9.81 * max(-float(prev[1]), -float(p[1]), 0.0)) ** 0.5
                budget = max(float(e.speed), fall, 1.0) * s.dt * 3.0 + 0.5
                if step > budget:
                    self.fail(frame, "continuity",
                              f"{eid} moved {step:.2f} m in one frame (speed {e.speed:.1f}, budget {budget:.2f})")
            self._prev[eid] = p.copy()

        # A rider must actually be on its mount, not beside it or inside it -- but only once
        # the climb has had time to finish.
        for rider, mt in mounts.items():
            if frame < mt.seated_from:
                continue
            r, m = s.entities.get(rider), s.entities.get(mt.eid)
            if r is None or m is None or not r.alive or not m.alive:
                continue
            lateral = float(np.linalg.norm((r.pos - m.pos)[[0, 2]]))
            if lateral > 0.45:
                self.fail(frame, "mount_xz", f"{rider} is {lateral:.2f} m from {mt.eid} in XZ")
            seat = float(m.size[1]) * mt.seat_frac
            got = float(-r.pos[1])
            if abs(got - seat) > 0.45:
                self.fail(frame, "mount_height",
                          f"{rider} sits at {got:.2f} m, seat of {mt.eid} is {seat:.2f} m "
                          f"(frac {mt.seat_frac})")

        # Anything the script says must be seen has to be inside the frustum, not merely at a
        # plausible distance. Elevation alone is not visibility: an aircraft at the right
        # height but behind the camera passes every distance check and appears in nothing.
        for eid, (f0, f1) in self.visible_windows.items():
            if not (f0 <= frame <= f1):
                continue
            e = s.entities.get(eid)
            self._active[eid] = self._active.get(eid, 0) + 1
            if e is None or not e.alive:
                continue
            bb = screen_bbox(cam, e.pos + np.array([0.0, -float(e.size[1]) / 2.0, 0.0]),
                             float(max(e.size)) / 2.0, 336, 192)
            # screen_bbox has THREE outcomes and only one of them means "not in shot":
            #   ()    dead or wholly offscreen        -> not visible
            #   None  sphere straddles the eye plane  -> the lens is inside the body, so
            #         it is maximally visible and cannot be bounded
            #   tuple its pixel extent                -> visible
            # `if bb:` scored None as invisible, i.e. the one case meaning "this fills the
            # frame" counted the same as "this is not there". On the shipped take the taxi
            # at f132 covers 11.4% of the frame and was reported not visible; taxi read 73%
            # and gunship 79% where both are really 100%. It passed only because 73 and 79
            # clear the 60% default, so the margin was quietly being spent.
            if bb is None or bb:
                self._seen_visible[eid] = self._seen_visible.get(eid, 0) + 1

    # -- reporting ----------------------------------------------------------
    def finish(self, min_visible_frac: float = 0.6) -> None:
        for eid, (f0, f1) in self.visible_windows.items():
            n = self._active.get(eid, 0)
            seen = self._seen_visible.get(eid, 0)
            frac = seen / max(n, 1)
            want = self.visible_min.get(eid, min_visible_frac)
            if frac < want:
                self.fail(f1, "visible",
                          f"{eid} on screen in {seen}/{n} frames ({frac:.0%}) of its window "
                          f"[{f0},{f1}]; wanted >= {want:.0%}")

    def report(self) -> str:
        if not self.violations:
            return "invariants: clean"
        by_rule: dict[str, list[Violation]] = {}
        for v in self.violations:
            by_rule.setdefault(v.rule, []).append(v)
        lines = [f"invariants: {len(self.violations)} violation(s) across {len(by_rule)} rule(s)"]
        for rule, vs in sorted(by_rule.items()):
            lines.append(f"  {rule}: {len(vs)}x, first at frame {vs[0].frame} -- {vs[0].detail}")
            if len(vs) > 1:
                lines.append(f"      last at frame {vs[-1].frame} -- {vs[-1].detail}")
        return "\n".join(lines)

    def assert_clean(self) -> None:
        if self.violations:
            raise AssertionError(self.report())

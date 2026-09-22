"""A library of agent-authored actions, each with the invariants it must satisfy.

These are the mechanisms the coding agent composes. They are text until the engine compiles
them, which is the point -- but unlike one-off rules written per demo, each is parameterised,
reused, and covered by `test_actions.py`.

Conventions that caused real bugs and are therefore stated once here:

  * **+Y is down.** An altitude of `a` metres is `pos[1] = -a`. Subtracting from y raises a
    body; the broom sat above its rider for exactly this reason.
  * **`pos` is a property returning a fresh array.** `e.pos[1] = x` writes to a temporary and
    is lost. Read it, mutate, assign back.
  * **Heading 180 means forward is -Z.** Anything meant to be seen must be placed ahead of
    where the subject is *going*, not at +Z by default. Correct elevation does not help a body
    that is behind the camera.
"""
from __future__ import annotations

APPROACH_AND_STOP = """
def {name}(s):
    e = s.entities.get("{eid}")
    if e is None or not e.alive:
        return
    along = float(e.pos[2]) * {sign}
    target = {stop_at} * {sign}
    if along > target:
        e.speed = max({v_min}, min({v_max}, (along - target) * {gain}))
    else:
        e.speed = 0.0
        e.attrs["arrived"] = 1.0
"""

HALT_SUBJECT = """
def {name}(s):
    e = s.entities.get("{eid}")
    h = s.get("{subject}")
    if e is None or not e.alive:
        return
    # The subject waits for what it called instead of walking past it.
    if abs(float(e.pos[2]) - float(h.pos[2])) < {radius}:
        h.speed = 0.0
        h.attrs["halted"] = 1.0
"""

MOUNT = """
def {name}(s):
    m = s.entities.get("{mount}")
    h = s.get("{rider}")
    if m is None or not m.alive:
        return
    # Do not begin until the mount is actually alongside. Starting the climb while it is still
    # 20 m away makes the rider cross that gap at the climb rate -- a visible teleport, and a
    # continuity violation every frame of it.
    gap = ((float(h.pos[0]) - float(m.pos[0])) ** 2
           + (float(h.pos[2]) - float(m.pos[2])) ** 2) ** 0.5
    if not m.attrs.get("{name}_started") and gap > {reach}:
        return
    m.attrs["{name}_started"] = 1.0
    t = float(m.attrs.get("{name}_t", 0.0)) + s.dt
    m.attrs["{name}_t"] = t
    lift = min(t / {climb_s}, 1.0)
    seat = float(m.size[1]) * {seat_frac} + float(m.attrs.get("bob", 0.0))
    p, mp = h.pos, m.pos
    y0 = float(m.attrs.get("{name}_y0", p[1]))
    m.attrs["{name}_y0"] = y0
    h.pos = [p[0] + (mp[0] - p[0]) * lift,
             y0 + (-seat - y0) * lift,
             p[2] + (mp[2] - p[2]) * lift]
    h.heading_deg = m.heading_deg
    h.speed = 0.0
    if lift >= 1.0:
        m.attrs["mounted"] = 1.0
        m.speed = {ride_speed}
"""

RIDE = """
def {name}(s):
    m = s.entities.get("{mount}")
    h = s.get("{rider}")
    if m is None or not m.alive or not m.attrs.get("mounted"):
        return
    t = float(m.attrs.get("gait_t", 0.0)) + s.dt
    m.attrs["gait_t"] = t
    m.attrs["bob"] = {bob} * float(np.sin(t * {bob_hz}))
    seat = float(m.size[1]) * {seat_frac} + float(m.attrs["bob"])
    mp = m.pos
    h.pos = [mp[0], -seat, mp[2]]
    h.heading_deg = m.heading_deg
    h.speed = 0.0
"""

BOARD_VEHICLE = """
def {name}(s):
    v = s.entities.get("{vehicle}")
    h = s.get("{rider}")
    if v is None or not v.alive or not v.attrs.get("arrived"):
        return
    t = float(v.attrs.get("{name}_t", 0.0)) + s.dt
    v.attrs["{name}_t"] = t
    k = min(t / {board_s}, 1.0)
    seat = float(v.size[1]) * {seat_frac}
    p, vp = h.pos, v.pos
    y0 = float(v.attrs.get("{name}_y0", p[1]))
    v.attrs["{name}_y0"] = y0
    h.pos = [p[0] + (vp[0] - p[0]) * k, y0 + (-seat - y0) * k, p[2] + (vp[2] - p[2]) * k]
    h.heading_deg = v.heading_deg
    h.speed = 0.0
    if k >= 1.0:
        v.attrs["boarded"] = 1.0
"""

DRIVE_OFF = """
def {name}(s):
    v = s.entities.get("{vehicle}")
    h = s.get("{rider}")
    if v is None or not v.alive or not v.attrs.get("boarded"):
        return
    t = float(v.attrs.get("{name}_t", 0.0)) + s.dt
    v.attrs["{name}_t"] = t
    v.speed = min({v_max}, {accel} * t)
    seat = float(v.size[1]) * {seat_frac}
    vp = v.pos
    h.pos = [vp[0], -seat, vp[2]]
    h.heading_deg = v.heading_deg
    h.speed = 0.0
"""

FLY_ON_PROP = """
def {name}(s):
    h = s.get("{rider}")
    pr = s.entities.get("{prop}")
    t = float(h.attrs.get("{name}_t", 0.0)) + s.dt
    h.attrs["{name}_t"] = t
    k = min(t / {climb_s}, 1.0)
    p = h.pos
    # Start from whatever height the rider is already at -- stepping off a camel is not a
    # teleport to the road and back.
    y0 = float(h.attrs.get("{name}_y0", p[1]))
    h.attrs["{name}_y0"] = y0
    alt = -y0 + ({alt_max} + y0) * k * k
    h.pos = [p[0], -alt, p[2] + {drift} * s.dt]
    h.speed = 0.0
    if pr is not None and pr.alive:
        hp = h.pos
        # +Y is down: ADD to place the prop below its rider.
        pr.pos = [hp[0], hp[1] + {offset}, hp[2]]
        pr.heading_deg = h.heading_deg
"""

DESCEND_PAST = """
def {name}(s):
    e = s.entities.get("{eid}")
    if e is None or not e.alive:
        return
    t = float(e.attrs.get("{name}_t", 0.0)) + s.dt
    e.attrs["{name}_t"] = t
    k = min(t / {dur_s}, 1.0)
    alt = {alt0} - ({alt0} - {alt1}) * (1.0 - (1.0 - k) * (1.0 - k))
    p = e.pos
    p[1] = -alt
    e.pos = p
"""

BALLISTIC = """
def {name}(s):
    for eid in {eids}:
        b = s.entities.get(eid)
        if b is None or not b.alive:
            continue
        t = float(b.attrs.get("t", 0.0)) + s.dt
        b.attrs["t"] = t
        if t < 0.0:
            continue
        p = b.pos
        p[1] = min(0.0, -float(b.attrs.get("alt0", 16.0)) + 0.5 * 9.81 * t * t)
        b.pos = p
"""


KNOCKDOWN = """
def {name}(s):
    a = s.get("{attacker}")
    t = s.entities.get("{target}")
    if t is None or not t.alive:
        return
    if not t.attrs.get("hit"):
        gap = ((float(a.pos[0]) - float(t.pos[0])) ** 2
               + (float(a.pos[2]) - float(t.pos[2])) ** 2) ** 0.5
        if gap > {reach}:
            return
        # One hit. No punch is rendered: the proxy shows the consequence, the caption
        # names the act, and the video model draws the blow.
        t.attrs["hit"] = 1.0
        a.attrs["{name}_pause"] = {pause_s}
        s.log("hit", t.eid, by=a.eid, distance=round(gap, 2))
    # The attacker stands over the fall, then carries on.
    left = float(a.attrs.get("{name}_pause", 0.0))
    if left > 0.0:
        a.attrs["{name}_pause"] = left - s.dt
        a.speed = 0.0
"""


def build(template: str, name: str, **kw) -> str:
    """Instantiate one action's source. Missing keys fail here, not at exec time."""
    return template.format(name=name, **kw)

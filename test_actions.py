"""Exercise each action in isolation and assert its invariants.

Every action defect so far was found by eye, after rendering: a broom above its rider, an
aircraft behind the camera, a protagonist walking past the car he called. Each is a violated
invariant. This runs them without rendering -- seconds, not minutes -- so a broken action
fails before it reaches a GPU.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from world import actions as A
from world import agent as ag
from world import compile as wc
from world import rules
from take_actions import PHASES as TAKE_PHASES
from take_actions import WINDOWS as TAKE_WINDOWS
from take_actions import simulate as simulate_take
from world.invariants import Invariants, Mount
from world.scenes import street_take
from world.state import Entity, WorldState

FPS = 24


def run(name: str, frames: int, spawns, sources, *, mounts=None, flying=None,
        visible=None, quiet=False, mount_from=0):
    """Simulate `frames` frames with the given spawns/rules, checking invariants each step.

    `mount_from` is the frame by which the rider must be seated: a climb takes time, so
    asserting the mount relation from frame 0 fails an action for doing its job.

    Each call gets its own RuleSet, so a rule added by one test cannot stay active in the
    next -- which is how a test containing no flight once flew the protagonist to 13 m.
    Snapshotting the module registry used to paper over that; the scoping is the fix.
    """
    s, ctl, _ = street_take()
    ruleset = rules.RuleSet()
    active = ruleset.names()
    inv = Invariants()
    inv.visible_windows = dict(visible or {})
    acts = [ag.Spawn(e) for e in spawns] + [ag.AddRule(n, src) for n, src in sources]
    active = ag.apply(s, acts, active, ruleset)
    for k in range(frames):
        ruleset.step(s, active)
        wc.advance_gait(s)
        h = s.get("courier")
        cam = ctl.update(h.pos, h.heading_deg, 1.0 / FPS)
        inv.check(s, cam, k, mounts=mounts, flying=flying)
    inv.finish()
    ok = not inv.violations
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok and not quiet:
        for line in inv.report().splitlines()[1:]:
            print("   ", line)
    return ok, s


def t_approach_and_stop():
    src = A.build(A.APPROACH_AND_STOP, "approach", eid="taxi", sign=1.0,
                  stop_at=6.0, v_min=1.5, v_max=9.0, gain=0.55)
    e = Entity(eid="taxi", kind="vehicle", position=[1.7, 0.0, 46.0],
               heading_deg=180.0, speed=8.0, size=[2.4, 2.1, 5.0])
    ok, s = run("approach_and_stop", 300, [e], [("approach", src)],
                visible={"taxi": (40, 150)})
    z = float(s.get("taxi").pos[2])
    if not (5.0 < z < 7.5):
        print(f"     stopped at z={z:.2f}, wanted ~6.0"); ok = False
    return ok


def t_board_and_drive():
    boards = A.build(A.BOARD_VEHICLE, "board", vehicle="taxi", rider="courier",
                     board_s=1.2, seat_frac=0.55)
    drive = A.build(A.DRIVE_OFF, "drive", vehicle="taxi", rider="courier",
                    v_max=7.0, accel=2.2, seat_frac=0.55)
    appr = A.build(A.APPROACH_AND_STOP, "approach", eid="taxi", sign=1.0,
                   stop_at=6.0, v_min=1.5, v_max=9.0, gain=0.55)
    halt = A.build(A.HALT_SUBJECT, "halt", eid="taxi", subject="courier", radius=9.0)
    e = Entity(eid="taxi", kind="vehicle", position=[1.7, 0.0, 30.0],
               heading_deg=180.0, speed=8.0, size=[2.4, 2.1, 5.0])
    ok, s = run("board_and_drive", 320, [e],
                [("approach", appr), ("halt", halt), ("board", boards), ("drive", drive)],
                mounts={"courier": Mount("taxi", seat_frac=0.55, seated_from=210)},
                visible={"taxi": (30, 300)})
    v = s.get("taxi")
    if not v.attrs.get("boarded"):
        print("     never boarded"); ok = False
    elif float(v.speed) < 3.0:
        print(f"     boarded but speed only {v.speed:.1f}"); ok = False
    return ok


def t_mount_and_ride():
    mnt = A.build(A.MOUNT, "mount", mount="camel", rider="courier",
                  climb_s=1.1, seat_frac=0.78, ride_speed=2.1, reach=3.0)
    ride = A.build(A.RIDE, "ride", mount="camel", rider="courier",
                   seat_frac=0.78, bob=0.045, bob_hz=6.2)
    e = Entity(eid="camel", kind="animal", position=[0.0, 0.0, 3.0],
               heading_deg=180.0, speed=0.0, size=[1.05, 2.25, 3.1])
    ok, s = run("mount_and_ride", 220, [e], [("mount", mnt), ("ride", ride)],
                mounts={"courier": Mount("camel", seat_frac=0.78, seated_from=40)},
                visible={"camel": (60, 210)})
    if not s.get("camel").attrs.get("mounted"):
        print("     never mounted"); ok = False
    return ok


def t_fly_on_prop():
    fly = A.build(A.FLY_ON_PROP, "fly", rider="courier", prop="broom",
                  climb_s=3.6, alt_max=13.5, drift=2.4, offset=0.62)
    e = Entity(eid="broom", kind="prop", position=[0.0, 0.0, 0.0],
               heading_deg=180.0, size=[0.10, 0.10, 1.9])
    ok, s = run("fly_on_prop", 140, [e], [("fly", fly)],
                flying={"courier", "broom"})
    h, b = s.get("courier"), s.get("broom")
    alt = -float(h.pos[1])
    if alt < 10.0:
        print(f"     only climbed to {alt:.1f} m"); ok = False
    # +Y is down, so the prop must have the LARGER y to be underneath.
    if float(b.pos[1]) <= float(h.pos[1]):
        print(f"     prop y={float(b.pos[1]):+.2f} is above rider y={float(h.pos[1]):+.2f}"); ok = False
    return ok


def t_descend_and_ballistic():
    desc = A.build(A.DESCEND_PAST, "descend", eid="gunship", dur_s=4.0, alt0=22.0, alt1=7.0)
    ball = A.build(A.BALLISTIC, "ballistic", eids='("ord0",)')
    g = Entity(eid="gunship", kind="vehicle", position=[-4.0, -22.0, -66.0],
               heading_deg=0.0, speed=11.0, size=[15.0, 3.4, 11.0])
    o = Entity(eid="ord0", kind="prop", position=[-3.0, -13.0, -34.0],
               heading_deg=0.0, size=[0.5, 0.5, 1.8], attrs={"alt0": 13.0, "t": -0.4})
    ok, s = run("descend_and_ballistic", 150, [g, o], [("descend", desc), ("ballistic", ball)],
                flying={"gunship", "ord0"},
                visible=None)  # heading is scenario-specific; covered in the take test
    if float(-s.get("gunship").pos[1]) > 9.0:
        print(f"     gunship still at {-float(s.get('gunship').pos[1]):.1f} m"); ok = False
    if float(s.get("ord0").pos[1]) < -0.01:
        print(f"     ordnance never landed (y={float(s.get('ord0').pos[1]):+.2f})"); ok = False
    return ok


def t_flight_camera():
    """The follow rig must let a climb read as a climb.

    Four things were wrong or unguarded at once. The rig forced `-abs(pitch)`, so it stared
    at the ground while the subject was above it and he left the top of the frame at 425 and
    never came back. The boom was rigid in Y, so it rose with him and held a 4 m stand-off
    all the way to 14 m: the same framing as a walk. Freeing the pitch alone then aimed at
    empty sky, and 38 frames of conditioning came back 98% blank. And nothing checked any of
    it, because `visible_windows` never listed the protagonist.
    """
    from compiler.follow import FollowCamera
    from world.scenes import StreetSpec, camera as scene_camera

    ok = True
    dt = 1.0 / FPS
    spec = StreetSpec()

    def fly(**kw):
        """Run a 0 -> 14 m climb and report (camera, subject) altitude and boom per frame."""
        cam = scene_camera(spec) if not kw else FollowCamera(
            distance=spec.boom_len, height=spec.cam_height, **kw)
        out = []
        for k in range(120):
            alt = 14.0 * min(k / 96.0, 1.0) ** 2
            tgt = np.array([0.0, -alt, 2.2 * k * dt])
            c = cam.update(tgt, 180.0, dt)
            out.append((-float(c.position[1]), alt, c.pitch_deg,
                        float(np.linalg.norm(c.position - tgt))))
        return out

    # 1. A rider sitting 1.76 m up on a camel is below the knee, so the rig is exactly the
    #    rigid boom it has always been. This is what keeps the mounted sections untouched.
    # Both rigs come from the same spec so only rise_gain/pull_back differ; built by hand
    # they also differed in boom_tau and zeta, and two different springs are not a test of
    # the knee.
    rigid = scene_camera(StreetSpec(rise_gain=1.0, pull_back=0.0))
    knee = scene_camera(spec)
    for k in range(60):
        tgt = np.array([0.0, -1.76, 2.2 * k * dt])
        a, b = rigid.update(tgt, 180.0, dt), knee.update(tgt, 180.0, dt)
        if not np.array_equal(a.position, b.position) or a.pitch_deg != b.pitch_deg:
            print(f"     below the knee the rig is not the rigid boom (frame {k})")
            ok = False
            break

    tr = fly()

    # 2. The camera must be allowed to look UP once the subject is above it.
    above = [(c, a, pi, d) for c, a, pi, d in tr if a > c + 0.5]
    if not above:
        print("     subject never gets above the camera; the climb is not being lagged")
        ok = False
    elif max(pi for _, _, pi, _ in above) <= 0.0:
        print(f"     camera never looks up (max pitch {max(pi for _,_,pi,_ in above):+.1f} deg) "
              f"while the subject is above it")
        ok = False

    # 3. The subject must genuinely gain height on the camera, not be carried along.
    sep = tr[-1][1] - tr[-1][0]
    if sep < 4.0:
        print(f"     camera rose with the subject: only {sep:.1f} m of separation at 14 m")
        ok = False

    # 4. And the boom must lengthen, or the ground leaves the shot and the conditioning
    #    goes blank.
    if tr[-1][3] < 3.0 * tr[0][3]:
        print(f"     boom did not pull back: {tr[0][3]:.1f} m -> {tr[-1][3]:.1f} m")
        ok = False

    print(f"[{'PASS' if ok else 'FAIL'}] flight_camera "
          f"(sep {sep:.1f} m, boom {tr[0][3]:.1f}->{tr[-1][3]:.1f} m, "
          f"max pitch {max(pi for _, _, pi, _ in tr):+.1f} deg)")
    return ok


def t_caption_compiler():
    """Captions must be compiled from the simulation, and the gate must be able to tell.

    The captions this replaces named every correct noun while describing the wrong action:
    "car" appears 11 times in the old window 0 and 8 in window 1, as scenery the protagonist
    stands beside, so a keyword check waves them through. The check therefore has to read
    what the PROTAGONIST does, and this test fails if it ever stops doing that.
    """
    import tempfile
    from pathlib import Path

    from world import narrate
    from world.take_content import content as take_content

    ok = True
    bad = narrate.audit_phrases()
    if bad:
        ok = False
        for b in bad[:4]:
            print("     " + b)

    # 1. Compiled captions agree with the simulation they came from.
    plan, _, _ = simulate_take(check=False)
    with tempfile.TemporaryDirectory() as td:
        narrate.build(plan, TAKE_PHASES, take_content(), td, TAKE_WINDOWS)
        problems = narrate.check(td, plan, TAKE_PHASES, TAKE_WINDOWS,
                                 actor_names=take_content()["actor_names"])
        if problems:
            ok = False
            print(f"     compiled captions do not match their own simulation: {problems[0]}")

        # 2. The decisive one: a sentence stuffed with every right noun but describing the
        #    wrong action must NOT pass. A noun-frequency check passes this happily.
        noun_rich = ("The car waits on the wet lane beside the camel and the broom while the "
                     "rainkeeper holds his mark on the streaming stone")
        got = narrate.classify(noun_rich)
        if got != narrate.STAND:
            ok = False
            print(f"     noun-rich standing sentence classified {got}, wanted STAND")
        for noun in ("car", "camel", "broom"):
            if noun not in noun_rich:
                ok = False
        # ... and the same nouns while he is actually driving must read as DRIVE, not as
        # whichever noun happens to appear most.
        drive = ("Past the camel and the parked car he rides inside the rain-cart, the broom "
                 "still strapped behind him")
        if narrate.classify(drive) != narrate.DRIVE:
            ok = False
            print(f"     noun-rich driving sentence classified {narrate.classify(drive)}")

        # 3. Corrupting one segment must be caught. Rewrite window 2 to claim he is standing
        #    when the simulation has him boarding or driving.
        w2 = Path(td) / "002.txt"
        txt = w2.read_text(encoding="utf-8")
        head, rest = txt.split("Protagonist action timeline:\n\n", 1)
        body, tail = rest.split("\n\nOther visible actors:", 1)
        segs = body.split("; ")
        segs[0] = segs[0].split(": ", 1)[0] + ": " + noun_rich + "."
        w2.write_text(head + "Protagonist action timeline:\n\n" + "; ".join(segs)
                      + "\n\nOther visible actors:" + tail, encoding="utf-8")
        if not narrate.check(td, plan, TAKE_PHASES, TAKE_WINDOWS,
                             actor_names=take_content()["actor_names"]):
            ok = False
            print("     a caption claiming he stands while the simulation drives him PASSED")

    print(f"[{'PASS' if ok else 'FAIL'}] caption_compiler "
          f"({len(narrate.PHRASES)} states, {len(narrate.SIGNATURE)} signatures)")
    return ok


def t_actor_captions():
    """The other-actors line must track each actor's real motion, and the gate must check it.

    Measured, not argued: window 1 was generated twice from the same seed, same proxy and
    same window-0 prefix, differing only in whether its caption said the rain-cart was
    travelling or halted. With "travels steadily" the cart was absent from the render and
    the proxy's 2 m depth slab came back as a parapet; with "halted alongside" a lacquered
    carriage appeared and the protagonist stepped into it. The caption decides whether the
    vehicle exists, so a caption that misreports motion is not cosmetic.
    """
    import tempfile
    from pathlib import Path

    from world import narrate
    from world.take_content import content as take_content

    ok = True
    plan, _, _ = simulate_take(check=False)
    track = narrate.derive_track(plan, TAKE_PHASES)
    names = take_content()["actor_names"]

    # 1. The cart really is halted over the stretch the control tested (proxy 180-210).
    phases = narrate.actor_phases(plan, "taxi", 90, 213, track)
    covering = [st for st, a, b in phases if a <= 195 <= b]
    if covering != [narrate.A_HALT]:
        ok = False
        print(f"     at proxy f195 the simulation has the cart {covering}, wanted A_HALT")

    # 2. Every actor signature is distinguishable from every other.
    for x, sx in narrate.ACTOR_SIGNATURE.items():
        for y, sy in narrate.ACTOR_SIGNATURE.items():
            if x != y and (sx in sy or sy in sx):
                ok = False
                print(f"     actor signatures {x}/{y} not distinguishable")
    for st, phrase in narrate.ACTOR_PHRASE.items():
        if narrate.classify_actor(phrase) != st:
            ok = False
            print(f"     ACTOR_PHRASE[{st}] classifies as {narrate.classify_actor(phrase)}")

    with tempfile.TemporaryDirectory() as td:
        narrate.build(plan, TAKE_PHASES, take_content(), td, TAKE_WINDOWS)
        if narrate.check(td, plan, TAKE_PHASES, TAKE_WINDOWS, actor_names=names):
            ok = False
            print("     freshly compiled captions failed their own actor check")

        # 3. The decisive negative: claim the cart is receding over the stretch where the
        #    simulation has it stopped dead. This is the shape of the real defect.
        w1 = Path(td) / "001.txt"
        txt = w1.read_text(encoding="utf-8")
        halted = narrate.ACTOR_PHRASE[narrate.A_HALT]
        if halted not in txt:
            ok = False
            print("     window 1 does not describe the cart as halted at all")
        else:
            w1.write_text(txt.replace(halted, narrate.ACTOR_PHRASE[narrate.A_AWAY]),
                          encoding="utf-8")
            probs = narrate.check(td, plan, TAKE_PHASES, TAKE_WINDOWS, actor_names=names)
            if not any("rain-cart" in x for x in probs):
                ok = False
                print("     a caption claiming the halted cart drives away PASSED the gate")

    print(f"[{'PASS' if ok else 'FAIL'}] actor_captions "
          f"({len(narrate.ACTOR_SIGNATURE)} actor states)")
    return ok


def t_water_surface():
    """Water must actually reach the DUV, and must vanish when the scene has none.

    Class 2 is 6.144% of the released pixels and appears in 23 of their 40 example
    directories (39 distinct geometries; 32 and 33 are byte-identical), and we
    emitted exactly none of it: `KIND_TO_CLASS` mapped water to 2 all along but no primitive
    could produce it. The captions had been describing "flooded terraces" over dry geometry.
    """
    import numpy as np

    from compiler.render import Camera, Rect, render_frame
    from world import compile as wc
    from world.scenes import StreetSpec, build
    from world.state import KIND_TO_CLASS

    ok = True
    cam = Camera(focal_px=206.0, position=np.array([0.0, -1.83, 0.0]), pitch_deg=-8.0,
                 yaw_deg=0.0)

    wet = build(StreetSpec())
    _z, ids = render_frame(cam, wc.PrimitiveCache().build(wet))
    n_wet = int((ids == KIND_TO_CLASS["water"]).sum())
    if n_wet == 0:
        print("     the default scene renders no water at all")
        ok = False

    # Negative: the same scene with the water turned off must produce none. If this still
    # shows class 2, the pixels were coming from something other than the water entities.
    dry = build(StreetSpec(n_terraces=0, channel_w=0.0))
    _z, ids_dry = render_frame(cam, wc.PrimitiveCache().build(dry))
    n_dry = int((ids_dry == KIND_TO_CLASS["water"]).sum())
    if n_dry != 0:
        print(f"     scene with no water entities still rendered {n_dry} class-2 pixels")
        ok = False

    # A sheet behind the camera must be culled, not drawn: Rect.bound() exists so the
    # screen-space cull can reject it the way it rejects every other primitive.
    behind = Rect(y=-0.02, x0=-40, x1=40, z0=-400, z1=-5, class_id=2)
    if behind.bound() is None:
        print("     Rect.bound() is None, so water falls back to a full-frame intersect")
        ok = False
    _z, ids_b = render_frame(cam, [behind])
    if int((ids_b == 2).sum()) != 0:
        print("     a water sheet entirely behind the camera was rendered")
        ok = False

    print(f"[{'PASS' if ok else 'FAIL'}] water_surface "
          f"(wet {n_wet} px, dry {n_dry} px)")
    return ok


def t_class_vocabulary():
    """Never emit a semantic id the released examples do not use.

    Measured over all 40 directories (39 distinct geometries): classes 3, 6, 7 and 11
    appear in none of them, so the vocabulary
    is {0,1,2,4,5,8,9,10}. We shipped class 11 on every broom frame -- an id the model was
    never trained on. This fails if anything drifts back outside the set.
    """
    import numpy as np

    from compiler.render import Camera, render_frame
    from world import compile as wc
    from world.state import RELEASED_CLASSES

    ok = True
    plan, _, _ = simulate_take(check=False)
    cam = Camera(focal_px=206.0, position=np.array([0.0, -1.83, 0.0]), pitch_deg=-8.0)
    seen = set()
    # Sample across the take so every kind that ever spawns is covered, the broom included.
    for k in (0, 120, 200, 260, 330, 400, 460, 483):
        st = WorldState.from_json(plan[k][1])
        _z, ids = render_frame(cam, wc.PrimitiveCache().build(st))
        seen |= {int(v) for v in np.unique(ids)}
    # class 1 is the miss/sky id, written by the frame writer rather than a primitive.
    bad = sorted(seen - RELEASED_CLASSES - {1})
    if bad:
        print(f"     emitted classes {bad}, which appear in 0 of the released 40")
        ok = False

    # Negative: a prop that claims an out-of-vocabulary id must be caught by this test.
    from world.state import Entity
    st = WorldState(seed=0)
    st.add(Entity("bad", "prop", [0.0, 0.0, 6.0], size=[0.4, 0.4, 2.0],
                  attrs={"class_id": 11}))
    _z, ids = render_frame(cam, wc.PrimitiveCache().build(st))
    if 11 not in {int(v) for v in np.unique(ids)}:
        print("     could not stage the negative case: class 11 prop rendered nothing")
        ok = False
    elif not (sorted({int(v) for v in np.unique(ids)} - RELEASED_CLASSES - {1}) == [11]):
        print("     the vocabulary check does not flag a class-11 prop")
        ok = False

    print(f"[{'PASS' if ok else 'FAIL'}] class_vocabulary "
          f"(take emits {sorted(seen)}, released {sorted(RELEASED_CLASSES)})")
    return ok


def t_camera_modes():
    """The rig must express the shot types their examples use, and "follow" must not move.

    Counted over the released 40: follows 39, rises/cranes 37, holds/static 36,
    pans/rotates 27, orbits/arcs 26, pushes or pulls 25. We had only follow, so five of
    their six shot types were inexpressible. Each mode is checked by the thing that
    distinguishes it, and each check fails if that mode silently degrades to follow.
    """
    import numpy as np

    from compiler.follow import FollowCamera

    ok = True
    dt = 1.0 / FPS

    def run(mode, n=96, speed=2.5, **kw):
        cam = FollowCamera(distance=2.40, height=1.83, mode=mode, **kw)
        out = []
        for k in range(n):
            tgt = np.array([0.0, 0.0, speed * k * dt])
            c = cam.update(tgt, 0.0, dt)
            out.append((c.position.copy(), float(np.linalg.norm(c.position - tgt)), tgt.copy()))
        return out

    # follow: the boom holds its stand-off as the subject walks away.
    f = run("follow")
    if abs(f[-1][1] - f[len(f) // 2][1]) > 0.6:
        print(f"     follow stand-off drifted {f[len(f)//2][1]:.2f} -> {f[-1][1]:.2f} m")
        ok = False

    # hold: the rig stops moving, so the subject recedes from it.
    h = run("hold")
    moved = float(np.linalg.norm(h[-1][0] - h[len(h) // 2][0]))
    if moved > 1e-9:
        print(f"     hold moved the camera {moved:.3f} m; it must be locked off")
        ok = False
    if h[-1][1] <= h[len(h) // 2][1]:
        print("     hold did not let the subject recede from the camera")
        ok = False

    # lock: body AND aim frozen, so the subject drifts out of centre. `hold` does not do
    # this -- it keeps aiming, which is why driving ex28's "preserving the exact initial
    # layout" from hold left its centroid at 0.495, exactly the symptom being fixed.
    from compiler.render import OUT_W
    lk = FollowCamera(distance=2.40, height=1.83, mode="lock")
    hd = FollowCamera(distance=2.40, height=1.83, mode="hold")
    lx, hx = [], []
    for k in range(96):
        # The subject must move ACROSS the frame, not straight down the optical axis: a
        # locked camera watching someone recede sees them shrink without shifting, so a
        # straight-ahead walk cannot distinguish lock from hold at all.
        tgt = np.array([2.2 * k * dt, 0.0, 2.5 * k * dt])
        for cam, acc in ((lk, lx), (hd, hx)):
            c = cam.update(tgt, 0.0, dt)
            cp = c.to_camera(tgt + np.array([0.0, -1.0, 0.0]))[0]
            if cp[2] > 0:
                acc.append(float(cp[0] / cp[2] * c.focal_px + OUT_W / 2) / OUT_W)
    if abs(lx[-1] - 0.5) < 0.05:
        print(f"     lock kept the subject centred at {lx[-1]:.3f}; the aim is not frozen")
        ok = False
    if abs(hx[-1] - 0.5) > 0.05:
        print(f"     hold let the subject leave centre at {hx[-1]:.3f}; it should keep aiming")
        ok = False

    # orbit: the boom's bearing around the subject sweeps.
    o = run("orbit", orbit_rate=30.0)
    def bearing(rec):
        d = rec[0] - rec[2]
        return np.degrees(np.arctan2(d[0], d[2]))
    swept = abs((bearing(o[-1]) - bearing(o[0]) + 180) % 360 - 180)
    if swept < 20.0:
        print(f"     orbit swept only {swept:.1f} deg; mode is being ignored")
        ok = False

    # push and pull: the boom length is driven toward push_to.
    # Stationary subject: while he walks, the follow spring trails him by roughly
    # speed * boom_tau, and that lag (~1.4 m at 2.5 m/s) is larger than the boom change
    # being measured. An earlier version of this test read the lag as a failed push.
    pu = run("push", speed=0.0, push_to=1.2, push_rate=2.0)
    pl = run("pull", speed=0.0, push_to=14.0, push_rate=4.0)
    if pu[-1][1] >= pu[0][1]:
        print(f"     push did not shorten the boom ({pu[0][1]:.2f} -> {pu[-1][1]:.2f} m)")
        ok = False
    if pl[-1][1] <= pl[0][1] + 2.0:
        print(f"     pull did not lengthen the boom ({pl[0][1]:.2f} -> {pl[-1][1]:.2f} m)")
        ok = False

    print(f"[{'PASS' if ok else 'FAIL'}] camera_modes "
          f"(hold fixed and still centred {hx[-1]:.2f}, lock drifts to {lx[-1]:.2f}, "
          f"orbit {swept:.0f} deg, push {pu[0][1]:.1f}->{pu[-1][1]:.1f} m, "
          f"pull {pl[0][1]:.1f}->{pl[-1][1]:.1f} m)")
    return ok


def t_aim_screen_x():
    """The rig must put the subject where it is told, not always at frame centre.

    Our six reproduced proxies sat at horizontal centroid 0.49-0.50 with near-zero variance
    while the released six span 0.350-0.499. That is not a tuning error: a follow boom
    solves one constraint, keep the subject projected at centre, and 0.49 is its fixed
    point. `shoulder_offset` cannot fix it -- it moves the camera body and the aim rotates
    straight back onto the subject.
    """
    import numpy as np

    from compiler.follow import FollowCamera
    from compiler.render import OUT_W

    ok = True

    def landed(fx):
        cam = FollowCamera(distance=2.4, height=1.7, aim_screen_x=fx)
        for k in range(60):
            c = cam.update(np.array([0.0, 0.0, 2.0 * k / FPS]), 0.0, 1.0 / FPS)
        tgt = np.array([0.0, -1.0, 2.0 * 59 / FPS])
        cp = c.to_camera(tgt)[0]
        return float(cp[0] / cp[2] * c.focal_px + OUT_W / 2) / OUT_W

    for fx in (0.5, 1.0 / 3.0, 2.0 / 3.0, 0.25):
        got = landed(fx)
        if abs(got - fx) > 0.02:
            print("     asked for %.3f, subject landed at %.3f" % (fx, got))
            ok = False

    # Negative: the default rig centres the subject, so if the offset is ever dropped every
    # request collapses to 0.5 and this separation disappears.
    spread = abs(landed(1.0 / 3.0) - landed(2.0 / 3.0))
    if spread < 0.25:
        print("     left and right framings differ by only %.3f; the offset is not applied"
              % spread)
        ok = False

    print("     1/3 -> %.3f, centre -> %.3f, 2/3 -> %.3f"
          % (landed(1.0 / 3.0), landed(0.5), landed(2.0 / 3.0)))
    print(f"[{'PASS' if ok else 'FAIL'}] aim_screen_x")
    return ok


def t_shot_parse():
    """The shot clause must be read from the prompt, and read the same way for all 40.

    Guards against the failure this replaces: the scene author ignored the shot clause
    entirely, so every example got the same framing. It also guards the opposite failure --
    a parser tuned until the six figures look right. The coverage assertions below are over
    the whole release, so a pattern that only fires on the six would fail them.
    """
    import os

    from world.shot import CENTRE, THIRD_LEFT, parse_shot

    ok = True
    root = os.environ.get("CWM_EXAMPLES", "examples40")
    if not os.path.isdir(root):
        print("     released examples not present (set CWM_EXAMPLES); parser checked on "
              "synthetic text only")
        synthetic = [
            ("The trailing camera begins near shoulder height behind her, holding the "
             "opening layout with the two rovers visible.", dict(hold=True)),
            ("She stands left of center with her back to the camera. The camera begins "
             "close behind her.", dict(subject_x=THIRD_LEFT)),
            ("The camera begins behind the sentinel in a steep downward-looking aerial "
             "view, with the sentinel centered prominently.",
             dict(subject_x=CENTRE, aerial=True)),
        ]
        for text, want in synthetic:
            sh = parse_shot(text)
            for k, v in want.items():
                if getattr(sh, k) != v:
                    print("     %r -> %s=%r, wanted %r" % (text[:40], k, getattr(sh, k), v))
                    ok = False
        print(f"[{'PASS' if ok else 'FAIL'}] shot_parse (synthetic only)")
        return ok

    exs = sorted(int(d) for d in os.listdir(root) if d.isdigit())
    shots = {}
    for ex in exs:
        f = os.path.join(root, str(ex), "prompts", "000.txt")
        with open(f, encoding="utf-8") as fh:
            shots[ex] = parse_shot(fh.read())

    # Every released prompt has the camera sentence the parse anchors on.
    no_cam = [ex for ex in exs if not shots[ex].source]
    if no_cam:
        print("     no camera sentence found in %s" % no_cam[:6])
        ok = False

    # Placement is the dominant signal and must parse for most of the release, not just
    # for the examples in the figures.
    n_place = sum(1 for ex in exs if shots[ex].subject_x is not None)
    if n_place < len(exs) // 2:
        print("     subject placement parsed for only %d/%d" % (n_place, len(exs)))
        ok = False

    # And it must not collapse to one value, or it is not reading anything.
    vals = {shots[ex].subject_x for ex in exs if shots[ex].subject_x is not None}
    if len(vals) < 2:
        print("     every placement parsed to the same value %r" % vals)
        ok = False

    n_hold = sum(1 for ex in exs if shots[ex].hold)
    print("     %d/%d placement, %d/%d hold, %d distinct placements"
          % (n_place, len(exs), n_hold, len(exs), len(vals)))
    print(f"[{'PASS' if ok else 'FAIL'}] shot_parse")
    return ok


def t_take_is_reentrant():
    """Two simulate() calls in one process must produce the same take.

    `AddRule` registers into a module-level registry in world/rules.py that outlives the
    take, and simulate() used to seed its active set from `sorted(rules.names())`. So the
    second run in a process started with every rule the FIRST run had added, active from
    frame 0, before the entities they reference exist -- FLY_ON_PROP fired immediately and
    the protagonist spent the whole take 14 m in the air.

    This also guards the reason that was invisible: the caption gate could not catch it,
    because compiling captions and then checking them reads the same window_states twice
    and agrees with itself no matter what the world is doing.
    """
    import json as _json
    import subprocess
    import sys as _sys
    from pathlib import Path

    import take_actions as _ta

    import numpy as np

    # A FRESH interpreter, because this test must not depend on what ran before it. Run
    # in-process it passes vacuously: by the time it executes, earlier tests have already
    # filled the registry, so it measures 14 -> 14 and sees no leak.
    probe = (
        "import json, sys\n"
        "sys.path.insert(0, %r)\n"
        "from world import rules\n"
        "import take_actions as T\n"
        "before = sorted(rules.names())\n"
        "a, _, _ = T.simulate(check=False)\n"
        "mid = sorted(rules.names())\n"
        "b, _, _ = T.simulate(check=False)\n"
        "ya = [json.loads(f[1])['entities']['courier']['position'][1] for f in a]\n"
        "yb = [json.loads(f[1])['entities']['courier']['position'][1] for f in b]\n"
        "print(json.dumps({'before': before, 'mid': mid, 'ya': ya, 'yb': yb}))\n"
        % str(Path(_ta.__file__).resolve().parent)
    )
    r = subprocess.run([_sys.executable, "-c", probe], capture_output=True, text=True)
    if r.returncode != 0:
        print("     probe subprocess failed:", r.stderr.strip().splitlines()[-1:])
        print("[FAIL] take_is_reentrant")
        return False
    d = _json.loads(r.stdout.strip().splitlines()[-1])
    before, mid = set(d["before"]), set(d["mid"])
    ya, yb = np.array(d["ya"]), np.array(d["yb"])

    ok = True

    # The registry must not grow across takes.
    if mid - before:
        print(f"     registry leaked {sorted(mid - before)} after one take")
        ok = False

    # And the two takes must be identical, frame for frame.
    if not np.array_equal(ya, yb):
        k = int(np.argmax(ya != yb))
        print(f"     run 2 diverges at frame {k}: courier y {ya[k]:+.2f} vs {yb[k]:+.2f}")
        ok = False

    # The specific symptom: airborne at frame 0 means a flight rule from a previous take.
    if float(-yb[0]) > 0.3:
        print(f"     run 2 starts {-yb[0]:.1f} m in the air")
        ok = False

    print(f"[{'PASS' if ok else 'FAIL'}] take_is_reentrant "
          f"(registry {len(before)} -> {len(mid)}, run2 opens at {-yb[0]:.2f} m)")
    return ok


if __name__ == "__main__":
    import numpy as np  # actions reference np inside their exec namespace
    tests = [t_approach_and_stop, t_board_and_drive, t_mount_and_ride,
             t_fly_on_prop, t_descend_and_ballistic, t_flight_camera,
             t_caption_compiler, t_actor_captions,
             t_water_surface, t_class_vocabulary, t_camera_modes,
             t_aim_screen_x, t_shot_parse, t_take_is_reentrant]
    results = [t() for t in tests]
    n_ok = sum(bool(r) for r in results)
    print(f"\n{n_ok}/{len(results)} actions clean")
    sys.exit(0 if n_ok == len(results) else 1)

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


# The take this repository ships, as phase runs over its 484 frames. Pinned because
# nothing else can pin it: `narrate.check` answers "do these captions describe this
# simulation?", and it answers correctly even when the simulation is wrong -- a take where
# a leaked rule flies the protagonist from frame 0 really IS a flying take, so captions
# saying FLY are a true description of it. The question "is this the take we meant?" has
# no source but an authored expectation, so here is the expectation.
EXPECTED_RUNS = [("WALK", 0, 79), ("STAND", 80, 177), ("BOARD", 178, 208),
                 ("DRIVE", 209, 249), ("MOUNT", 250, 277), ("RIDE_MOUNT", 278, 391),
                 ("FLY", 392, 483)]


def _runs(track):
    out = []
    for k, (st, _obj) in enumerate(track):
        if out and out[-1][0] == st:
            out[-1][2] = k
        else:
            out.append([st, k, k])
    return [tuple(r) for r in out]


def _check_expected_take(track, where):
    """True when `track` is the shipped take. Prints the first divergence otherwise."""
    got = _runs(track)
    if got == EXPECTED_RUNS:
        return True
    print(f"     {where}: simulation is not the shipped take")
    print(f"       expected {EXPECTED_RUNS}")
    print(f"       got      {got}")
    return False


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

    # check=True, not False. These three tests opted out of the invariants, and the
    # polluted world they were passing against produced 119 violations including
    # "mount_height: courier sits at 14.00 m, seat of taxi is 1.16 m". The invariants
    # were never the weak link -- the tests were declining to run them.
    # 1. Compiled captions agree with the simulation they came from.
    plan, _, _inv = simulate_take(check=True)
    if _inv.violations:
        print(f'     simulation violates its own invariants: {_inv.violations[0].detail}')
        ok = False
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

        # 3. The gate must be sensitive to the SIMULATION, not only to the file. Both
        #    existing negatives corrupt the text; this one keeps the text and moves the
        #    world underneath it. Without this, breaking `window_states` so that build and
        #    check both read the wrong frames would leave check(build(x)) == [] and every
        #    other assertion here still green.
        for shift in (15, 90):
            other = plan[shift:] + [plan[-1]] * shift
            moved = narrate.check(td, other, TAKE_PHASES, TAKE_WINDOWS,
                                  actor_names=take_content()["actor_names"])
            if not moved:
                print(f"     captions still matched a take shifted {shift} frames; the "
                      f"check does not read the simulation")
                ok = False

        # 4. Corrupting one segment must be caught. Rewrite window 2 to claim he is standing
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
    # check=True, not False. These three tests opted out of the invariants, and the
    # polluted world they were passing against produced 119 violations including
    # "mount_height: courier sits at 14.00 m, seat of taxi is 1.16 m". The invariants
    # were never the weak link -- the tests were declining to run them.
    plan, _, _inv = simulate_take(check=True)
    if _inv.violations:
        print(f'     simulation violates its own invariants: {_inv.violations[0].detail}')
        ok = False
    track = narrate.derive_track(plan, TAKE_PHASES)
    names = take_content()["actor_names"]

    # This test passed against a world where a leaked rule had the protagonist airborne
    # from frame 0, because the captions described that world correctly. Pin the world.
    if not _check_expected_take(track, "t_actor_captions"):
        ok = False

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
    # check=True, not False. These three tests opted out of the invariants, and the
    # polluted world they were passing against produced 119 violations including
    # "mount_height: courier sits at 14.00 m, seat of taxi is 1.16 m". The invariants
    # were never the weak link -- the tests were declining to run them.
    plan, _, _inv = simulate_take(check=True)
    if _inv.violations:
        print(f'     simulation violates its own invariants: {_inv.violations[0].detail}')
        ok = False
    from world import narrate as _nr
    if not _check_expected_take(_nr.derive_track(plan, TAKE_PHASES), "t_class_vocabulary"):
        ok = False
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
        # Every frame's full state, not just the courier: a leak into any entity counts,
        # and courier altitude is how the symptom SHOWS, not how it is detected.
        "same = sum(1 for x, y in zip(a, b) if x[1] == y[1])\n"
        # A fresh RuleSet must still hold exactly the builtins. This is the invariant the
        # scoping establishes, and unlike `mid - before` it can still fail: it catches a
        # RuleSet that shares one dict instead of copying, or anything gaining write
        # access to _BUILTIN. `mid - before` cannot fail any more, precisely because the
        # fix made rules.names() read a dict only the decorator writes.
        "fresh = sorted(rules.RuleSet().names())\n"
        "ya = [json.loads(f[1])['entities']['courier']['position'][1] for f in a]\n"
        "yb = [json.loads(f[1])['entities']['courier']['position'][1] for f in b]\n"
        "print(json.dumps({'before': before, 'mid': mid, 'same': same, 'fresh': fresh,\n"
        "                  'n': len(a), 'ya': ya, 'yb': yb}))\n"
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

    # The registry must not grow across takes. NOTE: this assertion is retired by the fix
    # itself -- rules.names() now reads a dict only the decorator writes -- so it is kept
    # as a named symptom, not as coverage.
    if mid - before:
        print(f"     registry leaked {sorted(mid - before)} after one take")
        ok = False

    # The live version of that check: a take must not be able to change what a NEW take
    # starts from, by any route, including a RuleSet that aliases instead of copying.
    if sorted(d["fresh"]) != sorted(d["before"]):
        print(f"     a fresh RuleSet starts from {sorted(d['fresh'])}, "
              f"not the builtins {sorted(d['before'])}")
        ok = False

    # Every frame of the two takes must match, over the whole state and not one field.
    if d["same"] != d["n"]:
        print(f"     only {d['same']}/{d['n']} frames identical between the two runs")
        ok = False

    if not np.array_equal(ya, yb):
        k = int(np.argmax(ya != yb))
        print(f"     run 2 diverges at frame {k}: courier y {ya[k]:+.2f} vs {yb[k]:+.2f}")
        ok = False

    # The specific symptom: airborne at frame 0 means a flight rule from a previous take.
    if float(-yb[0]) > 0.3:
        print(f"     run 2 starts {-yb[0]:.1f} m in the air")
        ok = False

    print(f"[{'PASS' if ok else 'FAIL'}] take_is_reentrant "
          f"(registry {len(before)} -> {len(mid)}, identical frames {d['same']}/{d['n']}, "
          f"run2 opens at {-yb[0]:.2f} m)")
    return ok


def t_visibility_counts_near_objects():
    """An object the camera is inside the bounding sphere of is visible, not invisible.

    `screen_bbox` has three outcomes: `()` dead or wholly offscreen, `None` for a sphere
    that straddles the eye plane and cannot be bounded, and a tuple otherwise. The
    invariant tested `if bb:`, and `None` is falsy, so the one case meaning "this thing is
    all around the lens" scored the same as "this thing is not in shot". Measured on the
    shipped take, the taxi at f132 fills 11.4% of the frame and was counted not visible;
    the report read taxi 73% and gunship 79% where the truth is 100% and 100%. It passed
    only because 73 and 79 clear the 60% default, so the margin was being eaten silently.

    Also pins the sphere to the body: `courier` carries the default size [1,1,1], so the
    0.97 threshold on the protagonist was testing a 0.5 m ball at his knees.
    """
    import numpy as np

    from compiler.render import Camera, screen_bbox
    from world.invariants import Invariants
    from world.scenes import StreetSpec, build
    from world.state import Entity, WorldState

    ok = True
    cam = Camera(focal_px=206.0, position=np.array([0.0, -1.7, 0.0]), pitch_deg=0.0)

    # A van-sized body straddling the lens: maximally visible, and unboundable.
    near = Entity("van", "vehicle", [0.0, 0.0, 1.2], size=[4.0, 2.0, 5.0])
    bb = screen_bbox(cam, near.pos + np.array([0.0, -1.0, 0.0]), 2.5, 336, 192)
    if bb is not None:
        print(f"     staging failed: wanted an unboundable sphere, screen_bbox gave {bb!r}")
        ok = False

    st = WorldState(seed=0)
    st.add(near)
    inv = Invariants()
    inv.visible_windows = {"van": (0, 0)}
    inv.check(st, cam, 0)
    inv.finish()
    if inv.violations:
        print(f"     a body wrapped around the lens scored invisible: "
              f"{inv.violations[0].detail}")
        ok = False

    # The OTHER direction, and the one that separates the correct predicate from the
    # plausible-looking wrong one. `if bb is not None:` also "flips the None case" but
    # changes nothing for the defect (the bad frames are all None either way) while making
    # () truthy -- so an entity 500 m behind the camera would start counting as on screen
    # and the visible invariant would stop firing for the case it was written for. A test
    # that only checks the first direction passes for that version too.
    behind = Entity("ghost", "vehicle", [0.0, 0.0, -500.0], size=[4.0, 2.0, 5.0])
    bb_behind = screen_bbox(cam, behind.pos + np.array([0.0, -1.0, 0.0]), 2.5, 336, 192)
    if bb_behind != ():
        print(f"     staging failed: wanted () for a body behind the camera, got {bb_behind!r}")
        ok = False
    st2 = WorldState(seed=0)
    st2.add(behind)
    inv2 = Invariants()
    inv2.visible_windows = {"ghost": (0, 0)}
    inv2.check(st2, cam, 0)
    inv2.finish()
    if not inv2.violations:
        print("     a body 500 m BEHIND the camera counted as visible")
        ok = False

    # The protagonist's sphere must cover the rig, not a ball at his knees.
    world = build(StreetSpec())
    c = world.get("courier")
    if float(max(c.size)) < 1.4:
        print(f"     courier bounding size is {list(c.size)}, a "
              f"{float(max(c.size)) / 2:.2f} m sphere -- smaller than the rig")
        ok = False

    print(f"[{'PASS' if ok else 'FAIL'}] visibility_counts_near_objects "
          f"(unboundable -> {'visible' if not inv.violations else 'INVISIBLE'}, "
          f"behind-camera -> {'not visible' if inv2.violations else 'VISIBLE'}, "
          f"courier sphere r={float(max(c.size)) / 2:.2f} m)")
    return ok


def t_checkdefs_scope():
    """checkdefs must mean "bound nowhere in THIS function", not "nowhere in the file".

    `_bound` collected bindings with `ast.walk`, which descends into nested bodies, and the
    `continue` after recording a nested function's name does not stop that descent. So the
    module scope ended up holding every local in the file and the real rule became "bound
    nowhere in the file". A name assigned in one function and read in a sibling -- exactly
    the fault the tool exists to catch, and the one that cost three GPU allocations --
    passed as clean while raising NameError at runtime.
    """
    import importlib.util
    import tempfile
    from pathlib import Path

    import take_actions as _ta

    root = Path(_ta.__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("checkdefs", root / "portb/checkdefs.py")
    cd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cd)

    ok = True
    cases = [
        # (source, must_flag, why)
        ("def alpha():\n    helper = 3\n    return helper\n\n\n"
         "def beta():\n    return helper\n", True, "sibling function local"),
        ("def alpha():\n    def inner():\n        v = 1\n        return v\n"
         "    return inner\n\n\ndef beta():\n    return v\n", True,
         "local of a nested function, read from a sibling"),
        ("W = 3\n\n\ndef alpha():\n    return W\n", False, "module global"),
        ("def alpha():\n    x = 1\n\n    def inner():\n        return x\n"
         "    return inner()\n", False, "closure over the enclosing function"),
        ("import os\n\n\ndef alpha():\n    return os.sep\n", False, "import"),
        ("def alpha(a, b=2):\n    return a + b\n", False, "parameters"),
        ("def alpha():\n    return len([1])\n", False, "builtin"),
        # The reverse of the scope bug: a nested function's own local must not be charged
        # to its parent. Pruning _bound without pruning _read would flag this.
        ("def alpha():\n    def inner():\n        q = 1\n        return q\n"
         "    return inner()\n", False, "local used only inside a nested function"),
        # Decorators and defaults ARE evaluated in the enclosing scope, so a missing one
        # there is still a real fault.
        ("def alpha():\n    def inner(x=missing):\n        return x\n"
         "    return inner\n", True, "default value read from the enclosing scope"),
        ("__all__ = ['alpha']\n\n\ndef alpha():\n    return __file__\n", False,
         "module dunder"),
        # Classes are scopes too, and have no parameter list. None of the first ten cases
        # had one, so the first version of the fix crashed on every file with a class.
        ("class C:\n    def m(self):\n        return self\n", False, "class method"),
        ("class C:\n    FIELD = 1\n\n    def m(self):\n        return FIELD\n", True,
         "class attribute read as a bare name inside a method"),
        ("import numpy as np\n\n\nclass C:\n    def m(self):\n        return np.pi\n",
         False, "import used inside a method"),
        # The mirror of the motivating fault: bound only in a NESTED function, read in the
        # PARENT. Also a runtime NameError, also missed before the fix. A fix that only
        # special-cases siblings closes one of the two.
        ("def alpha():\n    def inner():\n        z = 1\n        return z\n"
         "    inner()\n    return z\n", True, "nested local read in the parent"),
        ("def alpha():\n    return nowhere\n", True, "genuinely undefined"),
        # The false-positive traps: every one of these is legal Python and appears in this
        # repository, so a careless rewrite of the traversal breaks real files.
        ("def alpha(xs):\n    return [y * 2 for y in xs]\n", False,
         "comprehension variable"),
        ("def alpha():\n    f = lambda v: v + 1\n    return f(1)\n", False,
         "lambda parameter"),
        ("def alpha(xs):\n    return [w for v in xs if (w := v) > 0]\n", False,
         "walrus in a comprehension"),
        ("def alpha():\n    try:\n        pass\n    except ValueError as exc:\n"
         "        return exc\n", False, "except-as name"),
        ("G = 0\n\n\ndef setter():\n    global G\n    G = 1\n\n\n"
         "def alpha():\n    return G\n", False, "global assigned in another function"),
        ("def alpha():\n    import os\n    return os.sep\n", False,
         "import inside the function"),
        ("def alpha():\n    n = 0\n    n += 1\n    return n\n", False,
         "augmented assign of a local"),
    ]
    for src, must_flag, why in cases:
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "case.py"
            f.write_text(src, encoding="utf-8")
            import contextlib
            import io
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cd.check(str(f))
            flagged = rc != 0
            if flagged != must_flag:
                verb = "missed" if must_flag else "false-positived on"
                print(f"     checkdefs {verb} the {why} case")
                ok = False

    print(f"[{'PASS' if ok else 'FAIL'}] checkdefs_scope ({len(cases)} cases)")
    return ok


def t_fit_ground_refuses_z_depth():
    """Plane flatness determines focal length from euclidean range, and NEVER from z-depth.

    Back-projecting z-depth with a trial focal f' applies diag(f/f', f/f', 1). That is an
    affine map, and an affine map sends a plane to a plane, so flatness is exactly
    f-invariant: the sweep has no minimum and runs to the end of `focal_range`. On a
    known-answer scene built at f=206 it returned a confident 1200.0, the search ceiling.

    Normalising the ray divides by a norm that itself depends on f, which is nonlinear and
    does break the degeneracy -- so the euclidean branch is a real estimator and is kept.
    The defect was the z-depth branch answering instead of refusing, and z-depth is the
    only depth this project has: compiler/render.py writes "z along the optical axis, in
    metres -- never euclidean range".
    """
    import numpy as np

    from compiler.render import Camera, Plane, render
    from probe.geometry import fit_ground
    from probe.proxy import H, W

    ok = True
    TRUE_F, TRUE_H, TRUE_PITCH = 206.0, 1.67, -8.0
    cam = Camera(focal_px=TRUE_F, position=np.array([0.0, -TRUE_H, 0.0]),
                 pitch_deg=TRUE_PITCH)
    ground = [Plane(np.zeros(3), np.array([0.0, -1.0, 0.0]), class_id=4)]
    z, ids = render(cam, ground, width=W, height=H)
    mask = ids == 4

    # z-depth must be refused, not answered.
    refused = False
    try:
        got = fit_ground(z.astype(np.float64), mask, euclidean=False)
        refused = got is None
        if not refused:
            print(f"     fit_ground answered {got.focal_px:.1f} px from z-depth "
                  f"(truth {TRUE_F}); the sweep is monotone so that is the search ceiling")
    except ValueError:
        refused = True
    if not refused:
        ok = False

    # Euclidean range must still work, so this discriminates a real fix from deletion.
    dirs, fwd = cam.rays(W, H)
    cosang = dirs @ fwd
    euclid = np.where(z > 0, z / np.maximum(cosang, 1e-9), 0.0)
    fit = fit_ground(euclid.astype(np.float64), mask, euclidean=True)
    if fit is None:
        print("     euclidean fit returned None on a clean known-answer scene")
        ok = False
    elif abs(fit.focal_px - TRUE_F) > 10.0:
        print(f"     euclidean fit gave {fit.focal_px:.1f} px for a true {TRUE_F}")
        ok = False

    print(f"[{'PASS' if ok else 'FAIL'}] fit_ground_refuses_z_depth "
          f"(z-depth refused, euclidean -> "
          f"{fit.focal_px:.1f} px / {fit.height_m:.2f} m / {fit.pitch_deg:.1f} deg "
          f"for truth {TRUE_F} / {TRUE_H} / {TRUE_PITCH})")
    return ok


def t_validate_checks_sky_contract():
    """validate_conditions must check the one relation the writer treats as the contract.

    semantic == 1 means no hit, and a no-hit pixel has depth exactly 0. write_frame
    enforces it on the way out; the independent cold re-read never checked it, so a
    conditions directory produced by anything other than write_frame -- a hand edit, a
    different writer, a partial copy -- could carry sky with depth on it, or a hit pixel
    labelled sky, and still be reported clean.
    """
    import subprocess
    import sys as _sys
    import tempfile
    from pathlib import Path

    import numpy as np
    from PIL import Image

    import take_actions as _ta

    root = Path(_ta.__file__).resolve().parent
    ok = True

    def write_raw(d, sem, into):
        into.mkdir(parents=True, exist_ok=True)
        np.ascontiguousarray(d, dtype="<f4").tofile(into / "000000.depth.f32")
        Image.fromarray(np.ascontiguousarray(sem, dtype=np.uint8), "L").save(
            into / "000000.semantic_id.png")

    def validates(into):
        r = subprocess.run([_sys.executable, str(root / "validate_conditions.py"),
                            str(into), "--expect", "1"],
                           capture_output=True, text=True, cwd=str(root))
        return r.returncode == 0, r.stdout

    H, W = 192, 336
    good_d = np.full((H, W), 10.0, np.float32)
    good_s = np.full((H, W), 4, np.uint8)
    good_d[:20] = 0.0          # a sky band, depth exactly zero
    good_s[:20] = 1

    cases = [
        (good_d.copy(), good_s.copy(), True, "a well-formed frame"),
        None, None,
    ]
    bad1_d, bad1_s = good_d.copy(), good_s.copy()
    bad1_d[5, 5] = 42.0                    # sky pixel carrying a depth
    cases[1] = (bad1_d, bad1_s, False, "sky carrying depth")
    bad2_d, bad2_s = good_d.copy(), good_s.copy()
    bad2_d[100, 100] = 0.0                 # hit pixel at depth zero, not labelled sky
    cases[2] = (bad2_d, bad2_s, False, "a depth-0 pixel not labelled sky")

    for d, sem, want_pass, why in cases:
        with tempfile.TemporaryDirectory() as td:
            into = Path(td) / "conditions"
            write_raw(d, sem, into)
            passed, out = validates(into)
            if passed != want_pass:
                verb = "rejected" if want_pass else "accepted"
                print(f"     validate_conditions {verb} {why}")
                ok = False

    print(f"[{'PASS' if ok else 'FAIL'}] validate_checks_sky_contract "
          f"({len(cases)} frames)")
    return ok


def t_cache_matches_fresh_build():
    """A persistent PrimitiveCache must return exactly what a fresh build returns.

    This is the assertion that makes the cache safe rather than merely fast, and it is the
    one that was missing. `_signature` keyed on transform, gait, scale, size, alive and
    speed -- but NOT on `kind` or on `attrs["class_id"]`, both of which `_entity_shapes`
    branches on. That was dormant while the cache was constructed fresh at every call and
    therefore never cached; making it persist per worker turned it live.

    The failure mode is the quiet one: an entity whose ROLE changes without its TRANSFORM
    changing keeps its old semantic class forever. Geometry right, depth right, semantic
    channel stale. And it is exactly the case the override exists for -- a prop that
    becomes a conveyance when it is ridden, or released example 31's stag, which is a
    quadruped emitting class 8 because class is about role, not shape.
    """
    import numpy as np

    from compiler.render import Camera, render_frame
    from world import compile as wc
    from world.state import Entity, WorldState

    ok = True

    # 1. The staged case, at the pixel level: role changes, transform does not.
    cam = Camera(focal_px=206.0, position=np.array([0.0, -1.7, 0.0]), pitch_deg=0.0)
    st = WorldState(seed=0)
    st.add(Entity("thing", "prop", [0.0, -1.0, 6.0], size=[0.4, 0.4, 2.0],
                  attrs={"class_id": 0}))
    cache = wc.PrimitiveCache()
    render_frame(cam, cache.build(st))                 # frame 1, class 0
    st.get("thing").attrs["class_id"] = 10             # role change, no movement
    _z, cached_ids = render_frame(cam, cache.build(st))
    _z, fresh_ids = render_frame(cam, wc.PrimitiveCache().build(st))
    wrong = int((cached_ids != fresh_ids).sum())
    if wrong:
        print(f"     after a role change the cache rendered {wrong} px with the stale "
              f"class: cached {sorted(set(np.unique(cached_ids).tolist()))} vs fresh "
              f"{sorted(set(np.unique(fresh_ids).tolist()))}")
        ok = False

    # 2. The whole take, structurally. Primitive identity implies pixel identity and costs
    #    no ray tracing, so this can cover all 484 frames rather than a sample.
    plan, _, _ = simulate_take(check=False)
    persistent = wc.PrimitiveCache()
    bad_frames = []
    for k, sj, _cam in plan:
        world = WorldState.from_json(sj)
        a = [(type(q).__name__, q.class_id) for q in persistent.build(world)]
        b = [(type(q).__name__, q.class_id) for q in wc.PrimitiveCache().build(world)]
        if a != b:
            bad_frames.append(k)
    if bad_frames:
        print(f"     persistent cache differs from a fresh build on {len(bad_frames)} "
              f"frames, first at {bad_frames[0]}")
        ok = False

    print(f"[{'PASS' if ok else 'FAIL'}] cache_matches_fresh_build "
          f"(role change {wrong} px wrong, {len(plan)} take frames "
          f"{len(bad_frames)} differing)")
    return ok


def t_window_arithmetic():
    """The 124 / 90 / 34 windowing rule, asserted once instead of copied five times.

    FRAMES = 484 is currently justified by a comment. The constants live in four modules
    -- writer, caption, narrate, probe.proxy -- and nothing checked that they agree or
    that FRAMES actually tiles: setting it to 500 would silently produce a take the
    windowing cannot cover, with the last window running off the end of the plan.

    `writer.WINDOW_FRAMES` and `writer.STRIDE_FRAMES` lost their last consumer when the
    dead `window_starts` was deleted. Rather than drop them, they are asserted here, which
    is what they were for.
    """
    from compiler import caption as _cap
    from compiler import writer as _wr
    from probe import proxy as _px
    from world import narrate as _nr

    import take_actions as _ta

    ok = True
    W, S = _wr.WINDOW_FRAMES, _wr.STRIDE_FRAMES

    # One rule, one place. Every module that tiles frames must agree on it.
    for name, got in (("caption.WINDOW_FRAMES", _cap.WINDOW_FRAMES),
                      ("narrate.WINDOW_FRAMES", _nr.WINDOW_FRAMES),
                      ("proxy.WINDOW_FRAMES", _px.WINDOW_FRAMES)):
        if got != W:
            print(f"     {name} is {got}, writer says {W}")
            ok = False
    if _nr.STRIDE != S:
        print(f"     narrate.STRIDE is {_nr.STRIDE}, writer says {S}")
        ok = False

    # The take must tile exactly: 124 + 90*(k-1).
    want = W + S * (_ta.WINDOWS - 1)
    if _ta.FRAMES != want:
        print(f"     FRAMES is {_ta.FRAMES}; {_ta.WINDOWS} windows of {W} at stride {S} "
              f"tile {want}")
        ok = False

    # And the last window must end exactly on the last frame, not past it.
    last_start = S * (_ta.WINDOWS - 1)
    if last_start + W != _ta.FRAMES:
        print(f"     last window covers {last_start}..{last_start + W - 1}, take ends at "
              f"{_ta.FRAMES - 1}")
        ok = False

    # The Retake34 overlap is what the continuity prefix pins.
    if W - S != 34:
        print(f"     overlap is {W - S} frames, Retake34 expects 34")
        ok = False

    print(f"[{'PASS' if ok else 'FAIL'}] window_arithmetic "
          f"({_ta.WINDOWS} x {W} @ stride {S} = {want} frames, overlap {W - S})")
    return ok


def t_boom_holds_distance():
    """A follow rig must frame a subject walking at constant speed the way it frames a
    stationary one. Ours falls behind, and the fall-back IS the startup transient.

    The spring damped the camera's ABSOLUTE velocity: acc = w0^2 (want - pos) - 2 z w0 vel.
    For the camera to travel at the subject's speed v the acceleration must be zero, so
    w0^2 (want - pos) = 2 z w0 v and the rig settles a fixed 2 z v / w0 metres BEHIND the
    boom anchor. At the shipped constants (boom_tau 0.55, zeta 0.55) that is 1.15 m behind
    a 1.9 m/s walk -- which doubles ex30's 1.15 m boom and quarters the subject's apparent
    area. Frame 0 is the only frame framed correctly, because the camera is initialised at
    `want`; every later frame has sagged. That is why ex30 opens at 14.1% of frame and
    settles near 3%, and why its subject-y swings 0.67 -> 0.39 -> 0.63 on the way.

    Note on scope, because the obvious test is not enough on its own: a stationary subject
    shows no DISTANCE transient, since `want` never moves and pos == want, vel == 0 is
    already the fixed point -- the unfixed rig holds 2.208 m at frames 0, 30 and 123 to the
    last digit. It does NOT follow that a stationary subject is framed steadily. The aim yaw
    is seeded separately and wrongly and slides the subject about 57 px sideways over those
    same 30 frames; that is a different defect on a different axis, and it is covered by
    `t_aim_seeded_at_steady_state`. This test is about the boom's stand-off, so its
    discriminator has to be a subject that walks. The stationary case below is a regression
    guard for the boom only, and it passes both before and after.
    """
    from compiler.follow import FollowCamera

    ok = True
    dt = 1.0 / FPS
    D, SPEED = 1.15, 1.9

    def track(v, n=124):
        """Offset of the camera from the subject, in the subject's own frame."""
        cam = FollowCamera(distance=D, boom_tau=0.55, zeta=0.55, aim_tau=0.15, height=1.83)
        pos, out = np.zeros(3), []
        for _k in range(n):
            c = cam.update(pos, 0.0, dt)
            out.append(c.position - pos)
            pos = pos + np.array([0.0, 0.0, v * dt])
        return np.array(out)

    # 1. Regression guard: a stationary subject was never the broken case.
    st = track(0.0)
    still = float(np.abs(np.linalg.norm(st[30]) - np.linalg.norm(st[0])))
    if still > 1e-9:
        print(f"     stationary subject drifted {still:.4f} m between frames 0 and 30")
        ok = False

    # 2. The discriminator: constant velocity must be framed like stationary.
    mv = track(SPEED)
    n0, n30 = float(np.linalg.norm(mv[0])), float(np.linalg.norm(mv[30]))
    if abs(n30 - n0) > 0.02:
        print(f"     walking subject: stand-off {n0:.3f} m at frame 0 but {n30:.3f} m at "
              f"frame 30, a {n30 - n0:+.3f} m sag")
        ok = False

    # 3. And it must still hold once settled, not merely return somewhere.
    settled = np.linalg.norm(mv[64:], axis=1)
    err = float(np.max(np.abs(settled - n0)))
    if err > 0.02:
        print(f"     stand-off over the settled tail departs from frame 0 by up to "
              f"{err:.3f} m (min {settled.min():.3f}, max {settled.max():.3f})")
        ok = False

    # 4. The lag is speed-dependent, which is the signature of damping the wrong velocity:
    #    a rig that holds distance frames 1.0 m/s and 3.0 m/s identically.
    slow, fast = track(1.0), track(3.0)
    spread = float(abs(np.linalg.norm(slow[100]) - np.linalg.norm(fast[100])))
    if spread > 0.02:
        print(f"     stand-off depends on the subject's speed: {np.linalg.norm(slow[100]):.3f} m "
              f"at 1.0 m/s vs {np.linalg.norm(fast[100]):.3f} m at 3.0 m/s")
        ok = False

    print(f"[{'PASS' if ok else 'FAIL'}] boom_holds_distance "
          f"(walk sag {n30 - n0:+.3f} m, settled err {err:.3f} m, speed spread {spread:.3f} m)")
    return ok


def t_aim_seeded_at_steady_state():
    """Frame 0 must already be in follow position. A STATIONARY subject is the cleanest
    case: nothing about the scene changes, so every frame must be framed identically.

    It is not. `update()` seeds `self._yaw = heading_deg`, the subject's heading. But
    `shoulder_offset` puts the camera 0.45 m to one side of the boom axis, so the yaw that
    actually looks at the aim point differs from the heading by atan(shoulder / boom) --
    21.4 degrees at a 1.15 m boom. `aim_tau` then eases that error out over about 18
    frames, sliding the subject roughly 57 px across a 336 px frame while nothing in the
    world has moved.

    Measured at HEAD with a stationary subject and a 1.15 m boom:

        f     yaw      pitch      dist    bbox cols
        0   -5.1831  -27.0288   2.2077     46..185
        5  -17.3342  -27.0288   2.2077     90..222
       30  -21.3667  -27.0288   2.2077    103..234

    Distance and pitch are constant to four decimals from frame 0, so the boom spring is
    seeded correctly and never oscillates -- this is the aim axis alone, and it is a seeding
    error rather than a response to anything. The right fix is to seed the yaw from the aim
    direction. Shrinking `aim_tau` would hide it faster without removing it, and would
    stiffen tracking everywhere else.
    """
    from compiler.follow import FollowCamera
    from compiler.render import OUT_H, OUT_W, screen_bbox

    ok = True
    dt = 1.0 / FPS
    tgt = np.zeros(3)
    aim_pt = tgt + np.array([0.0, -0.9, 0.0])

    cam = FollowCamera(distance=1.15, boom_tau=0.55, zeta=0.55, aim_tau=0.15, height=1.83)
    poses = []
    for _k in range(31):
        c = cam.update(tgt, 0.0, dt)
        poses.append((screen_bbox(c, aim_pt, 0.45, OUT_W, OUT_H), c.yaw_deg, c.pitch_deg,
                      float(np.linalg.norm(c.position - tgt))))

    bb0, yaw0, pitch0, d0 = poses[0]
    bb30, yaw30, pitch30, d30 = poses[30]

    if bb0 != bb30:
        print(f"     stationary subject reframed between frame 0 and frame 30: "
              f"{bb0} -> {bb30}")
        ok = False
    if abs(yaw30 - yaw0) > 1e-6:
        print(f"     aim yaw slid {yaw0:.4f} -> {yaw30:.4f} deg with nothing moving")
        ok = False

    # Every frame, not just the two endpoints: a steady state has no interior excursion.
    moved = [k for k, (bb, _y, _p, _d) in enumerate(poses) if bb != bb0]
    if moved:
        print(f"     {len(moved)} of 31 frames differ from frame 0, first at {moved[0]}")
        ok = False

    # The boom axes were never the problem here; guard them so a fix that "works" by
    # disturbing distance or pitch is not mistaken for this one.
    if abs(d30 - d0) > 1e-9 or abs(pitch30 - pitch0) > 1e-9:
        print(f"     boom disturbed: dist {d0:.4f}->{d30:.4f}, pitch {pitch0:.4f}->{pitch30:.4f}")
        ok = False

    print(f"[{'PASS' if ok else 'FAIL'}] aim_seeded_at_steady_state "
          f"(yaw drift {yaw30 - yaw0:+.4f} deg, {len(moved)}/31 frames reframed)")
    return ok


def t_framed_holds_the_opening_layout():
    """"Preserving the opening layout" is a FOLLOW with the framing frozen, not a frozen lens.

    We routed the clause to `lock`, which freezes the camera body and the aim. Their data
    says the opposite. Subject median depth on class-8 pixels over the first 31 frames, the
    span the clause covers, is independent of how anything is framed:

        ex30   theirs 1.72-1.76 m (range 0.04)    ours 1.18-3.39 m (range 2.22)
        ex28   theirs 1.75-1.86 m (range 0.11)    ours 1.37-4.02 m (range 2.65)

    Their subject stays at a constant distance throughout. The layout holds BECAUSE the
    camera moves with the subject. And ex30's own prompt says so in as many words: "the
    camera follows at a nearly constant distance with a gentle walking bob", in the same
    sentence group as "preserving his large lower center-left placement". A lens that
    watches the subject walk from 1.18 m to 3.39 m is not preserving a layout; it is
    abandoning one.

    So the clause needs a third behaviour, distinct from both of ours:
      lock   - body and aim frozen. Right for a genuinely fixed lens.
      follow - body tracks the subject's heading, so the camera swings around it on a turn
               and the BACKGROUND rotates. Distance and screen position hold; layout does not.
      framed - follows, holding distance and screen position, but with the boom azimuth
               frozen at its opening value and secondary motion off. The subject stays put
               in frame and so does the background. That is what the clause describes.

    The routing lives in `world.shot.mode_for` so it is testable here rather than in the
    scene script, which is outside the repo.
    """
    from compiler.follow import FollowCamera
    from world import shot as shot_mod

    ok = True
    dt = 1.0 / FPS

    if not hasattr(shot_mod, "mode_for"):
        print("     world.shot has no mode_for: the hold clause is routed in the scene "
              "script, where it cannot be tested, and it routes to lock")
        print("[FAIL] framed_holds_the_opening_layout")
        return False

    sh = shot_mod.Shot(hold=True)
    routed = shot_mod.mode_for(sh)
    if routed != "framed":
        print(f"     hold clause routes to {routed!r}, not 'framed'")
        ok = False

    def run(mode, n=31, speed=1.9, turn=40.0):
        """Subject walks forward and turns, which is where the three modes separate."""
        cam = FollowCamera(distance=1.15, boom_tau=0.55, zeta=0.55, aim_tau=0.15,
                           height=1.83)
        pos, head, out = np.zeros(3), 0.0, []
        for k in range(n):
            cam.mode = mode
            c = cam.update(pos, head, dt)
            rel = c.position - pos
            # Azimuth of the camera around the subject: the background's orientation.
            out.append((float(np.linalg.norm(rel)),
                        float(np.degrees(np.arctan2(rel[0], rel[2]))),
                        c.yaw_deg))
            head = turn * (k + 1) / n
            th = np.radians(head)
            pos = pos + np.array([np.sin(th), 0.0, np.cos(th)]) * speed * dt
        arr = np.array(out)
        # Unwrap the azimuth: it crosses +/-180 here, and a raw arctan2 range reports a
        # 40 deg swing as 358.7, which would read as a much larger effect than it is.
        arr[:, 1] = np.degrees(np.unwrap(np.radians(arr[:, 1])))
        return arr

    got = run(routed)
    d_rng = float(got[:, 0].max() - got[:, 0].min())
    az_rng = float(got[:, 1].max() - got[:, 1].min())
    if d_rng > 0.05:
        print(f"     held span: stand-off ranges {got[:, 0].min():.2f}-{got[:, 0].max():.2f} m "
              f"({d_rng:.2f} m) while the subject walks")
        ok = False
    if az_rng > 1.0:
        print(f"     held span: boom azimuth swings {az_rng:.1f} deg as the subject turns, "
              f"so the background rotates and the layout is not preserved")
        ok = False

    # The separation that makes this a behaviour and not a rename.
    lock, follow = run("lock"), run("follow")
    lock_d = float(lock[:, 0].max() - lock[:, 0].min())
    follow_az = float(follow[:, 1].max() - follow[:, 1].min())
    if lock_d < 0.5:
        print(f"     lock no longer lets the subject walk away ({lock_d:.2f} m); the modes "
              f"are not distinct and this test proves nothing")
        ok = False
    if follow_az < 20.0:
        print(f"     follow no longer swings its azimuth on a turn ({follow_az:.1f} deg); "
              f"framed would be a synonym for follow")
        ok = False

    print(f"[{'PASS' if ok else 'FAIL'}] framed_holds_the_opening_layout "
          f"(framed: {d_rng:.3f} m / {az_rng:.1f} deg; lock lets {lock_d:.2f} m open; "
          f"follow swings {follow_az:.1f} deg)")
    return ok


def t_window_rejects_a_still_and_a_pinned_pitch():
    """Two degeneracies nothing checked, both of which shipped once the rig was corrected.

    Fixing the boom sag and the yaw seeding removed the only motion two of our six windows
    had. ex31 and ex36 then emitted ONE distinct frame repeated 124 times -- 5.17 seconds of
    conditioning with no temporal information in it at all. They passed validate_conditions,
    passed every invariant, and scored 0.000 on every axis of the steadiness table, which
    made them look like the best result on the page. The sag and the slide had been supplying
    apparent motion, and nothing noticed when it went away.

    The second is the twin of a saturation we already fixed. `t_flight_camera` used to report
    max pitch +35.0, which is the clamp, pinned there for 46 of 120 frames. The low side does
    the same thing: ex31 aims at the stag's feet (`aim_height=0.0`) and its pitch sits at
    -35.00 for all 124 frames. A clamp that saturates a whole window means the requested
    framing is unreachable, and that should be reported rather than silently clipped.

    Neither guard is about matching the released numbers. A still window and a pinned axis
    are wrong on their own terms.
    """
    import hashlib
    import subprocess
    import sys as _sys
    import tempfile
    from pathlib import Path as _P

    import numpy as np
    from PIL import Image

    import take_actions as _ta
    from world.invariants import pitch_saturation

    root = _P(_ta.__file__).resolve().parent
    ok = True
    H, W = 192, 336

    def write(into, frames):
        into.mkdir(parents=True, exist_ok=True)
        for k, (d, sm) in enumerate(frames):
            np.ascontiguousarray(d, dtype="<f4").tofile(into / f"{k:06d}.depth.f32")
            Image.fromarray(np.ascontiguousarray(sm, dtype=np.uint8), "L").save(
                into / f"{k:06d}.semantic_id.png")

    def validates(into, n):
        r = subprocess.run([_sys.executable, str(root / "validate_conditions.py"),
                            str(into), "--expect", str(n)],
                           capture_output=True, text=True, cwd=str(root))
        return r.returncode == 0, r.stdout + r.stderr

    base_d = np.full((H, W), 10.0, np.float32)
    base_s = np.full((H, W), 4, np.uint8)
    base_d[:20] = 0.0
    base_s[:20] = 1

    with tempfile.TemporaryDirectory() as td:
        # A still window: eight byte-identical frames.
        still = [(base_d.copy(), base_s.copy()) for _ in range(8)]
        d0 = _P(td) / "still"
        write(d0, still)
        good, out = validates(d0, 8)
        if good:
            print("     8 byte-identical frames validated clean; a still window is not a "
                  "conditioning signal")
            ok = False

        # The negative case: the same window with one frame differing must still pass, so
        # the guard cannot be satisfied by rejecting everything.
        moving = [(base_d.copy(), base_s.copy()) for _ in range(8)]
        moving[4][0][100, 100] = 11.0
        moving[4][1][100, 100] = 5
        d1 = _P(td) / "moving"
        write(d1, moving)
        good, out = validates(d1, 8)
        if not good:
            print(f"     a window with one differing frame was rejected: {out.strip()[:200]}")
            ok = False

    # Pitch saturation: all at the clamp is a failure, inside is fine, touching is fine.
    lo, hi = -35.0, 35.0
    cases = [
        ([lo] * 124, True, "pinned at pitch_min for the whole window"),
        ([hi] * 124, True, "pinned at pitch_max for the whole window"),
        ([-20.0] * 124, False, "constant but not at a clamp"),
        ([lo] * 60 + [-20.0] * 64, False, "saturated for half the window"),
        (list(np.linspace(lo, hi, 124)), False, "sweeping between the clamps"),
        # Both allowances, pinned by the real case that defeated the first version: a take
        # split across BOTH clamps, with a couple of frames in transit between them.
        ([lo] * 421 + [hi] * 61 + [0.0, 1.0], True, "482 of 484 across both clamps"),
        ([lo] * 400 + [0.0] * 84, False, "83% at a clamp, below the threshold"),
    ]
    for pitches, want_flag, why in cases:
        got = pitch_saturation(pitches, lo, hi)
        if bool(got) != want_flag:
            print(f"     pitch_saturation({why}) -> {got!r}, expected "
                  f"{'a report' if want_flag else 'nothing'}")
            ok = False

    # A guard nothing calls is not a guard. `pitch_saturation` was correct, tested, and
    # invoked only by this test: Invariants.check receives the camera every frame and never
    # recorded its pitch, finish() never called the helper, and the scene script built no
    # Invariants at all. The suite was green while three of six windows sat at the clamp
    # with nothing reporting it. That is the fourth time this shape has appeared here -- a
    # dead assertion, a floor that could not fail, a measurement of broken behaviour used as
    # a reference, and now a guard that is never reached -- so the wiring is asserted, not
    # just the helper.
    from compiler.render import Camera
    from world.invariants import Invariants
    from world.state import WorldState

    def run_take(pitch_series, clamp=(-35.0, 35.0)):
        inv = Invariants(pitch_clamp=clamp)
        st = WorldState(seed=0)
        for k, pd in enumerate(pitch_series):
            inv.check(st, Camera(focal_px=206.0, position=np.zeros(3), pitch_deg=pd), k)
        inv.finish()
        return [v for v in inv.violations if v.rule == "pitch"]

    if not run_take([-35.0] * 124):
        print("     a take pinned at the clamp for all 124 frames produced no violation: "
              "pitch_saturation exists but nothing calls it")
        ok = False
    if run_take([-20.0] * 124):
        print("     a take well inside the clamp was reported as saturated")
        ok = False

    print(f"[{'PASS' if ok else 'FAIL'}] window_rejects_a_still_and_a_pinned_pitch "
          f"({len(cases)} pitch cases, wiring checked)")
    return ok


def t_aim_screen_y():
    """Vertical placement must be a framing request, not a mis-aimed lens.

    `subject_low` parses for 14 of the 40 released prompts and was wired to nothing. The
    scene presets faked it through `aim_height` instead -- ex31 aims at 0.0 m, the stag's
    hooves, and ex28 at 0.55 m -- which is a different quantity: `aim_height` chooses the
    point ON THE SUBJECT the lens looks at. Dragging it to the feet tips the whole optical
    axis down until it hits the clamp, and then the framing stops responding at all. At
    875a1b9 ex28 and ex31 sat at pitch -35.00, the clamp, for every one of 124 frames, and
    that is also why they contained no sky: a camera pitched fully down has no horizon in
    it.

    The horizontal axis already has the right mechanism -- `aim_screen_x` rotates the
    optical axis and leaves the aim point alone. This is its vertical twin, and the
    placement words map the same way: "lower center" is the lower rule-of-thirds line, the
    same choice-from-the-vocabulary that `THIRD_LEFT` makes, not a fitted number.
    """
    import numpy as np

    from compiler.follow import FollowCamera
    from compiler.render import OUT_H
    from world.shot import LOWER_THIRD, Shot, aim_y_for

    ok = True

    def landed(fy, aim_height=0.9):
        """Where the SILHOUETTE lands -- rendered, not the projection of the aim point.

        The contract is a placement request: a request for 0.5 centres the figure. An
        earlier version of this test projected the aim point instead, which is exactly the
        quantity that was wrong: it passed while the same request moved 0.145 of frame
        height as `aim_height` went 0.875 -> 1.20, because it never looked at the subject.
        """
        from compiler.render import render_frame
        from world import compile as wc
        from world.state import Entity, WorldState

        cam = FollowCamera(distance=2.4, height=1.7, aim_height=aim_height,
                           aim_screen_y=fy)
        st = WorldState(seed=0)
        st.add(Entity("s", "human", [0.0, 0.0, 0.0], size=[0.55, 1.75, 0.40], speed=0.0))
        pitches = []
        for _k in range(60):
            c = cam.update(np.zeros(3), 0.0, 1.0 / FPS)
            pitches.append(c.pitch_deg)
        _z, ids = render_frame(c, wc.PrimitiveCache().build(st))
        r = np.nonzero(ids == 8)[0]
        return (float(r.mean() / ids.shape[0]) if r.size else float("nan")), pitches

    # The request must land on itself. Tolerance 0.04 rather than 0.02, because the
    # silhouette's projected centroid is not the projection of any single height: a 3-D
    # figure foreshortens, so the height that centres it moves a little with stand-off, and
    # a little more with pose. SILHOUETTE_CENTRE_FRAC is ONE measured constant covering
    # every distance and pose, and the residual it leaves runs 0.012 to 0.038 across the
    # cases here. The INVARIANCE assertion below is exact -- 0.000 -- and is held at 0.02,
    # so this looser absolute bound cannot hide the defect the fix is about.
    for fy in (0.5, LOWER_THIRD, 1.0 / 3.0, 0.6):
        got, _ = landed(fy)
        if abs(got - fy) > 0.04:
            print("     asked for %.3f, subject landed at %.3f" % (fy, got))
            ok = False

    # Invariance: the same request must land in the same place whatever the lens is aimed
    # at. This is the defect -- placement resolved against `aim_height`, so a request moved
    # 0.238 of frame height across the aim heights below while claiming to be a request.
    for fy in (0.5, LOWER_THIRD):
        got = [landed(fy, aim_height=ah)[0] for ah in (0.70, 0.875, 1.05, 1.20)]
        spread = max(got) - min(got)
        if spread > 0.02:
            print("     request %.3f moved %.3f across aim_height 0.70..1.20: %s"
                  % (fy, spread, " ".join("%.3f" % g for g in got)))
            ok = False

    # Negative: without the offset every request collapses to the boom's fixed point.
    lo, _ = landed(LOWER_THIRD)
    hi, _ = landed(1.0 / 3.0)
    if abs(lo - hi) < 0.15:
        print(f"     upper and lower requests land {abs(lo - hi):.3f} apart; the offset is "
              f"not reaching the axis")
        ok = False

    # The whole point: placing the subject low must NOT be bought by saturating the clamp.
    _, pitches = landed(LOWER_THIRD)
    from world.invariants import pitch_saturation
    sat = pitch_saturation(pitches, -35.0, 35.0)
    if sat:
        print(f"     placing the subject low saturated the axis: {sat}")
        ok = False

    # And the parse must reach it, or this is another mechanism wired to nothing.
    if aim_y_for(Shot(subject_low=True)) != LOWER_THIRD:
        print(f"     subject_low maps to {aim_y_for(Shot(subject_low=True))}, not LOWER_THIRD")
        ok = False
    if aim_y_for(Shot()) != 0.5:
        print(f"     a prompt that says nothing maps to {aim_y_for(Shot())}, not centre")
        ok = False

    print(f"[{'PASS' if ok else 'FAIL'}] aim_screen_y "
          f"(low {lo:.3f} vs upper-third {hi:.3f}, pitch unsaturated)")
    return ok


def t_articulation_covers_animals_and_planted_subjects():
    """Nothing in the rig animated an animal, or a subject standing still.

    Two frozen subjects came out of this, and only one of them was visible as a still
    window. `advance_gait` iterated `s.of_kind("human")`, so an animal never received a
    `gait_phase` at all, and the quadruped branch of `_entity_shapes` has no animation term
    -- the only trig in it is heading. ex31's stag walks 21.5 m at 4.2 m/s with rigid legs:
    its class-8 mask is byte-identical across all 124 frames, and it passes
    validate_conditions only because the BACKGROUND moves. The camel in the shipped take is
    the same rig and was never noticed, because it is never seen from a camera locked to it.

    The second is ex36, whose prompt plants the subject -- "remains planted ... without
    stepping, feet fixed" -- while still describing motion: "subtly turns her head and upper
    chest toward her right" and "her right hand swings a short distance beside her thigh".
    Gait is driven by distance covered, so at speed 0 there is no phase and no motion.

    Their own conditions say what articulation looks like at 336x192: the class-8 mask of
    their planted ex36 changes by a median of 47 px per frame, and their plainest walk,
    ex30, by 32. Ours was 0 for both ex31 and ex36.
    """
    import numpy as np

    from world import compile as wc
    from world.state import Entity, WorldState

    ok = True

    def stepped(kind, speed, n=8, **attrs):
        """Run advance_gait n times and return the primitive geometry each frame."""
        st = WorldState(seed=0)
        st.add(Entity("subj", kind, [0.0, 0.0, 6.0], size=[0.55, 1.75, 0.40],
                      speed=speed, attrs=dict(attrs)))
        geo = []
        for _k in range(n):
            wc.advance_gait(st)
            prims = wc.PrimitiveCache().build(st)
            vals = []
            for q in prims:
                for a in ("a", "b", "centre", "radius"):
                    v = getattr(q, a, None)
                    vals.append(np.atleast_1d(0.0 if v is None else v).ravel())
            geo.append(np.round(np.concatenate(vals), 5).tobytes())
        return st.get("subj"), geo

    # 1. An animal must receive a phase at all.
    e, geo = stepped("animal", 3.0)
    if e.attrs.get("gait_phase") is None:
        print("     a walking animal received no gait_phase")
        ok = False
    elif len(set(geo)) == 1:
        print(f"     a walking animal's geometry is identical across {len(geo)} frames: "
              f"the quadruped has no animation term")
        ok = False

    # 2. A planted subject must still articulate.
    for kind in ("human", "animal"):
        e, geo = stepped(kind, 0.0)
        if len(set(geo)) == 1:
            print(f"     a planted {kind} is geometrically identical across {len(geo)} "
                  f"frames; a live subject must never render a still")
            ok = False

    # 3. And idle must not leak into a walk: a moving subject's phase advances by exactly
    #    the distance it covered, which is what makes the shipped take reproducible.
    st = WorldState(seed=0)
    st.add(Entity("w", "human", [0.0, 0.0, 6.0], size=[0.55, 1.75, 0.40], speed=2.0))
    wc.advance_gait(st)
    want = 2.0 * st.dt / st.params.get("stride_m", 1.5)
    got = float(st.get("w").attrs.get("gait_phase", 0.0))
    if abs(got - want) > 1e-9:
        print(f"     a walking subject's gait_phase advanced {got:.6f}, expected {want:.6f}")
        ok = False
    if st.get("w").attrs.get("idle_phase"):
        print("     a walking subject was given an idle phase")
        ok = False

    # A window whose subject never moves is not caught by the still-frame guard, because
    # the BACKGROUND moves: ex31 passed validate_conditions with a class-8 mask that was
    # byte-identical across all 124 frames. So the condition directory gets its own check.
    import subprocess
    import sys as _sys
    import tempfile
    from pathlib import Path as _P

    from PIL import Image

    import take_actions as _ta

    root = _P(_ta.__file__).resolve().parent
    H, W = 192, 336

    def write(into, masks):
        into.mkdir(parents=True, exist_ok=True)
        for k, m in enumerate(masks):
            d = np.where(m, 5.0, 10.0).astype(np.float32)
            sm = np.where(m, 8, 4).astype(np.uint8)
            # Background scenery that moves every frame, so the window is never still and
            # the still-frame guard cannot fire. This is the ex31 shape exactly: a frozen
            # subject carried across a scene that is moving behind it.
            sm[150:170, (k * 17) % 300:(k * 17) % 300 + 18] = 5
            d[:20] = 0.0
            sm[:20] = 1
            np.ascontiguousarray(d, dtype="<f4").tofile(into / f"{k:06d}.depth.f32")
            Image.fromarray(np.ascontiguousarray(sm), "L").save(
                into / f"{k:06d}.semantic_id.png")

    def validates(into, n):
        r = subprocess.run([_sys.executable, str(root / "validate_conditions.py"),
                            str(into), "--expect", str(n)],
                           capture_output=True, text=True, cwd=str(root))
        return r.returncode == 0, r.stdout + r.stderr

    def blob(cx, cy, rw=26, rh=34):
        m = np.zeros((H, W), bool)
        m[max(0, cy - rh):cy + rh, max(0, cx - rw):cx + rw] = True
        return m

    with tempfile.TemporaryDirectory() as td:
        # Frozen subject, moving background -- the ex31 shape. Must be rejected.
        d0 = _P(td) / "frozen"
        write(d0, [blob(160, 110) for _k in range(8)])  # subject rigid, scene moving
        good, out = validates(d0, 8)
        if good:
            print("     a subject frozen for the whole window validated clean")
            ok = False

        # Articulating subject: must pass, so the guard is not simply rejecting everything.
        d1 = _P(td) / "moving"
        write(d1, [blob(160 + (k % 3), 110 + (k % 2)) for k in range(8)])
        good, out = validates(d1, 8)
        if not good:
            print(f"     an articulating subject was rejected: {out.strip()[-200:]}")
            ok = False

        # A subject too small to articulate visibly is exempt, not failed: ex34's aerial
        # sentinel is 0.44% of frame, and any honest motion there is sub-pixel after the
        # 2x2 pool. Theirs is 0.89% -- the gap is the subject's size, not its stillness.
        d2 = _P(td) / "tiny"
        write(d2, [blob(160, 110, rw=2, rh=2) for _k in range(8)])
        good, out = validates(d2, 8)
        if not good:
            print(f"     a subject below the visibility floor was failed rather than "
                  f"exempted: {out.strip()[-200:]}")
            ok = False

    print(f"[{'PASS' if ok else 'FAIL'}] articulation_covers_animals_and_planted_subjects")
    return ok


if __name__ == "__main__":
    import numpy as np  # actions reference np inside their exec namespace
    tests = [t_approach_and_stop, t_board_and_drive, t_mount_and_ride,
             t_fly_on_prop, t_descend_and_ballistic, t_flight_camera,
             t_caption_compiler, t_actor_captions,
             t_water_surface, t_class_vocabulary, t_camera_modes,
             t_aim_screen_x, t_shot_parse, t_take_is_reentrant,
             t_visibility_counts_near_objects, t_checkdefs_scope,
             t_fit_ground_refuses_z_depth, t_validate_checks_sky_contract,
             t_cache_matches_fresh_build, t_window_arithmetic,
             t_boom_holds_distance, t_aim_seeded_at_steady_state,
             t_framed_holds_the_opening_layout,
             t_window_rejects_a_still_and_a_pinned_pitch, t_aim_screen_y,
             t_articulation_covers_animals_and_planted_subjects]
    results = [t() for t in tests]
    n_ok = sum(bool(r) for r in results)
    print(f"\n{n_ok}/{len(results)} actions clean")
    sys.exit(0 if n_ok == len(results) else 1)

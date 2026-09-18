"""The 20-second multi-action take, built from tested actions and checked as it is generated.

Replaces the hand-written `demo_actions.py`. Every mechanism comes from `world/actions.py`,
which `test_actions.py` covers, and `world/invariants.py` runs on every frame -- so the three
defects that reached a GPU last time (a broom above its rider, an aircraft behind the camera,
a protagonist walking past the car he called) now fail here, in seconds, without rendering.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

from compiler import writer
from compiler.render import Camera, render_frame
from world import actions as A
from world import agent as ag
from world import compile as wc
from world import rules
from world import narrate
from world.invariants import Invariants, Mount
from world.scenes import street_take
from world.state import Entity, WorldState
from world.take_content import content as take_content

FPS = 24
FRAMES = 484          # 124 + 90*4 -> 5 windows -> 20.17 s
WINDOWS = 5

# When the camel is spawned. It has to be walking toward the protagonist well before the
# mount rule goes live at frame 250, because MOUNT will not start until the gap is inside
# `reach` -- and the camel is closing on a cart that is driving away from it. Spawning at
# 210 left it out of reach until 328, so the ride lasted 36 frames (1.5 s), too short to
# read. Swept, not guessed: 210->1.50 s, 190->2.33, 170->3.17, 150->4.00, 130->4.75, and
# 110/90 also 4.75. 130 is the first value on that plateau, where the ride is bounded by
# the mount rule going live at frame 250 rather than by how fast the camel can close, so
# it does not move if the camel's speed or the cart's departure changes.
CAMEL_SPAWN = int(os.environ.get("CWM_CAMEL_SPAWN", "130"))

# What the protagonist is doing, keyed to the attributes the actions in `script()` write
# while they run. Ordered latest-first because the flags latch: `mounted` stays set after he
# has left the camel, `boarded` after he has left the cart, so the first match must be the
# most recent phase. This is the ONLY description of the take's phases -- the captions are
# compiled from it, so there is nothing for them to drift against.
PHASES = [
    narrate.Phase(narrate.FLY, "courier", "fly_t", None, "broom"),
    narrate.Phase(narrate.RIDE_MOUNT, "camel", None, "mounted", "camel"),
    narrate.Phase(narrate.MOUNT, "camel", "mount_t", None, "camel"),
    narrate.Phase(narrate.DRIVE, "taxi", "drive_t", None, "rain-cart"),
    narrate.Phase(narrate.RIDE_VEHICLE, "taxi", None, "boarded", "rain-cart"),
    narrate.Phase(narrate.BOARD, "taxi", "board_t", None, "rain-cart"),
]


def script():
    """(frame, actions) plus the invariants the result must satisfy."""
    taxi = Entity(eid="taxi", kind="vehicle", position=[1.7, 0.0, 44.0],
                  heading_deg=180.0, speed=9.0, size=[2.4, 2.1, 5.0])
    camel = Entity(eid="camel", kind="animal", position=[-0.6, 0.0, -14.0],
                   heading_deg=0.0, speed=3.2, size=[1.05, 2.25, 3.1])
    gunship = Entity(eid="gunship", kind="vehicle", position=[-4.0, -22.0, 66.0],
                     heading_deg=180.0, speed=12.0, size=[15.0, 3.4, 11.0])
    ords = [Entity(eid=f"ord{i}", kind="prop", position=[x, -13.0, z],
                   heading_deg=0.0, size=[0.5, 0.5, 1.8], attrs={"alt0": 13.0, "t": t})
            for i, (x, z, t) in enumerate(((-3.0, 34.0, -0.3), (2.2, 27.0, -1.0),
                                           (-0.6, 21.0, -1.7)))]
    # He rides it, so it is a conveyance: class 10, the id their examples give a
    # "small rideable platform". It used to be 11, which appears in none of the released
    # examples (40 directories, 39 distinct geometries -- 32 and 33 are identical).
    broom = Entity(eid="broom", kind="prop", position=[0.0, 0.0, 0.0],
                   heading_deg=0.0, size=[0.10, 0.10, 1.9],
                   attrs={"class_id": 10})

    S = {
        # --- call a car, wait for it, get in, drive off -------------------------------
        6: [ag.Spawn(taxi),
            ag.AddRule("approach", A.build(A.APPROACH_AND_STOP, "approach", eid="taxi",
                                           sign=1.0, stop_at=5.5, v_min=1.6, v_max=9.0, gain=0.6)),
            ag.AddRule("halt", A.build(A.HALT_SUBJECT, "halt", eid="taxi",
                                       subject="courier", radius=9.0))],
        104: [ag.AddRule("board", A.build(A.BOARD_VEHICLE, "board", vehicle="taxi",
                                          rider="courier", board_s=1.3, seat_frac=0.55))],
        150: [ag.AddRule("drive", A.build(A.DRIVE_OFF, "drive", vehicle="taxi",
                                          rider="courier", v_max=6.5, accel=2.0, seat_frac=0.55))],
        # --- a camel comes the other way; leave the car for it -------------------------
        250: [ag.EnableRule("drive", False), ag.EnableRule("board", False),
              ag.EnableRule("halt", False),
              ag.AddRule("mount", A.build(A.MOUNT, "mount", mount="camel", rider="courier",
                                          climb_s=1.2, seat_frac=0.78, ride_speed=2.2,
                                          reach=3.0)),
              ag.AddRule("ride", A.build(A.RIDE, "ride", mount="camel", rider="courier",
                                         seat_frac=0.78, bob=0.045, bob_hz=6.2))],
        # --- an airstrike ahead of where the camel is carrying him ---------------------
        312: [ag.Spawn(gunship), *[ag.Spawn(o) for o in ords],
              ag.AddRule("descend", A.build(A.DESCEND_PAST, "descend", eid="gunship",
                                            dur_s=4.0, alt0=22.0, alt1=7.0)),
              ag.AddRule("ballistic", A.build(A.BALLISTIC, "ballistic",
                                              eids='("ord0", "ord1", "ord2")'))],
        # --- and finally a broom -------------------------------------------------------
        392: [ag.Spawn(broom), ag.EnableRule("ride", False), ag.EnableRule("mount", False),
              ag.EnableRule("camel_in", False),
              ag.AddRule("fly", A.build(A.FLY_ON_PROP, "fly", rider="courier", prop="broom",
                                        climb_s=3.4, alt_max=14.0, drift=2.2, offset=0.62))],
    }
    # Merged, NOT written as a dict key. A parameterised key silently overwrites any
    # literal entry on the same frame: with CAMEL_SPAWN=150 the camel replaced
    # `150: AddRule("drive")`, the drive rule was never registered, and the take quietly
    # became one where the protagonist boards a cart that then never moves. Python keeps
    # the last duplicate key and says nothing.
    S.setdefault(CAMEL_SPAWN, []).extend(
        [ag.Spawn(camel),
         ag.AddRule("camel_in", A.build(A.APPROACH_AND_STOP, "camel_in", eid="camel",
                                        sign=-1.0, stop_at=2.0, v_min=1.0,
                                        v_max=4.0, gain=0.5))])
    mounts = {"courier": Mount("taxi", seat_frac=0.55, seated_from=215)}
    return S, mounts


def simulate(check: bool = True):
    s, ctl, _ = street_take()
    active = list(rules.names())
    S, _ = script()
    inv = Invariants()
    # What must be seen, and when. `visible` covers elevation *and* bearing: an aircraft at the
    # right height but behind the camera passes every distance check and appears in nothing.
    inv.visible_windows = {"taxi": (80, 150), "camel": (300, 385),
                           "gunship": (330, 430), "ord0": (345, 420),
                           # The protagonist was never checked, which is how a take that
                           # loses him off the top of the frame for the last 2.5 s passed
                           # its own invariants. He is the shot; he is on screen or it is
                           # not a take.
                           "courier": (0, FRAMES - 1)}
    inv.visible_min = {"courier": 0.97}
    plan = []
    for k in range(FRAMES):
        if k in S:
            active = ag.apply(s, S[k], active)
        rules.step(s, active)
        wc.advance_gait(s)
        h = s.get("courier")
        cam = ctl.update(h.pos, h.heading_deg, 1.0 / FPS)
        if check:
            # The rider is on the car from 145, on the camel from 285, and flying from 400;
            # `flying` exempts bodies that are legitimately off the road.
            if k < 250:
                mounts = {"courier": Mount("taxi", 0.55, 215)}
            elif k < 392:
                mounts = {"courier": Mount("camel", 0.78, 350)}
            else:
                mounts = {}
            inv.check(s, cam, k, mounts=mounts,
                      flying=({"gunship", "ord0", "ord1", "ord2"} |
                              ({"courier", "broom"} if k >= 392 else set())))
        plan.append((k, s.to_json(), (cam.focal_px, tuple(float(x) for x in cam.position),
                                      cam.pitch_deg, cam.yaw_deg, cam.roll_deg)))
    if check:
        inv.finish()
        print(inv.report())
    return plan, s, inv


def _init_worker():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"


def _render_one(job):
    cond, k, state_json, cam_t = job
    st = WorldState.from_json(state_json)
    focal, pos, pitch, yaw, roll = cam_t
    cam = Camera(focal_px=focal, position=np.array(pos), pitch_deg=pitch,
                 yaw_deg=yaw, roll_deg=roll)
    z, ids = render_frame(cam, wc.PrimitiveCache().build(st))
    writer.write_frame(Path(cond), k, z, ids)
    return k


def main() -> int:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "cwm-data" / "take_actions")
    plan, s, inv = simulate(check=True)
    if inv.violations and os.environ.get("CWM_FORCE") != "1":
        print("refusing to render a take that violates its own invariants "
              "(set CWM_FORCE=1 to override)")
        return 1

    # Captions come out of the same simulation as the proxy, into the same directory, in the
    # same run. Previously the proxy was rendered here and the prompts were hand-written
    # somewhere else, so a take could be regenerated with captions describing the take it
    # used to be -- which is exactly what happened.
    written = narrate.build(plan, PHASES, take_content(), out_dir / "prompts", WINDOWS)
    print(f"compiled {len(written)} captions -> {out_dir / 'prompts'}")
    bad = narrate.check(out_dir / "prompts", plan, PHASES, WINDOWS,
                        actor_names=take_content()["actor_names"])
    if bad:
        print("captions disagree with the simulation that just produced them:")
        for b in bad:
            print("  -", b)
        return 1

    cond = out_dir / "conditions"
    cond.mkdir(parents=True, exist_ok=True)
    import multiprocessing as mp
    n = int(os.environ.get("CWM_WORKERS", max(mp.cpu_count() - 2, 1)))
    print(f"rendering {FRAMES} frames on {n} workers", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(n, initializer=_init_worker) as pool:
        done = 0
        # chunksize=1, not 4. Per-frame cost varies 18x -- the same 880 primitives cost 5.7 s
        # when the camera is among the vegetation and 0.3 s when it is looking down an open
        # road -- and the expensive frames come in runs. Handing out blocks of four drops a
        # whole run on one worker while the rest idle.
        for _ in pool.imap_unordered(_render_one, [(str(cond), *p) for p in plan], chunksize=1):
            done += 1
            if done % 120 == 0:
                print(f"  {done}/{FRAMES}", flush=True)
    s.save(out_dir / "final_state.json")
    print(f"render took {time.time()-t0:.0f}s -> {cond}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

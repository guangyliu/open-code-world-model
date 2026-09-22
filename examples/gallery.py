"""Six gallery scenes: our geometry, our actions, our camera, our compiled captions -- anchored
on a released first frame for its style.

    CWM_EXAMPLES=/path/to/examples40 python examples/gallery.py            # all six
    CWM_EXAMPLES=... python examples/gallery.py boulevard stream          # a subset
    ... --calib   # simulate + render frame 0 only, print composition vs the anchor's

Each scene writes out/gallery/<name>/{conditions,prompts,first_frame.png,meta.json}. The
first frame is copied from the released example whose style the scene borrows; frame 0 of
our proxy is composed to sit on it (subject centroid, stand-off, horizon), measured rather
than eyeballed, and everything after frame 0 is scripted here.

These are meant to be generated twice by MiniMax-H3: once in its native reference mode
(anchor + DUV video reference + caption, no LoRA) and once with the CWM LoRA, same seed, so
the only variable is whether the model was taught to read the proxy.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compiler import writer                                   # noqa: E402
from compiler.render import Camera, render_frame              # noqa: E402
from world import actions as A                                # noqa: E402
from world import agent as ag                                 # noqa: E402
from world import compile as wc                               # noqa: E402
from world import narrate                                     # noqa: E402
from world import rules                                       # noqa: E402
from world.gallery_content import CONTENT                     # noqa: E402
from world.invariants import Invariants, Mount                # noqa: E402
from world.scenes import StreetSpec, build, camera            # noqa: E402
from world.state import Entity, WorldState                    # noqa: E402

FPS, FRAMES = 24, 124
WINDOWS = int(os.environ.get("CWM_GALLERY_WINDOWS", "1"))
FRAMES = 124 + 90 * (WINDOWS - 1)          # the released 124 / 90-stride / Retake34 layout
EXAMPLES = Path(os.environ.get("CWM_EXAMPLES", "examples40"))
OUT = Path(os.environ.get("CWM_GALLERY_OUT", "out/gallery" if WINDOWS == 1 else f"out/gallery_{WINDOWS}w"))
JUMP_V0, G = 3.2, 9.81


def _extra(eid, kind, pos, size, heading=0.0, cls=None):
    e = Entity(eid, kind, list(pos), heading_deg=heading, size=list(size))
    if cls is not None:
        e.attrs["class_id"] = cls
    return e


# Non-protagonist people are class 0 in the released vocabulary (class 8 is the protagonist).
def _person(eid, pos, heading=180.0):
    return _extra(eid, "human", pos, [0.55, 1.75, 0.40], heading, cls=0)


SCENES = {
    # 1. Walk up the boulevard; a shuttle comes down it and halts; board; drive off.
    "boulevard": dict(
        anchor=40, target=dict(cx=0.417, cy=0.508, z8=2.96, h=0.844),
        spec=dict(has_road=True, road_x=0.0, road_half_width=7.0, n_vehicles=0, n_peds=0,
                  n_terraces=0, channel_w=0.0, tree_step=22.0, tree_lat=12.0, n_buildings=4,
                  bldg_lat=(12.0, 16.0), bldg_h=(5.0, 9.0), n_backdrop=6,
                  base_speed=1.5, cam_height=2.5, boom_len=2.75, aim_height=0.875,
                  aim_screen_x=0.417, aim_screen_y=0.56),
        hook="boulevard",
        scale=1.25,
        script=lambda: {
            # The shuttle has 5.2 s to arrive, be boarded and pull away, so it starts close:
            # 20 m out at 7.5 m/s with a stiff stop reaches its mark by ~2.3 s.
            0: [ag.Spawn(_extra("shuttle", "vehicle", [1.2, 0.0, 20.0], [2.0, 1.8, 4.4], 180.0)),
                ag.AddRule("approach", A.build(A.APPROACH_AND_STOP, "approach", eid="shuttle",
                                               sign=1.0, stop_at=5.0, v_min=2.5, v_max=7.5, gain=1.6)),
                ag.AddRule("halt", A.build(A.HALT_SUBJECT, "halt", eid="shuttle",
                                           subject="courier", radius=4.5))],
            58: [ag.AddRule("board", A.build(A.BOARD_VEHICLE, "board", vehicle="shuttle",
                                             rider="courier", board_s=1.0, seat_frac=0.55))],
            88: [ag.AddRule("drive", A.build(A.DRIVE_OFF, "drive", vehicle="shuttle",
                                             rider="courier", v_max=6.0, accel=3.0, seat_frac=0.55))],
        },
        phases=[narrate.Phase(narrate.DRIVE, "shuttle", "drive_t", None, "court shuttle"),
                narrate.Phase(narrate.RIDE_VEHICLE, "shuttle", None, "boarded", "court shuttle"),
                narrate.Phase(narrate.BOARD, "shuttle", "board_t", None, "court shuttle")],
        mounts=lambda k: {"courier": Mount("shuttle", 0.55, 86)} if k >= 58 else {},
        visible={"shuttle": (0, FRAMES - 1), "courier": (0, FRAMES - 1)},
    ),
    # 2. Run up the shoulder; a wagon comes down the causeway on the right and passes; the
    #    camera eases back and widens as it does.
    "causeway": dict(
        anchor=19, target=dict(cx=0.501, cy=0.571, z8=3.79, h=0.573),
        spec=dict(has_road=True, road_x=6.0, road_half_width=4.5, n_vehicles=0, n_peds=0,
                  n_terraces=0, channel_w=0.0, tree_step=14.0, tree_lat=8.0, n_buildings=2,
                  bldg_lat=(10.0, 14.0), bldg_h=(3.0, 6.0), n_backdrop=9,
                  backdrop_h=(60.0, 160.0), base_speed=3.0, cam_height=1.15, boom_len=3.75,
                  aim_height=0.875, aim_screen_x=0.501, aim_screen_y=0.54),
        hook="causeway",
        script=lambda: {
            0: [ag.Spawn(_extra("wagon", "vehicle", [4.8, 0.0, 14.0], [2.2, 2.3, 4.0], 180.0)),
                ag.Spawn(_extra("lotus", "vehicle", [7.5, 0.0, 42.0], [2.6, 2.6, 5.0], 0.0)),
                ag.AddRule("roll", A.build(A.APPROACH_AND_STOP, "roll", eid="wagon", sign=-1.0,
                                           stop_at=-60.0, v_min=3.0, v_max=3.0, gain=1.0)),
                ag.AddRule("lotus_on", A.build(A.APPROACH_AND_STOP, "lotus_on", eid="lotus", sign=1.0,
                                               stop_at=400.0, v_min=1.6, v_max=1.6, gain=1.0))],
        },
        camera=lambda ctl, k: setattr(ctl, "mode", "pull") if k >= 40 else setattr(ctl, "mode", "follow"),
        camera_setup=lambda ctl: setattr(ctl, "push_to", 6.5) or setattr(ctl, "push_rate", 1.2),
        phases=[],
        visible={"wagon": (0, 60), "courier": (0, FRAMES - 1)},
    ),
    # 3. Run along the river's edge, then break right away from the water; the follow rig
    #    swings round behind him.
    "stream": dict(
        anchor=4, target=dict(cx=0.436, cy=0.606, z8=3.87, h=0.438),
        spec=dict(has_road=False, n_vehicles=0, n_peds=0, n_terraces=0, channel_w=0.0,
                  tree_step=16.0, tree_lat=14.0, n_buildings=0, n_backdrop=10,
                  backdrop_dist=(200.0, 600.0), backdrop_h=(30.0, 90.0),
                  base_speed=4.5, cam_height=1.0, boom_len=3.9, aim_height=0.875,
                  aim_screen_x=0.436, aim_screen_y=0.53),
        hook="stream",
        scale=0.75,
        script=lambda: {
            56: [ag.SetGoal("courier", [[18.0, 0.0, 14.0 + 4.5 * 56 / 24 + 4.0], [60.0, 0.0, 60.0]])],
        },
        phases=[],
        visible={"courier": (0, FRAMES - 1)},
    ),
    # 4. Walk to the reliquary, stand while the camera orbits, walk on.
    "witchroad": dict(
        anchor=30, target=dict(cx=0.363, cy=0.584, z8=1.76, h=0.807),
        spec=dict(has_road=False, n_vehicles=0, n_peds=0, tree_step=17.0, tree_lat=6.5,
                  n_buildings=2, bldg_step=70.0, n_backdrop=9, n_terraces=0, channel_w=0.0,
                  base_speed=1.9, cam_height=1.45, boom_len=1.82, aim_height=0.875,
                  aim_screen_x=0.363, aim_screen_y=0.568),
        hook="reliquary",
        pause=(46, 96),
        camera=lambda ctl, k: setattr(ctl, "mode", "orbit") if 50 <= k < 96 else setattr(ctl, "mode", "follow"),
        camera_setup=lambda ctl: setattr(ctl, "orbit_rate", 22.0),
        script=lambda: {},
        phases=[],
        visible={"courier": (0, FRAMES - 1)},
    ),
    # 5. Run across the apron and leap the curb, twice.
    "frostcourt": dict(
        anchor=25, target=dict(cx=0.447, cy=0.586, z8=3.83, h=0.630),
        spec=dict(has_road=False, n_vehicles=0, n_peds=0, n_terraces=0, channel_w=0.0,
                  tree_step=30.0, tree_lat=16.0, n_buildings=4, bldg_lat=(9.0, 13.0),
                  bldg_h=(6.0, 12.0), bldg_step=40.0, n_backdrop=8,
                  base_speed=3.0, cam_height=1.3, boom_len=3.85, aim_height=0.875,
                  aim_screen_x=0.447, aim_screen_y=0.555),
        hook="frostcourt",
        jumps=(28, 82, 150, 214, 286, 352),      # only the first two fall inside one window
        script=lambda: {},
        phases=[narrate.Phase(narrate.JUMP, "courier", "jump_t", None, "")],
        visible={"courier": (0, FRAMES - 1)},
    ),
    # 6. Run up the road, stop at the drums while the camera pushes in, run on.
    "monastery": dict(
        anchor=13, target=dict(cx=0.575, cy=0.544, z8=4.03, h=0.536),
        spec=dict(has_road=True, road_x=0.0, road_half_width=6.0, n_vehicles=0, n_peds=0,
                  n_terraces=0, channel_w=0.0, tree_step=24.0, tree_lat=13.0, n_buildings=3,
                  bldg_lat=(11.0, 15.0), bldg_h=(5.0, 9.0), bldg_step=35.0, n_backdrop=9,
                  backdrop_h=(60.0, 150.0), base_speed=2.6, cam_height=1.2, boom_len=4.0,
                  aim_height=0.875, aim_screen_x=0.575, aim_screen_y=0.512),
        hook="monastery",
        pause=(58, 100),
        # push_to is the length both push and pull aim for, so the pull back out has to
        # re-aim it at the opening boom before switching mode.
        camera=lambda ctl, k: (setattr(ctl, "mode", "push") if 58 <= k < 100
                               else (setattr(ctl, "push_to", 4.0), setattr(ctl, "mode", "pull")) if 100 <= k < 124
                               else setattr(ctl, "mode", "follow")),
        camera_setup=lambda ctl: (setattr(ctl, "push_to", 2.2), setattr(ctl, "push_rate", 1.5)),
        script=lambda: {},
        phases=[],
        visible={"courier": (0, FRAMES - 1)},
    ),
}


def _ridge(s: WorldState, dist: float, height: tuple, n: int = 14, ahead_only: bool = True):
    """Hills at the horizon. The default street's backdrop sits at 320-900 m and leaves
    30-37% of the frame as sky; the anchors have 5-12%, because their worlds have terrain
    and buildings that fill the horizon. A ring of large boxes at `dist` does the same."""
    rng = np.random.default_rng(int(dist))
    for k in range(n):
        ang = (2 * np.pi * k / n) - np.pi
        if ahead_only and abs(ang) > 2.2:
            continue
        d = dist * float(rng.uniform(0.85, 1.25))
        s.add(_extra(f"ridge{int(dist)}_{k}", "scenery",
                     [d * float(np.sin(ang)), 0.0, d * float(np.cos(ang))],
                     [float(rng.uniform(40.0, 90.0)), float(rng.uniform(*height)), float(rng.uniform(40.0, 90.0))]))


def _hook(name: str, s: WorldState, spec: StreetSpec) -> None:
    if name == "boulevard":
        s.add(_extra("roadster", "vehicle", [-2.6, 0.0, 0.8], [1.9, 1.5, 4.6], 0.0))
        s.add(_extra("carriage", "vehicle", [-3.2, 0.0, 22.0], [2.4, 2.4, 6.0], 0.0))
        for k, z in enumerate((16.0, 24.0, 31.0)):
            s.add(_person(f"android{k}", [8.2, 0.0, z]))
        for k in range(4):                       # garden walls along the right sidewalk
            s.add(_extra(f"wall{k}", "scenery", [11.5, 0.0, 10.0 + 30.0 * k], [1.0, 3.0, 26.0]))
    elif name == "causeway":
        for side, x in ((-1, -9.5), (1, 11.0)):  # low parapets
            s.add(_extra(f"parapet{side:+d}", "scenery", [x, 0.0, 120.0], [0.6, 0.9, 240.0]))
        s.add(_extra("sanctuary", "scenery", [-14.0, 0.0, 40.0], [8.0, 7.0, 10.0]))
        _ridge(s, 110.0, (14.0, 30.0))
        _ridge(s, 190.0, (30.0, 60.0))
    elif name == "stream":
        s.add(Entity("river", "water", [-14.0, -0.02, 120.0], size=[26.0, 0.0, 300.0]))
        s.add(Entity("pond", "water", [-14.0, -0.02, 300.0], size=[30.0, 0.0, 120.0]))
        _ridge(s, 90.0, (10.0, 24.0))
        _ridge(s, 170.0, (24.0, 48.0))
    elif name == "reliquary":
        s.add(_extra("reliquary", "scenery", [1.4, 0.0, 66.0], [3.0, 4.2, 3.0]))
    elif name == "frostcourt":
        s.add(_extra("wall", "scenery", [-9.0, 0.0, 30.0], [6.0, 8.0, 26.0]))
        for k, (x, z) in enumerate(((4.5, 18.0), (0.8, 26.0))):     # gateway columns
            s.add(_extra(f"column{k}", "scenery", [x, 0.0, z], [0.9, 6.0, 0.9]))
        s.add(_extra("curb", "scenery", [0.0, 0.0, 9.0], [24.0, 0.22, 0.6]))
        s.add(_person("steward", [7.5, 0.0, 34.0]))
        for k, (x, z) in enumerate(((-16.0, 60.0), (12.0, 70.0), (-8.0, 95.0), (6.0, 110.0))):
            s.add(_extra(f"hallfar{k}", "scenery", [x, 0.0, z], [14.0, 12.0, 14.0]))
        _ridge(s, 120.0, (30.0, 60.0))
    elif name == "monastery":
        s.add(_extra("pole", "scenery", [-2.6, 0.0, 6.0], [0.5, 7.0, 0.5]))
        s.add(_extra("balustrade", "scenery", [7.5, 0.0, 60.0], [0.5, 1.1, 120.0]))
        s.add(_extra("drums", "scenery", [4.6, 0.0, 15.0], [1.6, 1.3, 6.0], 0.0))
        for k, (x, z) in enumerate(((5.4, 12.2), (6.3, 17.6), (6.6, 19.0), (6.9, 20.4))):
            s.add(_person(f"novice{k}", [x, 0.0, z]))
        for k in range(3):
            s.add(_extra(f"hall{k}", "scenery", [13.0, 0.0, 20.0 + 30.0 * k], [8.0, 7.0, 22.0]))
        _ridge(s, 130.0, (18.0, 36.0))
        _ridge(s, 220.0, (40.0, 75.0))


def build_scene(name: str):
    sc = SCENES[name]
    spec = replace(StreetSpec(seed=500 + sc["anchor"]), **sc["spec"])
    s = build(spec)
    h = s.get("courier")
    h.attrs["waypoints"] = [[0.0, 0.0, 400.0]]
    if "scale" in sc:
        h.attrs["scale"] = sc["scale"]
        h.size = [0.55 * sc["scale"], 1.75 * sc["scale"], 0.40 * sc["scale"]]
        ctl_h = spec.aim_height * sc["scale"]
        spec = replace(spec, aim_height=ctl_h)
    _hook(sc["hook"], s, spec)
    ctl = camera(spec)
    if "camera_setup" in sc:
        sc["camera_setup"](ctl)
    return s, ctl, spec, sc


def simulate(name: str, check: bool = True):
    s, ctl, spec, sc = build_scene(name)
    ruleset = rules.RuleSet()
    active = ruleset.names()
    S = sc["script"]()
    inv = Invariants(pitch_clamp=(ctl.pitch_min, ctl.pitch_max))
    inv.visible_windows = dict(sc.get("visible", {}))
    inv.visible_min = {"courier": 0.97}
    h = s.get("courier")
    vy, plan = 0.0, []
    pause = sc.get("pause")
    jumps = set(sc.get("jumps", ()))
    jumping = False       # NOT "y < 0": a rider seated in a vehicle is below y=0 too, and
                          # reading that as airborne applied gravity to him in his seat
    for k in range(FRAMES):
        if k in S:
            active = ag.apply(s, S[k], active, ruleset)
        if pause:
            if pause[0] <= k < pause[1]:
                h.speed = 0.0
            elif k == pause[1]:
                h.speed = spec.base_speed
        if "camera" in sc:
            sc["camera"](ctl, k)
        ruleset.step(s, active)
        # A standing jump: ballistic, +Y down. `jump_t` is present while airborne, which is
        # what the JUMP phase reads.
        if k in jumps and not jumping:
            vy, jumping = JUMP_V0, True
        if jumping:
            p = h.pos
            alt = max(0.0, -p[1] + vy * (1.0 / FPS))
            vy -= G / FPS
            p[1] = -alt
            h.pos = p
            if alt <= 0.0:
                vy, jumping = 0.0, False
                h.attrs.pop("jump_t", None)
            else:
                h.attrs["jump_t"] = float(h.attrs.get("jump_t", 0.0)) + 1.0 / FPS
        airborne = jumping
        wc.advance_gait(s)
        cam = ctl.update(h.pos, h.heading_deg, 1.0 / FPS)
        if check:
            mounts = sc["mounts"](k) if "mounts" in sc else {}
            inv.check(s, cam, k, mounts=mounts, flying=({"courier"} if airborne else set()))
        plan.append((k, s.to_json(), (cam.focal_px, tuple(float(x) for x in cam.position),
                                      cam.pitch_deg, cam.yaw_deg, cam.roll_deg)))
    if check:
        inv.finish()
        print(inv.report())
    return plan, s, inv, sc


def _init_worker():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"


def _render_one(job):
    cond, k, sj, cam_t = job
    st = WorldState.from_json(sj)
    focal, pos, pitch, yaw, roll = cam_t
    cam = Camera(focal_px=focal, position=np.array(pos), pitch_deg=pitch, yaw_deg=yaw, roll_deg=roll)
    z, ids = render_frame(cam, wc.PrimitiveCache().build(st))
    if cond is None:
        return z, ids
    writer.write_frame(Path(cond), k, z, ids)
    return k


def frame0_stats(z, ids):
    m = ids == 8
    ys, xs = np.nonzero(m)
    sky = float((ids == 1).mean())
    return dict(cx=round(float(xs.mean() / 336), 3) if m.any() else None,
                cy=round(float(ys.mean() / 192), 3) if m.any() else None,
                h=round(float((ys.max() - ys.min()) / 192), 3) if m.any() else None,
                z8=round(float(np.median(z[m & (z > 0)])), 2) if m.any() else None,
                sky=round(sky, 3),
                classes={int(c): round(float((ids == c).mean()), 3) for c in np.unique(ids)})


def calib(name: str):
    plan, s, inv, sc = simulate(name, check=False)
    k, sj, cam_t = plan[0]
    z, ids = _render_one((None, k, sj, cam_t))
    got = frame0_stats(z, ids)
    t = sc["target"]
    print(f"{name:11s} anchor ex{sc['anchor']}  target cx {t['cx']} cy {t['cy']} z8 {t['z8']} h {t['h']}")
    print(f"{'':11s}            got    cx {got['cx']} cy {got['cy']} z8 {got['z8']} h {got['h']}  sky {got['sky']}  {got['classes']}")
    return got


def run(name: str) -> int:
    out = OUT / name
    plan, s, inv, sc = simulate(name, check=True)
    if inv.violations:
        print(f"{name}: invariants violated; not rendering")
        return 1
    content = CONTENT[name]
    written = narrate.build(plan, sc["phases"], content, out / "prompts", WINDOWS)
    bad = narrate.check(out / "prompts", plan, sc["phases"], WINDOWS, actor_names=content["actor_names"])
    if bad:
        print(f"{name}: captions disagree with the simulation:")
        for b in bad:
            print("  -", b)
        return 1
    shutil.copy(EXAMPLES / str(sc["anchor"]) / "first_frame.png", out / "first_frame.png")
    cond = out / "conditions"
    cond.mkdir(parents=True, exist_ok=True)
    import multiprocessing as mp
    n = int(os.environ.get("CWM_WORKERS", 24))
    t0 = time.time()
    with mp.get_context("spawn").Pool(n, initializer=_init_worker) as pool:
        list(pool.imap_unordered(_render_one, [(str(cond), *r) for r in plan], chunksize=1))
    k, sj, cam_t = plan[0]
    z, ids = _render_one((None, k, sj, cam_t))
    meta = dict(scene=name, anchor_example=sc["anchor"], frames=FRAMES, windows=WINDOWS, target=sc["target"],
                frame0=frame0_stats(z, ids), events=[(e.kind, round(e.t, 2), e.subject) for e in s.events][:40],
                render_s=round(time.time() - t0, 1))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{name}: {FRAMES} frames ({WINDOWS} windows) in {meta['render_s']}s -> {out}   captions {len(written)}")
    return 0


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    names = args or list(SCENES)
    bad = [n for n in names if n not in SCENES]
    if bad:
        print(f"unknown scene(s) {bad}; valid: {', '.join(SCENES)}", file=sys.stderr)
        return 2
    if "--calib" in sys.argv:
        for n in names:
            calib(n)
        return 0
    rc = 0
    for n in names:
        rc |= run(n)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

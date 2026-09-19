"""Author our proxy for six of the authors' released examples, one window each.

Run from the repository root with the released 40 available:

    CWM_EXAMPLES=/path/to/examples40 python examples/repro.py        # all six
    CWM_EXAMPLES=/path/to/examples40 python examples/repro.py 30 36  # just these

This is the README centrepiece. For each example we render 124 frames of OUR conditions
against the scenario described by THEIR prompts/000.txt, so that a video made from our
proxy can be put beside one made from theirs with the anchor, caption and seed held fixed.

What this is not: a geometric round-trip of their conditions. We have never recovered their
scene; these are our own scenes authored to the same brief, and the honest claim is that our
proxy drives the same model to comparable behaviour.

Each preset records what the released prompt says and what we could and could not build, so
the divergences are written down next to the thing that causes them rather than discovered
later by a reader.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

# Runnable as `python examples/repro.py N` from any directory: the repo root is the
# script's parent, not its own directory, and only the latter is on sys.path by default.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compiler import writer
from compiler.render import Camera, render_frame
from world import compile as wc
from world import rules
from world.scenes import StreetSpec, build, camera
from world.invariants import pitch_saturation
from world.shot import CENTRE, LOWER_THIRD, mode_for, parse_shot
from world.state import Entity, WorldState

FPS = 24
FRAMES = 124
EXAMPLES = Path(os.environ.get("CWM_EXAMPLES", "examples40"))

# How long "the camera begins ... preserving the opening layout" holds for. The prompts say
# the camera BEGINS by holding and only ex28 says when it stops ("before following her"),
# so this is one judgement call applied uniformly rather than a per-example number: a
# quarter of the 124-frame window. It is not fitted to anything.
HOLD_FRAMES = 31

OUT = Path(os.environ.get("CWM_REPRO_OUT", "out/repro"))
# Suffix so a corrected re-render lands beside the old one instead of overwriting
# conditions a queued job may already be consuming.
SUFFIX = os.environ.get("CWM_REPRO_SUFFIX", "")

# Per example: the StreetSpec overrides, plus a builder hook that adds or removes entities.
# `note` is the divergence we already know about and will report.
PRESETS = {
    18: dict(
        why="water: class 2 is 52.8% of their window 0, the strongest water case in the 40",
        spec=dict(has_road=False, n_vehicles=0, n_peds=0, tree_step=26.0, tree_lat=13.0,
                  n_buildings=0, n_backdrop=7, n_terraces=0, channel_w=0.0,
                  # "begins behind Conan at roughly shoulder height, slightly toward his
                  # left side" -- a height and a lateral offset, no vertical placement.
                  base_speed=1.1, cam_height=1.55, boom_len=1.35,
                  aim_screen_y=CENTRE),
        hook="canal",
        note="their canal is left-offset with sloping terraces both sides and lamps; ours is "
             "a straight channel with plain banks and no lamps",
    ),
    30: dict(
        why="plain walk: no vehicle, mount or carried object; their own prompt says so",
        spec=dict(has_road=False, n_vehicles=0, n_peds=0, tree_step=17.0, tree_lat=6.5,
                  n_buildings=2, bldg_step=70.0, n_backdrop=9, n_terraces=0, channel_w=0.0,
                  # "He begins large in the lower center-left, seen from behind".
                  # aim_height is half his 1.75 m so aim_screen_y places HIM, not a point
                  # above his head; see FollowCamera.aim_screen_y.
                  base_speed=1.9, boom_len=1.15, aim_height=0.875,
                  aim_screen_y=LOWER_THIRD),
        hook="reliquary",
        note="their pale slab road reads as class 0, not class 4, so ours uses bare ground "
             "with no road slab; the reliquary ahead is a plain box",
    ),
    28: dict(
        why="vehicle at close range: a class-10 body fills 15.2% of their window 0, the "
            "largest vehicle presence in any window 0",
        spec=dict(has_road=True, road_x=0.0, road_half_width=4.2, n_vehicles=0, n_peds=0,
                  tree_step=19.0, tree_lat=7.5, n_buildings=0, n_backdrop=5,
                  n_terraces=0, channel_w=0.0, base_speed=2.6, cam_height=1.7,
                  # "She is seen from behind at lower center, with her body angled toward
                  # the upper right and her lower legs continuing beyond the bottom edge."
                  boom_len=1.15, aim_height=0.875, aim_screen_y=LOWER_THIRD),
        hook="bigcraft",
        note="their vehicle is a living silk-moth with wings and a crowned head; ours is the "
             "wireframe box vehicle, so the silhouette is a hull, not a moth",
    ),
    31: dict(
        why="animal subject: the protagonist IS a white stag, and their proxy still classes "
            "it 8 -- class 8 encodes the protagonist, not the species",
        spec=dict(has_road=False, n_vehicles=0, n_peds=0, tree_step=24.0, tree_lat=11.0,
                  n_buildings=0, n_backdrop=8, n_terraces=0, channel_w=0.0,
                  # "the large white stag occupies the lower center foreground, seen
                  # directly from behind", and the camera sentence: "keeping the animal low
                  # and central". aim_height=0.0 aimed at its hooves to fake this, which
                  # put it HIGH in frame and pinned the pitch at -35.00 for all 124 frames.
                  base_speed=4.2, cam_height=1.6, boom_len=2.2, aim_height=0.875,
                  aim_screen_y=LOWER_THIRD),
        hook="stag",
        note="our quadruped rig is the camel built for the other take; proportions are "
             "camel-like, not stag-like, and there are no antlers",
    ),
    34: dict(
        why="flight: the only window-0 timeline where the protagonist is genuinely airborne "
            "and banking with the ground receding below",
        spec=dict(has_road=True, road_x=0.0, road_half_width=5.0, n_vehicles=6, n_peds=0,
                  tree_step=7.0, tree_lat=9.0, n_buildings=4, bldg_step=55.0,
                  n_backdrop=11, n_terraces=1, terrace_lat0=26.0, terrace_w=22.0,
                  channel_w=0.0, base_speed=0.0, cam_height=1.8,
                  # "a steep downward-looking aerial view": the only preset still on
                  # aim_height rather than aim_screen_y, kept as an honest residual.
                  rise_gain=1.25, aim_height=-1.4, boom_len=4.2, pull_back=0.7),
        hook="flight",
        note="THE WEAKEST OF THE SIX, and kept deliberately. Theirs is a high oblique "
             "vista: terrain filling the frame, a lake and winding roads, the subject "
             "small and incidental. Ours is a subject-following boom, so it reads as a "
             "corridor at low altitude. Widening the pitch clamp to -70 deg was tried "
             "and discarded: it tilted far enough down but lost the subject out of "
             "frame and took sky from 18% to 0% against their 35%. A vista shot needs a "
             "camera that frames terrain rather than a subject; this rig has none.",
    ),
    36: dict(
        why="wide road with distant machines: road 46.9% and sky 26.0%, a much wider "
            "composition than our own take, and previously run so it is comparable",
        spec=dict(has_road=True, road_x=0.0, road_half_width=6.5, n_vehicles=3, n_peds=0,
                  tree_step=16.0, tree_lat=10.5, n_buildings=2, bldg_step=90.0,
                  n_backdrop=9, n_terraces=0, channel_w=0.0, base_speed=0.0,
                  # "The trailing camera begins near shoulder height behind the solitary
                  # female route surveyor": shoulder height on a 1.75 m figure, not head
                  # height. Her prompt states no vertical placement -- "She stands left of
                  # center" -- so the framing is centred. aim_height=1.62 was the backwards
                  # knob pushed the other way, and is what put her at y=0.79.
                  cam_height=1.45, boom_len=1.45, aim_screen_y=CENTRE),
        hook="surveyor",
        note="their three machines differ in size and type; ours are three identical box "
             "vehicles at the described distances",
    ),
}


def _hook(name: str, s: WorldState, spec: StreetSpec) -> None:
    """Scenario-specific geometry, on top of the shared street build."""
    if name == "canal":
        # Water under and around the walker: the canal he wades along, plus raised banks.
        s.add(Entity("canal", "water", [0.0, -0.05, spec.path_len / 2.0],
                     size=[9.0, 0.0, spec.path_len]))
        for side in (-1.0, 1.0):
            for k in range(7):
                s.add(Entity(f"bank{side:+.0f}{k}", "scenery",
                             [side * 7.4, 0.0, 20.0 + 60.0 * k],
                             size=[3.2, 2.2, 58.0]))
    elif name == "reliquary":
        s.add(Entity("reliquary", "scenery", [1.4, 0.0, 66.0], size=[3.0, 4.2, 3.0]))
    elif name == "bigcraft":
        # The large body lying diagonally across the lane just ahead of her.
        s.add(Entity("craft", "vehicle", [1.5, 0.0, 7.2], heading_deg=52.0,
                     size=[5.2, 2.6, 7.0]))
        s.add(Entity("cart2", "vehicle", [4.0, 0.0, 34.0], heading_deg=20.0,
                     size=[1.9, 1.7, 2.8]))
    elif name == "stag":
        # Protagonist is a quadruped but still the protagonist, so class 8 like theirs.
        s.entities.pop("courier", None)
        s.add(Entity("courier", "animal", [0.0, 0.0, 0.0], heading_deg=0.0,
                     speed=spec.base_speed, size=[1.15, 1.75, 2.5],
                     attrs={"class_id": 8,
                            "waypoints": [list(w) for w in spec.waypoints]}))
    elif name == "flight":
        h = s.get("courier")
        h.attrs["waypoints"] = [[0.0, 0.0, 400.0]]
    elif name == "surveyor":
        for k, (x, z, sz) in enumerate(((3.9, 21.0, [2.3, 1.9, 4.6]),
                                        (-1.1, 47.0, [2.0, 1.7, 3.4]),
                                        (1.0, 96.0, [2.2, 1.8, 4.0]))):
            s.add(Entity(f"machine{k}", "vehicle", [x, 0.0, z], heading_deg=90.0, size=sz))


def released_shot(ex: int):
    """The shot the authors' own window-0 prompt describes."""
    return parse_shot((EXAMPLES / str(ex) / "prompts" / "000.txt").read_text(
        encoding="utf-8"))


def build_example(ex: int):
    p = PRESETS[ex]
    spec = replace(StreetSpec(seed=100 + ex), **p["spec"])
    # The shot clause is an input, not decoration. Reading it is the whole point: without
    # this every subject lands at frame centre because that is the follow boom's fixed
    # point, and our six sat at 0.49-0.50 against a released spread of 0.350-0.499.
    sh = released_shot(ex)
    if sh.subject_x is not None:
        spec = replace(spec, aim_screen_x=sh.subject_x)
    s = build(spec)
    ctl = camera(spec)
    _hook(p["hook"], s, spec)
    return s, ctl, spec, p, sh


def simulate(ex: int):
    s, ctl, spec, p, sh = build_example(ex)
    active = list(rules.names())
    plan = []
    for k in range(FRAMES):
        # "preserving the exact initial layout", "holding the opening layout". This used
        # to lock the lens off, which their own conditions contradict: their subject holds
        # a constant distance right through the span (ex30 1.72-1.76 m, ex28 1.75-1.86 m)
        # and ex30's prompt says "the camera follows at a nearly constant distance". The
        # clause is a framed follow; `world.shot.mode_for` owns the reading.
        ctl.mode = mode_for(sh) if (sh.hold and k < HOLD_FRAMES) else "follow"
        rules.step(s, active)
        wc.advance_gait(s)
        h = s.get("courier")
        if p["hook"] == "flight":
            # A simple climb: the flight action is written for the other take's script, so
            # the altitude is driven directly here.
            alt = 3.0 + 16.0 * (k / FRAMES) ** 0.8
            pos = h.pos
            h.pos = [pos[0], -alt, pos[2] + 3.4 / FPS]
            h.speed = 0.0
        cam = ctl.update(h.pos, h.heading_deg, 1.0 / FPS)
        plan.append((k, s.to_json(), (cam.focal_px, tuple(float(x) for x in cam.position),
                                      cam.pitch_deg, cam.yaw_deg, cam.roll_deg)))
    return plan


def _init_worker():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"


def _render_one(job):
    cond, k, sj, cam_t = job
    st = WorldState.from_json(sj)
    focal, pos, pitch, yaw, roll = cam_t
    cam = Camera(focal_px=focal, position=np.array(pos), pitch_deg=pitch,
                 yaw_deg=yaw, roll_deg=roll)
    z, ids = render_frame(cam, wc.PrimitiveCache().build(st))
    writer.write_frame(Path(cond), k, z, ids)
    return k


def main() -> int:
    import multiprocessing as mp

    args = sys.argv[1:]
    valid = ", ".join(str(k) for k in sorted(PRESETS))
    if any(a in ("-h", "--help") for a in args) or not all(a.lstrip("-").isdigit() for a in args):
        print(f"usage: {Path(sys.argv[0]).name} [N ...]   regenerate released examples N (default: all)\n"
              f"       valid N: {valid}\n"
              f"       reads CWM_EXAMPLES (default examples40), writes CWM_REPRO_OUT (default out/repro)")
        return 0 if any(a in ("-h", "--help") for a in args) else 2
    want = [int(a) for a in args] or sorted(PRESETS)
    # Validate before touching the filesystem, so a typo does not leave an empty out/repro/exNN.
    unknown = [ex for ex in want if ex not in PRESETS]
    if unknown:
        print(f"no preset for example(s) {unknown}; valid N: {valid}", file=sys.stderr)
        return 2
    for ex in want:
        cond = OUT / f"ex{ex:02d}{SUFFIX}" / "conditions"
        cond.mkdir(parents=True, exist_ok=True)
        plan = simulate(ex)
        # The pitch column of the plan is the pitch that actually renders. A clamp that
        # saturates the whole window means the requested framing was unreachable and what
        # ships is the limit; report it rather than letting it pass silently.
        pitches = [fr[2][2] for fr in plan]
        lo, hi = StreetSpec().pitch_min, StreetSpec().pitch_max
        sat = pitch_saturation(pitches, lo, hi)
        n_lo = sum(1 for x in pitches if abs(x - lo) <= 1e-6)
        n_hi = sum(1 for x in pitches if abs(x - hi) <= 1e-6)
        if sat:
            print(f"  ex{ex:02d}: {sat}")
        elif n_lo or n_hi:
            print(f"  ex{ex:02d}: pitch at a clamp for {n_lo} low / {n_hi} high of "
                  f"{len(pitches)} frames")
        t0 = time.time()
        with mp.get_context("spawn").Pool(24, initializer=_init_worker) as pool:
            list(pool.imap_unordered(_render_one, [(str(cond), *r) for r in plan],
                                     chunksize=1))
        print(f"ex{ex:02d}: {FRAMES} frames in {time.time()-t0:.0f}s -> {cond}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

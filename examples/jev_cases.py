"""Four small scenes, one pedestrian under test each, rules vs Jev, camera on THAT pedestrian.

    python examples/jev_cases.py            # all four, both policies, side-by-side videos
    python examples/jev_cases.py wall car   # a subset

The earlier comparison put the camera on the courier and the interesting pedestrian was
20 m away behind a wall; this one follows the pedestrian being tested and prints his last
decision, its confidence and the state numbers that drove it on the frame.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image, ImageDraw                    # noqa: E402

from compiler.render import render                  # noqa: E402
from world import compile as wc                     # noqa: E402
from world import rules                             # noqa: E402
from world.scenes import StreetSpec, build, camera  # noqa: E402
from world.state import Entity                      # noqa: E402
import moves                                        # noqa: E402
from jev_npc import ego_state, apply, jev_batch, bearing_to, PED_SPEED   # noqa: E402

FPS, DT = 24, 1.0 / 24
DECIDE_EVERY = 12
OUT = Path("/tmp/claude-502/jev_cases")


def base():
    spec = StreetSpec(seed=21)
    s = build(spec)
    h = s.get("courier")
    h.speed = 0.0
    h.attrs.pop("waypoints", None)
    for k in ("ped0", "ped1", "ped2"):
        e = s.get(k)
        e.alive = False                      # scenes add their own people
    return s, spec


def ped(s, eid, pos, goal):
    e = s.add(Entity(eid, "human", list(pos), size=[0.55, 1.75, 0.40]))
    e.heading_deg = float(np.degrees(np.arctan2(goal[0] - pos[0], goal[2] - pos[2])))
    e.speed = PED_SPEED
    e.attrs["waypoints"] = [list(goal)]
    e.attrs["wp_radius"] = 1.0
    return e


def scene_wall():
    """A 6 m wall across the sidewalk, goal behind it; open at both ends."""
    s, spec = base()
    s.add(Entity("wall", "scenery", [2.5, 0.0, 18.0], size=[6.0, 3.0, 0.8]))
    a = ped(s, "walker", (2.0, 0.0, 25.0), (2.0, 0.0, -20.0))      # 7 m to the wall, 5 s
    return s, spec, a, "wall across the path, goal straight behind it"


def scene_car():
    """A parked car straddling the line to the goal: pass on either side."""
    s, spec = base()
    s.add(Entity("parked", "vehicle", [2.2, 0.0, 18.0], heading_deg=0.0, size=[1.9, 1.5, 4.4]))
    a = ped(s, "walker", (2.0, 0.0, 26.0), (2.0, 0.0, -20.0))
    return s, spec, a, "parked car on the line to the goal"


def scene_head_on():
    """Another pedestrian coming straight at him on the same line."""
    s, spec = base()
    a = ped(s, "walker", (2.0, 0.0, 26.0), (2.0, 0.0, -20.0))
    ped(s, "oncoming", (2.0, 0.0, 10.0), (2.0, 0.0, 50.0))         # 16 m apart, meet at ~6 s
    return s, spec, a, "oncoming pedestrian on the same line"


def scene_cross():
    """The courier walks up the street; the walker's path crosses his 10 m ahead."""
    s, spec = base()
    h = s.get("courier")
    h.position = [0.0, 0.0, 0.0]
    h.heading_deg, h.speed = 0.0, 2.0
    a = ped(s, "walker", (8.0, 0.0, 12.0), (-8.0, 0.0, 12.0))     # both reach the crossing near 6 s
    return s, spec, a, "crossing the protagonist's path"


SCENES = {"wall": scene_wall, "car": scene_car, "head_on": scene_head_on, "cross": scene_cross}


def run(name, policy, frames=288):
    s, spec, a, title = SCENES[name]()
    peds = [e for e in s.of_kind("human") if e.eid != "courier"]
    goals = {e.eid: e.attrs["waypoints"][0] for e in peds}
    ctl = camera(spec)
    ctl.distance = 3.2
    cache = wc.PrimitiveCache()
    imgs, log, lat = [], [], []
    last = {e.eid: None for e in peds}
    d0 = bearing_to(a, goals[a.eid])[1]
    stopped = blocked = 0
    for k in range(frames):
        if policy == "jev" and k % DECIDE_EVERY == 0:
            states = [ego_state(s, e, goals[e.eid]) for e in peds]
            res = jev_batch(states)
            for e, st, (ans, dt) in zip(peds, states, res):
                an = ans["answers"]
                apply(e, an["maneuver"]["choice"], goals[e.eid], s)
                e.attrs["_pos_at_decision"] = e.pos.copy()
                last[e.eid] = dict(f=k, choice=an["maneuver"]["choice"], conf=an["maneuver"]["confidence"],
                                   vetoed=e.attrs.get("vetoed", ""), blocked=an["path_blocked"]["noul"],
                                   yld=an["yield"]["noul"], st=st)
                log.append({"f": k, "eid": e.eid, **{x: y for x, y in last[e.eid].items() if x != "st"}, "state": st})
                lat.append(dt)
        active = ["blocking", "locomotion"] + (["waypoints", "separation"] if policy == "rules" else [])
        rules.step(s, active)
        wc.advance_gait(s)
        for e in peds:
            if "_pos_at_decision" in e.attrs:
                e.attrs["walked_since"] = float(np.linalg.norm(e.pos - e.attrs["_pos_at_decision"]))
        if a.speed == 0.0: stopped += 1
        if a.attrs.get("blocked_by"): blocked += 1
        cam = ctl.update(a.pos, a.heading_deg, DT)
        z, ids = render(cam, cache.build(s), width=336, height=192)
        img = np.concatenate([moves.up(moves.LUT[ids]), moves.up(moves.depth_img(z))], 0)
        imgs.append(overlay(img, policy, k, a, goals[a.eid], last[a.eid]))
    d1 = bearing_to(a, goals[a.eid])[1]
    summ = dict(scene=title, policy=policy, progress_m=round(d0 - d1, 2), frames_stopped=stopped,
                frames_blocked=blocked, reached=bool(d1 < 1.5))
    if policy == "jev":
        summ.update(calls=len(log), latency_median_s=round(float(np.median(lat)), 2),
                    choices={c: sum(1 for r in log if r["choice"] == c and r["eid"] == a.eid) for c in set(r["choice"] for r in log)},
                    vetoed=sum(1 for r in log if r["vetoed"]))
    return imgs, summ, log


def overlay(img, policy, k, a, goal, last):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    f = moves._font(bold=True, size=15); m = moves._font(mono=True, size=12)
    rel, dist = bearing_to(a, goal)
    d.rectangle([0, 384 * 2 - 92, 672, 384 * 2], fill=(24, 25, 32))
    d.text((8, 384 * 2 - 88), f"{policy.upper()}   t={k/FPS:4.1f}s   goal {dist:4.1f} m at {rel:+4.0f}°   speed {a.speed:.1f}"
           + ("   BLOCKED" if a.attrs.get("blocked_by") else ""), font=f, fill=(255, 214, 64) if policy == "jev" else (140, 220, 160))
    if policy == "rules":
        d.text((8, 384 * 2 - 66), "steer toward the goal; separation pushes people apart; blocking stops him at solids", font=m, fill=(200, 202, 214))
    elif last:
        c = last["st"]["clear_metres_before_solid"]
        d.text((8, 384 * 2 - 66), f"decision @ {last['f']/FPS:4.1f}s: {last['choice']}  conf {last['conf']:.2f}"
               + (f"   VETOED ({last['vetoed']} into a solid)" if last["vetoed"] else "")
               + f"   path_blocked {last['blocked']:.2f}  yield {last['yld']:.2f}", font=m, fill=(255, 214, 64))
        d.text((8, 384 * 2 - 48), f"clear m: L90 {c['hard_left_90']:>5} L60 {c['far_left_60']:>5} L30 {c['left_30']:>5} ahead {c['ahead']:>5} R30 {c['right_30']:>5} R60 {c['far_right_60']:>5} R90 {c['hard_right_90']:>5}", font=m, fill=(200, 202, 214))
        st = last["st"]
        d.text((8, 384 * 2 - 30), f"toward goal clear {st['clear_metres_if_I_turn_toward_goal_now']:>5} m   nearest person {st['nearest_other_pedestrian']['distance_m']} m at {st['nearest_other_pedestrian']['bearing_deg']:+.0f}°   protagonist {st['protagonist']['distance_m']} m at {st['protagonist']['bearing_deg']:+.0f}°   last: {st['last_decision'][:28]}", font=m, fill=(200, 202, 214))
    return np.array(im)


def main():
    names = sys.argv[1:] or list(SCENES)
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for n in names:
        R, sr, _ = run(n, "rules")
        J, sj, log = run(n, "jev")
        gap = np.full((R[0].shape[0], 12, 3), (24, 25, 32), np.uint8)
        moves.write([np.concatenate([r, gap, j], 1) for r, j in zip(R, J)], OUT / f"{n}.mp4")
        Image.fromarray(np.concatenate([R[120], gap, J[120]], 1)).save(OUT / f"{n}_mid.png")
        json.dump(log, open(OUT / f"{n}_jev_log.json", "w"), indent=1)
        results[n] = {"rules": sr, "jev": sj}
        print(n, json.dumps(sr), "\n   ", json.dumps(sj))
    json.dump(results, open(OUT / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()

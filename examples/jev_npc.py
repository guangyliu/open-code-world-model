"""Test 3: Jev as the pedestrians' 'System One' -- tactical judgement at 2 Hz, code keeps the veto.

A street with a wall across it, parked and moving cars, three pedestrians and the courier
walking a fixed route. Each pedestrian either runs the built-in rules (waypoints +
separation + blocking) or asks Jev every 0.5 s what to do from an egocentric text state:
clear distance in five forward sectors, bearing/distance to its goal, to the courier and
to the nearest other pedestrian, whether it was just blocked. Jev returns one manoeuvre
and two judgements; the `blocking` rule still runs every frame and overrides anything that
would walk into a solid. Both policies get the same seed and the same world.

    python examples/jev_npc.py --policy rules   # baseline
    python examples/jev_npc.py --policy jev     # Jev at 2 Hz, decisions logged
    python examples/jev_npc.py --replay out/jev_npc/jev/decisions.json

Measured per pedestrian over the take: distance made good toward the goal, frames spent
stopped, blocked events, near-misses with the courier, and for Jev: calls, latency, tokens,
what it chose and with what confidence.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.request

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compiler.render import render                    # noqa: E402
from world import compile as wc                       # noqa: E402
from world import rules                               # noqa: E402
from world.rules import _solid_footprints             # noqa: E402
from world.scenes import StreetSpec, build, camera    # noqa: E402
from world.state import Entity                        # noqa: E402
import moves                                          # noqa: E402

FPS, DT = 24, 1.0 / 24
FRAMES = 480                       # 20 s
DECIDE_EVERY = 12                  # 2 Hz
URL = "https://api.typesafe.ai/v1/systemone"
KEY_PATH = os.path.expanduser("~/.config/typesafe/key")
PED_SPEED = 1.3
GOALS = {"ped0": [2.0, 0.0, -30.0], "ped1": [-2.0, 0.0, 80.0], "ped2": [6.0, 0.0, -30.0]}

Q = {
    "maneuver": {"type": "choice",
                 # Literal reading, third revision. v2 said 'prefer keep_going when the way ahead is
                 # clear' and the pedestrian, having sidestepped a wall, walked away from his goal at
                 # 90 degrees for 15 s with a clear path ahead -- exactly as told.
                 "instructions": "You are this pedestrian walking to a goal. Choose the next half-second of movement. Rules in order: (1) if the goal is more than 30 degrees off your facing AND clear_metres_if_I_turn_toward_goal_now is at least 2, choose turn_toward_goal; (2) if the goal is roughly ahead (within 30 degrees) and the way ahead is clear for 2 m or more, keep_going; (3) if the way ahead is blocked within 2 m, move toward the sector with the most clear metres (step or veer) rather than stopping; (4) stop only when every sector is blocked or a person is about to cross your path.",
                 "criteria": {
                     "keep_going": "continue straight at walking pace",
                     "veer_left": "turn about 40 degrees to the left and keep walking",
                     "veer_right": "turn about 40 degrees to the right and keep walking",
                     "step_left": "turn 90 degrees left and walk (e.g. along a wall face to get round it)",
                     "step_right": "turn 90 degrees right and walk (e.g. along a wall face to get round it)",
                     "turn_toward_goal": "turn to face the goal bearing and walk",
                     "stop": "stand still this half-second (someone is about to cross your path, or you are boxed in)",
                     "turn_around": "reverse direction",
                     "other": "none of these fits"}},
    "path_blocked": {"type": "noul", "instructions": "Continuing straight for the next 2 metres would hit something solid (wall or car)."},
    "yield": {"type": "noul", "instructions": "Another person is within 2.5 metres and closing on my path, so I should give way."},
}


# ----------------------------------------------------------------------------- world ----
def make_world(seed=11):
    spec = StreetSpec(seed=seed)
    s = build(spec)
    h = s.get("courier")
    h.attrs["waypoints"] = [[0.0, 0.0, 30.0], [3.0, 0.0, 60.0], [0.0, 0.0, 120.0]]
    h.speed = 1.6
    # A wall across the sidewalk side at z=20, leaving the road open: pedestrians heading
    # -z on x in [0, 5] must go round it.
    s.add(Entity("wall", "scenery", [3.0, 0.0, 20.0], heading_deg=0.0, size=[7.0, 3.0, 0.8]))
    for k, (eid, g) in enumerate(GOALS.items()):
        e = s.get(eid)
        e.position = [[2.0, 0.0, 44.0], [-2.0, 0.0, -6.0], [5.5, 0.0, 40.0]][k]
        e.heading_deg = 180.0 if g[2] < e.position[2] else 0.0
        e.speed = PED_SPEED
        e.attrs["waypoints"] = [g]
        e.attrs["wp_radius"] = 1.0
    return s, spec


def clear_dist(s, e, bearing_deg, max_d=12.0, step=0.25):
    """Metres of free travel along a bearing before a solid footprint, up to max_d.

    A point test from the first step out, NOT a body-width test from x=0: a pedestrian the
    blocking rule has stopped 5 cm from a wall face is, by the body-width test, inside the
    wall's expanded box in every direction, and the state then read 'all five sectors
    0.25 m' -- boxed in -- to which 'stop' is the right answer. That was the state, not
    the model, refusing to let him walk along the face. (The drone project's lesson: the
    state has to contain the answer.)
    """
    th = np.radians(bearing_deg)
    d = np.array([np.sin(th), 0.0, np.cos(th)])
    right = np.array([d[2], 0.0, -d[0]])
    hw = float(e.size[0]) / 2.0
    fps = _solid_footprints(s, e)
    x = step
    while x < max_d:
        # The same two-corner test the blocking rule applies, so what the state calls
        # clear is what the rule will let him walk. A centre-line point test called a
        # corner-hugging path clear and the rule then refused every step of it.
        for corner in (e.pos + d * x + right * hw, e.pos + d * x - right * hw):
            for o, c, ax, az, hx, hz in fps:
                q = corner - c
                if abs(float(q @ ax)) < hx and abs(float(q @ az)) < hz:
                    return round(x - step, 2)
        x += step
    return max_d


def bearing_to(e, target):
    v = np.asarray(target, float) - e.pos
    want = np.degrees(np.arctan2(v[0], v[2]))
    rel = ((want - e.heading_deg + 180.0) % 360.0) - 180.0
    return float(rel), float(np.hypot(v[0], v[2]))


def ego_state(s, e, goal):
    rel_goal, d_goal = bearing_to(e, goal)
    h = s.get("courier")
    rel_c, d_c = bearing_to(e, h.pos)
    others = [(bearing_to(e, o.pos), o.eid) for o in s.of_kind("human") if o.eid not in (e.eid, "courier")]
    if others:
        (rel_o, d_o), o_id = min(others, key=lambda t: t[0][1])
    else:
        (rel_o, d_o), o_id = (0.0, 99.0), "none"
    sectors = {name: clear_dist(s, e, e.heading_deg + off)
               for name, off in (("hard_left_90", -90), ("far_left_60", -60), ("left_30", -30), ("ahead", 0),
                                 ("right_30", 30), ("far_right_60", 60), ("hard_right_90", 90), ("behind", 180))}
    # The joined fact, stated outright: Jev does not do the arithmetic of looking up the
    # goal bearing in the sector table, so 'turn_toward_goal' with the goal 0.5 m behind
    # a wall face read as a fine idea and the pedestrian oscillated step / turn / step.
    goal_clear = clear_dist(s, e, e.heading_deg + rel_goal)
    return {
        "who": f"pedestrian {e.eid} on a city sidewalk",
        "last_decision": e.attrs.get("last_decision", "none yet"),
        "metres_walked_since_last_decision": round(float(e.attrs.get("walked_since", 0.0)), 2),
        "clear_metres_if_I_turn_toward_goal_now": goal_clear,
        "walking_speed_m_s": round(float(e.speed), 2),
        "was_blocked_by_solid_last_step": bool(e.attrs.get("blocked_by")),
        "clear_metres_before_solid": sectors,
        "goal": {"bearing_deg_relative_to_facing": round(rel_goal, 0), "distance_m": round(d_goal, 1),
                 "note": "negative bearing = to my left, positive = to my right, 0 = straight ahead"},
        "protagonist": {"bearing_deg": round(rel_c, 0), "distance_m": round(d_c, 1), "their_speed_m_s": round(float(h.speed), 2)},
        "nearest_other_pedestrian": {"id": o_id, "bearing_deg": round(rel_o, 0), "distance_m": round(d_o, 1)},
    }


def jev_batch(states):
    key = open(KEY_PATH).read().strip()

    def one(st):
        body = {"state": st, "model": "jev-latest", "questions": Q}
        req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        t = time.time()
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    return json.load(r), time.time() - t
            except Exception as err:
                last = err
                time.sleep(1.0 * (attempt + 1))
        raise last

    with ThreadPoolExecutor(len(states)) as ex:
        return list(ex.map(one, states))


def apply(e, choice, goal, s=None):
    """Code keeps the veto: a turn into something solid within 1.5 m is refused and the
    pedestrian keeps his current heading instead. The refusal is recorded on the entity."""
    e.attrs["vetoed"] = ""
    if s is not None and choice in ("veer_left", "veer_right", "step_left", "step_right", "turn_toward_goal"):
        off = {"veer_left": -40.0, "veer_right": 40.0, "step_left": -90.0, "step_right": 90.0,
               "turn_toward_goal": bearing_to(e, goal)[0]}[choice]
        if clear_dist(s, e, e.heading_deg + off) < 1.5:
            e.attrs["vetoed"] = choice
            e.speed = PED_SPEED
            e.attrs["last_decision"] = f"{choice} (vetoed by code: that way is blocked within 1.5 m)"
            return
    e.attrs["last_decision"] = choice
    if choice == "veer_left":
        e.heading_deg -= 40.0; e.speed = PED_SPEED
    elif choice == "veer_right":
        e.heading_deg += 40.0; e.speed = PED_SPEED
    elif choice == "step_left":
        e.heading_deg -= 90.0; e.speed = PED_SPEED
    elif choice == "step_right":
        e.heading_deg += 90.0; e.speed = PED_SPEED
    elif choice == "turn_toward_goal":
        rel, _ = bearing_to(e, goal); e.heading_deg += rel; e.speed = PED_SPEED
    elif choice == "stop":
        e.speed = 0.0
    elif choice == "turn_around":
        e.heading_deg += 180.0; e.speed = PED_SPEED
    else:                                   # keep_going / other
        e.speed = PED_SPEED


# ------------------------------------------------------------------------------ run ----
def run(policy, out, replay=None, video=True):
    s, spec = make_world()
    peds = [s.get(k) for k in GOALS]
    ctl = camera(spec)
    h = s.get("courier")
    cache = wc.PrimitiveCache()
    frames, log, lat, toks = [], [], [], 0
    prog0 = {e.eid: bearing_to(e, GOALS[e.eid])[1] for e in peds}
    stopped = {e.eid: 0 for e in peds}
    blocked = {e.eid: 0 for e in peds}
    near = {e.eid: 0 for e in peds}
    seen_block = {e.eid: "" for e in peds}
    active_rules = ["blocking", "locomotion"] + (["waypoints", "separation"] if policy == "rules" else [])
    for k in range(FRAMES):
        if policy == "jev" and k % DECIDE_EVERY == 0:
            states = [ego_state(s, e, GOALS[e.eid]) for e in peds]
            if replay is not None:
                answers = [replay[str(k)][e.eid] for e in peds]
            else:
                res = jev_batch(states)
                answers = [r[0]["answers"] for r in res]
                lat += [r[1] for r in res]
                toks += sum(r[0]["usage"]["input_tokens"] for r in res)
            for e, st, a in zip(peds, states, answers):
                apply(e, a["maneuver"]["choice"], GOALS[e.eid], s)
                e.attrs["_pos_at_decision"] = e.pos.copy()
                log.append({"f": k, "eid": e.eid, "state": st, "choice": a["maneuver"]["choice"], "vetoed": e.attrs.get("vetoed", ""),
                            "conf": round(a["maneuver"]["confidence"], 2),
                            "path_blocked": round(a["path_blocked"]["noul"], 2), "yield": round(a["yield"]["noul"], 2),
                            "probabilities": a["maneuver"]["probabilities"]})
        # The courier walks his route under the built-in rules regardless of policy.
        rules.step(s, active_rules if policy == "rules" else ["blocking", "locomotion"])
        if policy == "jev":
            # keep the courier on his waypoints without steering the pedestrians
            wps = h.attrs["waypoints"]; i = int(h.attrs.get("wp_index", 0)) % len(wps)
            rel, d = bearing_to(h, wps[i])
            if d < 1.5: h.attrs["wp_index"] = i + 1
            else: h.heading_deg += float(np.clip(rel, -90 * DT, 90 * DT))
        wc.advance_gait(s)
        for e in peds:
            if "_pos_at_decision" in e.attrs:
                e.attrs["walked_since"] = float(np.linalg.norm(e.pos - e.attrs["_pos_at_decision"]))
            if e.speed == 0.0: stopped[e.eid] += 1
            b = e.attrs.get("blocked_by", "")
            if b and b != seen_block[e.eid]: blocked[e.eid] += 1
            seen_block[e.eid] = b
            if bearing_to(e, h.pos)[1] < 0.9: near[e.eid] += 1
        if video:
            cam = ctl.update(h.pos, h.heading_deg, DT)
            z, ids = render(cam, cache.build(s), width=336, height=192)
            frames.append(np.concatenate([moves.up(moves.LUT[ids]), moves.up(moves.depth_img(z))], 0))
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for e in peds:
        d_now = bearing_to(e, GOALS[e.eid])[1]
        summary[e.eid] = dict(progress_m=round(prog0[e.eid] - d_now, 2), start_dist=round(prog0[e.eid], 1),
                              end_dist=round(d_now, 1), frames_stopped=stopped[e.eid], blocked_events=blocked[e.eid],
                              frames_within_0p9m_of_courier=near[e.eid])
    if policy == "jev":
        from collections import Counter
        summary["jev"] = dict(calls=len(log) if replay is None else 0, latency_median_s=round(float(np.median(lat)), 3) if lat else None,
                              latency_max_s=round(float(max(lat)), 3) if lat else None, input_tokens=toks,
                              choices=dict(Counter(r["choice"] for r in log)), vetoed=sum(1 for r in log if r["vetoed"]),
                              mean_conf=round(float(np.mean([r["conf"] for r in log])), 2))
        by_frame = {}
        for r in log:
            by_frame.setdefault(str(r["f"]), {})[r["eid"]] = {"maneuver": {"choice": r["choice"], "confidence": r["conf"], "probabilities": r["probabilities"]},
                                                              "path_blocked": {"noul": r["path_blocked"]}, "yield": {"noul": r["yield"]}}
        json.dump(by_frame, open(out / "decisions.json", "w"))
        json.dump(log, open(out / "log.json", "w"), indent=1)
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    if video:
        moves.write(frames, out / f"{policy}.mp4")
    return summary, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["rules", "jev"], default="rules")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--replay", type=Path, default=None)
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    policy = "jev" if a.replay else a.policy
    out = a.out or Path("out/jev_npc") / policy
    rep = json.load(open(a.replay)) if a.replay else None
    summary, log = run(policy, out, rep, video=not a.no_video)
    print(json.dumps(summary, indent=1))
    if policy == "jev":
        for r in log[:12]:
            print(r["f"], r["eid"], r["choice"], r["conf"], "blocked?", r["path_blocked"], "yield?", r["yield"],
                  "| ahead", r["state"]["clear_metres_before_solid"]["ahead"], "goal", r["state"]["goal"]["bearing_deg_relative_to_facing"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

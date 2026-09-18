"""Compile a take's captions from the same simulation that compiles its proxy.

The captions used to be hand-written. They drifted: they never once described the
protagonist boarding or driving the car -- which the script spends frames 104-250 doing --
and they brought the camel in a whole window early. The model followed the caption and
ignored the DUV, so at proxy frame 200 (proxy: driving the cart) it rendered him reaching up
to climb a camel, exactly as the caption said. That does not merely spoil one take: while
the caption can contradict the proxy, no adherence measurement can separate "the model
ignored the proxy" from "the model obeyed a caption that disagreed with it".

So the caption is compiled, never authored. Structure comes from compiler/caption.py, which
was fitted to the released prompts; the *content* of what the protagonist does comes from
`derive_track`, which reads the very attributes `world/actions.py` writes while it runs.
There is no second description of the script anywhere to drift against.

Two rules this module exists to keep:

  * **State, not nouns.** A keyword check passes the broken captions easily -- "car" appears
    11 times in the old window 0 and 8 times in window 1, as scenery the protagonist stands
    next to. Every phrase here therefore carries a SIGNATURE naming what the PROTAGONIST is
    doing, and `classify` reads a sentence back to a state through those signatures alone.
  * **The gate re-reads the file.** `check` parses the prompts on disk rather than trusting
    anything in memory, so a stale prompt directory -- the actual failure -- fails loudly.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from compiler.caption import Caption, WINDOW_SECONDS

WINDOW_FRAMES = 124
STRIDE = 90
SEGMENTS = 7            # released prompts use 6-10, mode 7

# Preambles, byte-for-byte from the released prompts. Window 0 pins the anchor frame;
# every continuation window pins the 34-frame rolling history Retake34 hands it.
PREAMBLE_W0 = (
    "Use the provided RGB image as the authoritative first frame at 0.00 seconds. Continue "
    "directly from this exact visual state without restarting or redesigning the "
    "protagonist, mounts, camera, composition, positions, scale, orientation, materials, "
    "lighting, water, or environment. Preserve identity and appearance continuously while "
    "performing the timestamped actions below."
)
PREAMBLE_WN = (
    "The first 1.4167 seconds are the immutable 34-frame rolling history and remain "
    "unchanged, continuing the established camera motion, positions, scale, orientation, "
    "materials, lighting, weather, and ongoing actions without a cut, restart, or reset."
)

# ---------------------------------------------------------------- protagonist states
STAND, WALK = "STAND", "WALK"
BOARD, RIDE_VEHICLE, DRIVE = "BOARD", "RIDE_VEHICLE", "DRIVE"
MOUNT, RIDE_MOUNT, FLY = "MOUNT", "RIDE_MOUNT", "FLY"

# The one substring that identifies each state when a sentence is read back. Each names the
# protagonist's own action: none of them is a noun that could appear as scenery.
SIGNATURE = {
    STAND: "his mark",
    WALK: "on foot",
    BOARD: "climbs in",
    RIDE_VEHICLE: "seated inside",
    DRIVE: "rides inside",
    MOUNT: "pulls himself up",
    RIDE_MOUNT: "astride",
    FLY: "on the broom",
}

# Prose in the register of the released captions. Every variant must contain its own
# signature and no other state's -- `audit_phrases` enforces exactly that.
PHRASES = {
    STAND: [
        "The rainkeeper holds his mark on the wet stone, weight settled, rain running from "
        "the brim of his conical hat",
        "He stays on his mark, head level, the mantle of rain-blackened reeds hanging heavy "
        "across his shoulders",
        "He keeps his mark on the streaming paving, boots planted, watching the lane ahead",
    ],
    WALK: [
        "The rainkeeper carries himself along the wet causeway on foot, his mantle swinging "
        "with the stride",
        "He moves up the lane on foot, boots finding the pale channels of runoff between the "
        "paving stones",
        "He continues along the causeway on foot, slanting rain passing the brim of his hat",
    ],
    BOARD: [
        "He takes hold of the frame of the {obj} and climbs in, gathering the wet mantle "
        "behind him",
        "He ducks his broad hat beneath the low roof and climbs in through the open side of "
        "the {obj}",
        "He climbs in off the streaming stone, settling his weight into the {obj}",
    ],
    RIDE_VEHICLE: [
        "He is seated inside the halted {obj}, framed by its open side, the rain unbroken "
        "beyond it",
        "He sits seated inside the {obj} with his boots clear of the wet road",
        "He remains seated inside the stopped {obj}, the lamps burning amber beside him",
    ],
    DRIVE: [
        "He rides inside the {obj} as it pulls away along the flooded lane",
        "He rides inside the moving {obj}, the causeway sliding past its open side",
        "He rides inside the {obj} while it carries him on down the wet roadway",
    ],
    MOUNT: [
        "He steps in close to the flank of the {obj} and pulls himself up onto its back",
        "He pulls himself up the side of the {obj}, taking his weight on its shoulder",
        "He pulls himself up and swings a leg across the back of the {obj}",
    ],
    RIDE_MOUNT: [
        "He rides astride the {obj}, rocking with its gait above the streaming road",
        "He sits astride the {obj} as it carries him along the wet lane",
        "He stays astride the {obj}, his mantle swaying with each step it takes",
    ],
    FLY: [
        "He rises clear of the ground on the broom, his mantle streaming beneath him",
        "He climbs steadily on the broom, the wet road falling away below him",
        "He holds his climb on the broom, high above the valley with the lane far beneath",
    ],
}

# Actor motion states. Same discipline as the protagonist's: each carries one signature
# that identifies the MOTION, so a clause can be read back and checked against the
# simulation rather than trusted.
A_HALT, A_NEAR, A_AWAY, A_DESC, A_FALL, A_CARRY, A_ALONG = (
    "A_HALT", "A_NEAR", "A_AWAY", "A_DESC", "A_FALL", "A_CARRY", "A_ALONG")
# Within this many metres, what matters is that the actor is right beside him, not the sign
# of the change in distance.
ALONGSIDE_M = 5.0

ACTOR_SIGNATURE = {
    A_HALT: "stands halted",
    A_NEAR: "comes up the",
    A_AWAY: "draws away",
    A_DESC: "comes down along",
    A_FALL: "falls from",
    A_CARRY: "carries him",
    A_ALONG: "draws level with him",
}

ACTOR_PHRASE = {
    A_HALT: "It stands halted on the wet stone, motionless and close by",
    A_NEAR: "It comes up the flooded lane toward him, slowing as it closes",
    A_AWAY: "It draws away along the wet lane, receding from him",
    A_DESC: "It comes down along the line of the road, growing in the frame as it descends",
    A_FALL: "It falls from the aircraft and bursts on the wet stone, throwing smoke and spray",
    A_CARRY: "It carries him along the flooded lane at a steady walk",
    A_ALONG: "It draws level with him on the wet stone, slowing as it comes",
}

# The rig, described from what it actually does. The hand-written captions claimed the
# camera "performs a smooth orbit around him"; this take flies a follow boom and never
# orbits, and a false camera claim misleads the model exactly as a false action does.
CAMERA_GROUND = (
    "The camera follows him in third person from behind and a little to one side, holding "
    "him off centre at close range while the flooded causeway moves past."
)
CAMERA_FLIGHT = (
    "The camera follows him in third person from behind and a little to one side; as he "
    "gains height it keeps back and holds its own altitude, so he rises through the frame "
    "and the roadway and terraces fall away far below him."
)

# How the window opens, for the positional sentence of the scene block.
OPENING = {
    STAND: "At this instant he stands still on {place}",
    WALK: "At this instant he is crossing {place} on foot",
    BOARD: "At this instant he is climbing in through the open side of the rain-cart on {place}",
    RIDE_VEHICLE: "At this instant he sits inside the halted rain-cart on {place}",
    DRIVE: "At this instant he is being carried along {place} inside the moving rain-cart",
    MOUNT: "At this instant he is climbing onto the back of the pale camel on {place}",
    RIDE_MOUNT: "At this instant he sits high on the back of the pale camel on {place}",
    FLY: "At this instant he is already aloft on the broom, well above {place}",
}


def audit_phrases() -> list[str]:
    """Every variant carries its own signature and no other state's. Run by the tests."""
    bad = []
    for state, variants in PHRASES.items():
        sig = SIGNATURE[state]
        for v in variants:
            low = v.lower()
            if sig not in low:
                bad.append(f"{state}: variant missing its signature {sig!r}: {v[:40]}...")
            for other, osig in SIGNATURE.items():
                if other != state and osig in low:
                    bad.append(f"{state}: variant also matches {other} signature {osig!r}")
    for s, sig in SIGNATURE.items():
        for t, tsig in SIGNATURE.items():
            if s != t and (sig in tsig or tsig in sig):
                bad.append(f"signatures {s}/{t} are not distinguishable: {sig!r} vs {tsig!r}")
    for s in (STAND, WALK, BOARD, RIDE_VEHICLE, DRIVE, MOUNT, RIDE_MOUNT, FLY):
        if s not in OPENING:
            bad.append(f"{s} has no OPENING sentence")
    return bad


def classify_actor(clause: str) -> str | None:
    """Read one other-actors clause back to the motion state it asserts."""
    low = clause.lower()
    hit = [st for st, sig in ACTOR_SIGNATURE.items() if sig in low]
    return hit[0] if len(hit) == 1 else None


def classify(sentence: str) -> str | None:
    """Read one timeline sentence back to the protagonist state it asserts.

    Signatures only -- deliberately blind to which nouns the sentence mentions, because the
    captions this replaces mentioned every right noun while describing the wrong action.
    """
    low = sentence.lower()
    hit = [s for s, sig in SIGNATURE.items() if sig in low]
    return hit[0] if len(hit) == 1 else None


# ------------------------------------------------------------------- state derivation
@dataclass
class Phase:
    """One scripted phase: which body carries the bookkeeping, and which attrs mean what.

    `during` is the attribute an action sets while it is running; `after` the one it sets
    once complete. They are the attributes `world/actions.py` itself writes, so the caption
    is keyed to the mechanism rather than to a second, driftable description of it.
    """

    state: str
    holder: str
    during: str | None
    after: str | None
    obj: str


def derive_track(plan, phases: list[Phase]) -> list[tuple[str, str]]:
    """Per frame, what the protagonist is doing. First matching phase wins, so `phases`
    must be ordered latest-first: `mounted` stays set after he has left the camel."""
    out = []
    for _, sj, _ in plan:
        ents = json.loads(sj)["entities"]
        state, obj = None, ""
        for ph in phases:
            e = ents.get(ph.holder)
            if e is None or not e.get("alive", True):
                continue
            at = e.get("attrs", {})
            if ph.during and ph.during not in at:
                continue
            if ph.after and not at.get(ph.after):
                continue
            state, obj = ph.state, ph.obj
            break
        if state is None:
            c = ents["courier"]
            state = WALK if float(c.get("speed", 0.0)) > 0.2 else STAND
            obj = ""
        out.append((state, obj))
    return out


def _segment_bounds(n_segments: int = SEGMENTS):
    edges = [round(WINDOW_SECONDS * k / n_segments, 2) for k in range(n_segments + 1)]
    edges[-1] = WINDOW_SECONDS
    return list(zip(edges[:-1], edges[1:]))


def run_progress(track) -> list[float]:
    """Per frame, how far through the current unbroken run of its state it is, in [0, 1)."""
    out = [0.0] * len(track)
    i = 0
    while i < len(track):
        j = i
        while j < len(track) and track[j] == track[i]:
            j += 1
        span = j - i
        for k in range(i, j):
            out[k] = (k - i) / span
        i = j
    return out


def window_states(track, w: int, n_segments: int = SEGMENTS):
    """The dominant protagonist state of each segment of window `w`.

    Dominant, not first: a segment straddling a transition should be captioned by what the
    protagonist is doing for most of it, and a 17-frame slice is too short to describe twice.
    """
    f0 = w * STRIDE
    prog = run_progress(track)
    out = []
    for i in range(n_segments):
        a = f0 + (i * WINDOW_FRAMES) // n_segments
        b = f0 + ((i + 1) * WINDOW_FRAMES) // n_segments
        seg = track[a:b] or track[a:a + 1]
        counts = {}
        for s, o in seg:
            counts[(s, o)] = counts.get((s, o), 0) + 1
        state, obj = max(counts.items(), key=lambda kv: kv[1])[0]
        # Progress is averaged over the frames of this segment that are actually in that
        # state, so a segment straddling a transition is placed by its own half of the run.
        own = [prog[k] for k in range(a, min(b, len(prog))) if track[k][0] == state]
        out.append((state, obj, sum(own) / len(own) if own else 0.0))
    return out


# ------------------------------------------------------------------- other actors
def actor_state(ents, eid, prev_pos, attached_to):
    """One actor, one frame. Motion is read from displacement, not the `speed` field.

    `speed` is what a rule intends; displacement is what the DUV will actually show, and in
    window 1 the two disagree on the frame that mattered.
    """
    e = ents.get(eid)
    if e is None or not e.get("alive", True):
        return None
    p = np.array(e["position"], float)
    if prev_pos is None:
        return None
    step = float(np.linalg.norm(p - prev_pos))
    rise = -float(p[1]) - -float(prev_pos[1])
    if eid.startswith("ord") and float(p[1]) < -1e-6:
        return A_FALL
    if rise < -0.02:
        return A_DESC
    if step < 0.01:
        return A_HALT
    if attached_to == eid:
        return A_CARRY
    c = np.array(ents["courier"]["position"], float)
    d_now = float(np.hypot(p[0] - c[0], p[2] - c[2]))
    d_prev = float(np.hypot(prev_pos[0] - c[0], prev_pos[2] - c[2]))
    if d_now < ALONGSIDE_M:
        return A_ALONG
    return A_NEAR if d_now < d_prev else A_AWAY


def actor_phases(plan, eid, f0: int, f1: int, track, min_run: int = 10):
    """Contiguous motion phases of one actor across a window, as (state, k0, k1).

    Runs shorter than `min_run` frames are folded into their neighbour: a cart that creeps
    for three frames as it settles is halted, not a second act.
    """
    seq = []
    prev = None
    for k in range(f0, f1 + 1):
        ents = json.loads(plan[k][1])["entities"]
        attached = track[k][1] if track[k][0] in (RIDE_MOUNT, DRIVE, RIDE_VEHICLE) else None
        att_eid = None
        if attached:
            for cand, e in ents.items():
                if cand != "courier" and e.get("alive", True):
                    cp = np.array(e["position"], float)
                    c = np.array(ents["courier"]["position"], float)
                    if float(np.hypot(cp[0] - c[0], cp[2] - c[2])) < 0.45:
                        att_eid = cand
                        break
        st = actor_state(ents, eid, prev, att_eid)
        e = ents.get(eid)
        prev = np.array(e["position"], float) if e is not None else None
        seq.append((k, st))
    runs = []
    for k, st in seq:
        if st is None:
            continue
        if runs and runs[-1][0] == st:
            runs[-1][2] = k
        else:
            runs.append([st, k, k])
    merged = []
    for r in runs:
        if merged and (r[2] - r[1] + 1) < min_run:
            merged[-1][2] = r[2]
        else:
            merged.append(r)
    return [(st, a, b) for st, a, b in merged]


def _other_actors(plan, w: int, names: dict, track) -> str:
    """Every actor that is really present, described phase by phase.

    Presence is measured, and so is motion: an actor alive for less than a fifth of the
    window is left out rather than announced early, and one that approaches and then stops
    gets both clauses with their own sub-ranges instead of one averaged lie.
    """
    f0 = w * STRIDE
    f1 = min(f0 + WINDOW_FRAMES - 1, len(plan) - 1)
    n = f1 - f0 + 1
    lines = []
    for eid, label in names.items():
        alive = 0
        for k in range(f0, f1 + 1):
            e = json.loads(plan[k][1])["entities"].get(eid)
            if e is not None and e.get("alive", True):
                alive += 1
        if alive < max(1, n // 5):
            continue
        for st, a, b in actor_phases(plan, eid, f0, f1, track):
            if (b - a + 1) < max(1, n // 10):
                continue
            t0 = round(WINDOW_SECONDS * (a - f0) / max(n - 1, 1), 2)
            t1 = round(WINDOW_SECONDS * (b - f0) / max(n - 1, 1), 2)
            lines.append(f"{t0:.2f}\u2013{t1:.2f}s, {label}: {ACTOR_PHRASE[st]}")
    if not lines:
        return (f"0.00\u2013{WINDOW_SECONDS:.2f}s: No other visible person, animal, vessel, "
                f"or independently moving identity appears; only the distant rain-carts and "
                f"the falling rain move in the scene.")
    return "; ".join(lines) + "."


# ------------------------------------------------------------------- assembly
def window_caption(track, plan, w: int, content: dict, n_segments: int = SEGMENTS) -> str:
    states = window_states(track, w, n_segments)
    actions = []
    for state, obj, frac in states:
        variants = PHRASES[state]
        actions.append(variants[min(int(frac * len(variants)), len(variants) - 1)]
                       .format(obj=obj))
    camera = CAMERA_FLIGHT if any(st == FLY for st, _, _ in states) else CAMERA_GROUND
    cap = Caption(
        protagonist=content["identity"],
        scene=OPENING[states[0][0]].format(place=content["place"]) + ". " + content["scene"],
        actions=actions,
        other_actors_appearance=content["others_appearance"] + " " + camera,
        other_actors=_other_actors(plan, w, content["actor_names"], track),
    )
    body = cap.render()
    return (PREAMBLE_W0 if w == 0 else PREAMBLE_WN) + "\n\n" + body


def build(plan, phases, content: dict, out_dir, windows: int,
          n_segments: int = SEGMENTS) -> list[Path]:
    """Write prompts/000.txt .. for a take. Returns the paths written.

    LF, not CRLF: `compiler.caption.to_prompt` emits CRLF per the materialize script, but
    every released prompt on this cluster is stored with bare LF and that is what
    run_cp.py reads, so that is what is written here.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    track = derive_track(plan, phases)
    written = []
    for w in range(windows):
        txt = window_caption(track, plan, w, content, n_segments)
        p = out_dir / f"{w:03d}.txt"
        p.write_text(txt, encoding="utf-8", newline="\n")
        written.append(p)
    return written


# ------------------------------------------------------------------- the gate
def check(prompt_dir, plan, phases, windows: int, n_segments: int = SEGMENTS,
          actor_names: dict | None = None) -> list[str]:
    """Do the captions on disk describe what this simulation actually does?

    Reads the files back rather than the objects that produced them, so a prompt directory
    copied from another take -- the failure this exists for -- cannot pass.
    """
    track = derive_track(plan, phases)
    label_to_eid = {v: k for k, v in (actor_names or {}).items()}
    problems = []
    for w in range(windows):
        p = Path(prompt_dir) / f"{w:03d}.txt"
        if not p.exists():
            problems.append(f"w{w}: {p} missing")
            continue
        txt = p.read_text(encoding="utf-8")
        want_pre = PREAMBLE_W0 if w == 0 else PREAMBLE_WN
        if not txt.startswith(want_pre):
            problems.append(f"w{w}: wrong preamble for a "
                            f"{'first' if w == 0 else 'continuation'} window")
        m = re.search(r"Protagonist action timeline:\s*\n\n(.*?)(?:\n\n|$)", txt, re.S)
        if not m:
            problems.append(f"w{w}: no protagonist action timeline")
            continue
        segs = [s.strip() for s in m.group(1).split(";") if s.strip()]
        want = window_states(track, w, n_segments)
        if len(segs) != len(want):
            problems.append(f"w{w}: {len(segs)} timeline segments, simulation needs {len(want)}")
            continue
        for i, (seg, (state, _obj, _frac)) in enumerate(zip(segs, want)):
            got = classify(seg)
            if got is None:
                problems.append(f"w{w} seg{i}: says nothing identifiable about the "
                                f"protagonist; simulation has him {state}")
            elif got != state:
                problems.append(f"w{w} seg{i}: caption says {got}, simulation says {state}")

        # The other-actors line gets the same treatment. Window 1 once told the model the
        # rain-cart was driving off down the lane while the proxy had it stopped 2.3 m from
        # the protagonist, and nothing here noticed.
        # Actor clauses are only checkable against a label->eid map. Without one the caller
        # has opted out; the protagonist line above is still checked either way.
        if not label_to_eid:
            continue
        mo = re.search(r"Other visible actors:\s*\n\n(.*?)\s*$", txt, re.S)
        if not mo:
            problems.append(f"w{w}: no other-actors section")
            continue
        body = mo.group(1).strip()
        if body.lower().startswith("0.00") and "no other visible" in body.lower():
            continue
        f0 = w * STRIDE
        f1 = min(f0 + WINDOW_FRAMES - 1, len(track) - 1)
        for clause in body.split(";"):
            clause = clause.strip().rstrip(".")
            if not clause:
                continue
            m2 = re.match(r"(\d+\.\d+)\u2013(\d+\.\d+)s,\s*(.+?):\s*(.*)$", clause)
            if not m2:
                problems.append(f"w{w}: unparseable actor clause {clause[:50]!r}")
                continue
            t0, t1, label, what = m2.groups()
            st = classify_actor(what)
            if st is None:
                problems.append(f"w{w}: actor clause for {label!r} says nothing "
                                f"identifiable about its motion")
                continue
            eid = label_to_eid.get(label)
            if eid is None:
                problems.append(f"w{w}: actor clause names {label!r}, not an actor of "
                                f"this take")
                continue
            a = f0 + int(round(float(t0) / WINDOW_SECONDS * (f1 - f0)))
            b = f0 + int(round(float(t1) / WINDOW_SECONDS * (f1 - f0)))
            real = {s2 for s2, x, y in actor_phases(plan, eid, f0, f1, track)
                    if not (y < a or x > b)}
            if st not in real:
                problems.append(f"w{w}: {label} is claimed {st} over {t0}-{t1}s but the "
                                f"simulation has it {sorted(real) or ['absent']}")
    return problems

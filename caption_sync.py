"""Gate: do a take's captions on disk describe what that take's simulation actually does?

Run before spending an allocation, next to checkdefs / test_cull / test_actions.

This exists because a take was generated with captions copied from an older version of the
script. The captions never described the protagonist boarding or driving the cart -- which
the simulation spends frames 178-328 doing -- and brought the camel in a window early, and
the model obeyed the caption and ignored the DUV. While that is possible, no adherence
number means anything: "the model ignored the proxy" and "the model obeyed a caption that
contradicted the proxy" produce the same pictures.

A noun check does NOT catch this, and that is the whole point. The word "car" appears 11
times in the broken window 0 and 8 times in window 1 -- as scenery the protagonist happens
to be standing next to. So this compares what the PROTAGONIST does, frame by frame, against
what each caption segment claims he does, via the signatures in world/narrate.py.

Usage:
    caption_sync.py <take_dir>         # take_dir holds prompts/ (and usually conditions/)
    caption_sync.py --report <dir>     # print the per-window table instead of just failing
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from take_actions import PHASES, WINDOWS, simulate
from world import narrate


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    report = "--report" in sys.argv
    if not args:
        print(__doc__.strip().splitlines()[-3])
        return 2
    take = Path(args[0])
    pdir = take / "prompts" if (take / "prompts").is_dir() else take

    plan, _, _ = simulate(check=False)
    track = narrate.derive_track(plan, PHASES)

    if report:
        print("win  frames      simulation says the protagonist is        caption segments")
        for w in range(WINDOWS):
            want = narrate.window_states(track, w)
            f0 = w * narrate.STRIDE
            seq = []
            for st, _o, _f in want:
                if not seq or seq[-1] != st:
                    seq.append(st)
            p = pdir / f"{w:03d}.txt"
            got = "(missing)"
            if p.exists():
                txt = p.read_text(encoding="utf-8")
                if "Protagonist action timeline:" in txt:
                    body = txt.split("Protagonist action timeline:")[1]
                    body = body.split("Other visible actors:")[0]
                    cls = [narrate.classify(s) or "?" for s in body.split(";") if s.strip()]
                    out = []
                    for c in cls:
                        if not out or out[-1] != c:
                            out.append(c)
                    got = " -> ".join(out)
            print("w%d   %3d-%3d     %-40s %s"
                  % (w, f0, f0 + narrate.WINDOW_FRAMES - 1, " -> ".join(seq), got))
        print()

    from world.take_content import content as take_content
    problems = narrate.check(pdir, plan, PHASES, WINDOWS,
                             actor_names=take_content()["actor_names"])
    if problems:
        print(f"caption_sync: {len(problems)} disagreement(s) between {pdir} and the "
              f"simulation")
        for b in problems:
            print("  -", b)
        return 1
    print(f"caption_sync: captions in {pdir} match the simulation "
          f"({WINDOWS} windows x {narrate.SEGMENTS} segments)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

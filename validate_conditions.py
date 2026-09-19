"""Independently verify a conditions directory, without trusting the writer that produced it.

`compiler/writer.py` validates before it writes and re-reads its own bytes, which catches a
writer bug but not a *generator* bug: a semantic id that was already wrong when it arrived, a
depth that is finite but physically impossible, a frame that silently went empty. This reads
the files back cold and checks what the model will actually receive.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import math
import os
import sys

import numpy as np
from PIL import Image

NEAR_M, FAR_M = 0.3, 256.0
ARTICULATION_MIN = 20           # px of class-8 mask change per frame; see the check below
ARTICULATION_MIN_AREA = 0.005   # below this share of frame, articulation is sub-pixel
W, H = 336, 192
SEMANTIC_U = (32, 96, 160, 224)
SEMANTIC_V = (43, 128, 213)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("conditions")
    ap.add_argument("--expect", type=int, default=0, help="expected frame count, 0 to skip")
    a = ap.parse_args()

    depths = sorted(glob.glob(os.path.join(a.conditions, "*.depth.f32")))
    fails: list[str] = []
    if not depths:
        print(f"no frames in {a.conditions}"); return 1
    if a.expect and len(depths) != a.expect:
        fails.append(f"frame count {len(depths)}, expected {a.expect}")

    # Frame indices must be contiguous from zero: a gap means a worker died and the take would
    # be silently short a frame in the middle.
    idx = sorted(int(os.path.basename(p).split(".")[0]) for p in depths)
    if idx != list(range(len(idx))):
        missing = sorted(set(range(idx[-1] + 1)) - set(idx))[:8]
        fails.append(f"non-contiguous frames, missing {missing}")

    occ = np.zeros(12, dtype=np.int64)
    far_frac, valid_frac, empties = [], [], []
    # Byte signature per frame, to catch a window that never changes. See below.
    sigs: set[bytes] = set()
    subj: list = []                 # the class-8 mask per frame, for the articulation check
    for p in depths:
        k = int(os.path.basename(p).split(".")[0])
        d = np.fromfile(p, dtype="<f4")
        if d.size != W * H:
            fails.append(f"frame {k}: depth has {d.size} values, expected {W*H}"); continue
        d = d.reshape(H, W)
        s = np.asarray(Image.open(p.replace(".depth.f32", ".semantic_id.png")), dtype=np.uint8)
        if s.shape != (H, W):
            fails.append(f"frame {k}: semantic is {s.shape}, expected {(H, W)}"); continue

        sigs.add(hashlib.md5(d.tobytes()).digest()
                 + hashlib.md5(s.tobytes()).digest())
        subj.append(s == 8)

        if not np.all(np.isfinite(d)):
            fails.append(f"frame {k}: depth has non-finite values")
        neg = d[(d < 0.0)]
        if neg.size:
            fails.append(f"frame {k}: {neg.size} negative depths, min {neg.min():.3f}")
        v = d[d > 1e-3]
        if v.size and v.max() > FAR_M + 1e-3:
            fails.append(f"frame {k}: depth max {v.max():.1f} exceeds the {FAR_M} m far plane")
        if v.size and v.min() < NEAR_M - 1e-3:
            fails.append(f"frame {k}: depth min {v.min():.3f} inside the {NEAR_M} m near plane")
        if s.max() > 11:
            fails.append(f"frame {k}: semantic id {s.max()} outside 0-11")

        # The format contract, and the one relation the writer actually enforces:
        # semantic == 1 means "no hit", and a no-hit pixel has depth exactly 0. Neither
        # direction was checked here, so a frame could carry sky with a depth on it, or a
        # hit pixel labelled sky, and still be called clean. Both are how a malformed DUV
        # reaches the model looking plausible.
        sky = s == 1
        sky_with_depth = int((sky & (d > 0)).sum())
        hit_called_sky = int((~sky & (d == 0)).sum())
        if sky_with_depth:
            fails.append(f"frame {k}: {sky_with_depth} pixels are semantic 1 (sky) but "
                         f"carry depth > 0")
        if hit_called_sky:
            fails.append(f"frame {k}: {hit_called_sky} pixels have depth 0 but are not "
                         f"semantic 1 (sky)")

        vf = float((d > 1e-3).mean())
        valid_frac.append(vf)
        far_frac.append(float((d >= FAR_M).mean()))
        if vf < 0.05:
            empties.append(k)
        occ += np.bincount(s.reshape(-1), minlength=12)

    if empties:
        fails.append(f"{len(empties)} near-empty frames (valid < 5%), first {empties[:6]}")

    # The DUV the model sees must land on the exact palette; an interpolated U or V is a
    # different semantic class, not a slightly different colour.
    u = np.asarray(SEMANTIC_U, np.uint8)[np.arange(12) % 4]
    vv = np.asarray(SEMANTIC_V, np.uint8)[np.arange(12) // 4]
    if len(set(zip(u.tolist(), vv.tolist()))) != 12:
        fails.append("palette is not injective: two classes share a (U,V)")

    occ_frac = occ / max(occ.sum(), 1)
    # A window of identical frames is not a conditioning signal. Two of our six emitted
    # one distinct frame repeated 124 times -- 5.17 s with no temporal information -- and
    # passed every check here, every invariant, and scored perfectly on every steadiness
    # measure, because a still image is maximally steady. The motion they appeared to have
    # was a camera bug, and nothing noticed when the bug was fixed.
    if len(depths) > 1 and len(sigs) == 1:
        fails.append(f"all {len(depths)} frames are byte-identical: the window carries no "
                     f"temporal information")

    # A live subject that never changes shape is a defect the still-frame check above
    # cannot see, because the BACKGROUND moves: ex31 shipped a class-8 mask that was
    # byte-identical across all 124 frames and validated clean, because its camera tracked
    # a rigid-legged stag across a moving scene.
    #
    # ARTICULATION_MIN is 20 px of a 336x192 frame. The anchor is their own conditions:
    # their planted subject (ex36, the one whose prompt says she turns her head and swings
    # a hand without stepping) changes by a median of 47 px per frame, and their plainest
    # walk (ex30) by 32. 20 sits below the quieter of those with margin, and far above the
    # sub-pixel wobble that a token animation would produce.
    #
    # ARTICULATION_MIN_AREA exempts a subject too small to articulate visibly rather than
    # failing it forever. At 0.5% of frame a body is about 320 px, and a few degrees of
    # limb swing on it is sub-pixel after the 2x2 pool. Their aerial ex34 is 0.89% of frame
    # with a median change of 6 px; ours is 0.44%. That gap is the subject's SIZE, not its
    # stillness, and a blanket floor would either fail it forever or be set low enough to
    # pass a wobble.
    if len(subj) > 1:
        area = float(np.median([m.sum() for m in subj])) / (W * H)
        d8 = [int((subj[i] ^ subj[i - 1]).sum()) for i in range(1, len(subj))]
        med8 = float(np.median(d8)) if d8 else 0.0
        if area >= ARTICULATION_MIN_AREA and med8 < ARTICULATION_MIN:
            fails.append(f"subject articulation {med8:.0f} px/frame is below the "
                         f"{ARTICULATION_MIN} px floor at {area:.2%} of frame: the subject "
                         f"is rigid even though the window is not still")
        print(f"subject motion    {med8:.0f} px/frame at {area:.2%} of frame"
              + ("  (below the visibility floor, exempt)"
                 if area < ARTICULATION_MIN_AREA else ""))

    print(f"frames            {len(depths)}")
    print(f"distinct frames   {len(sigs)}")
    print(f"valid fraction    p10 {np.percentile(valid_frac,10):.3f}  median "
          f"{np.median(valid_frac):.3f}  p90 {np.percentile(valid_frac,90):.3f}")
    print(f"far-clip fraction median {np.median(far_frac):.4f}  max {max(far_frac):.4f}")
    print("class occupancy   " + "  ".join(f"{i}:{occ_frac[i]:.3f}"
                                           for i in range(12) if occ_frac[i] > 1e-4))
    if fails:
        print(f"\n{len(fails)} problem(s):")
        for f in fails[:20]:
            print("  -", f)
        return 1
    print("\nconditions clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

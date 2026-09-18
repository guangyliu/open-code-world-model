"""Independently verify a conditions directory, without trusting the writer that produced it.

`compiler/writer.py` validates before it writes and re-reads its own bytes, which catches a
writer bug but not a *generator* bug: a semantic id that was already wrong when it arrived, a
depth that is finite but physically impossible, a frame that silently went empty. This reads
the files back cold and checks what the model will actually receive.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import numpy as np
from PIL import Image

NEAR_M, FAR_M = 0.3, 256.0
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
    for p in depths:
        k = int(os.path.basename(p).split(".")[0])
        d = np.fromfile(p, dtype="<f4")
        if d.size != W * H:
            fails.append(f"frame {k}: depth has {d.size} values, expected {W*H}"); continue
        d = d.reshape(H, W)
        s = np.asarray(Image.open(p.replace(".depth.f32", ".semantic_id.png")), dtype=np.uint8)
        if s.shape != (H, W):
            fails.append(f"frame {k}: semantic is {s.shape}, expected {(H, W)}"); continue

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
    print(f"frames            {len(depths)}")
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

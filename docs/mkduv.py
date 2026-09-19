"""Render each proxy's semantic map as a video, so every output clip can sit under the
geometry that drove it.

Semantic ids are uint8 in both released condition families, so this needs no depth decode
and no family handling -- the gray16/metric split only affects depth.

Upscaled with nearest neighbour on purpose: the proxy is 336x192 and blocky, and smoothing
it would suggest a precision it does not have.
"""
import glob
import os
import sys

import av
import numpy as np
from PIL import Image

COL = {0: (92, 92, 110), 1: (150, 200, 245), 2: (60, 130, 220), 4: (120, 110, 100),
       5: (55, 145, 70), 8: (255, 70, 70), 9: (255, 170, 40), 10: (250, 225, 60)}
LUT = np.zeros((256, 3), np.uint8)
LUT[:] = (255, 0, 255)                       # unmapped ids scream, rather than blending in
for k, v in COL.items():
    LUT[k] = v

EX = [18, 28, 30, 31, 34, 36]
REL = os.environ.get("CWM_EXAMPLES", "examples40")
OURS = os.environ.get("CWM_REPRO", "out/repro")
DST = sys.argv[1] if len(sys.argv) > 1 else "docs/video"
SCALE = 2

os.makedirs(DST, exist_ok=True)


def write(frames, path):
    h, w = frames[0].shape[:2]
    c = av.open(path, mode="w", options={"movflags": "+faststart"})
    s = c.add_stream("libx264", rate=24)
    s.width, s.height, s.pix_fmt = w * SCALE, h * SCALE, "yuv420p"
    s.options = {"crf": "26", "preset": "slow"}
    for f in frames:
        big = np.repeat(np.repeat(f, SCALE, 0), SCALE, 1)
        for pkt in s.encode(av.VideoFrame.from_ndarray(big, format="rgb24")):
            c.mux(pkt)
    for pkt in s.encode():
        c.mux(pkt)
    c.close()
    return os.path.getsize(path)


total = 0
for n in EX:
    sem = np.load(f"{REL}/{n}/condition/000.npz")["semantic_id"]
    bad = sorted(set(np.unique(sem)) - set(COL))
    assert not bad, f"ex{n} released: unmapped ids {bad}"
    total += write([LUT[f] for f in sem], f"{DST}/duv_rel_ex{n}.mp4")

    pngs = sorted(glob.glob(f"{OURS}/ex{n}/conditions/*.semantic_id.png"))[:len(sem)]
    assert len(pngs) == len(sem), f"ex{n} ours: {len(pngs)} frames, expected {len(sem)}"
    ours = []
    for p in pngs:
        a = np.array(Image.open(p))
        a = a[..., 0] if a.ndim == 3 else a
        bad = sorted(set(np.unique(a)) - set(COL))
        assert not bad, f"{p}: unmapped ids {bad}"
        ours.append(LUT[a])
    total += write(ours, f"{DST}/duv_our_ex{n}.mp4")
    print(f"  ex{n}: {len(sem)} frames each")

print(f"\n12 proxy videos, {total/1e6:.1f} MB total, into {DST}")

"""Per-chunk drift of the chunked LynnReal takes, against the single-shot 124-frame take.

Two no-reference texture measures per frame, both on the luma at 1/2 resolution:
  hf   = mean |frame - gaussian_blur(frame)|   (high-frequency energy: grain, coarsening)
  lap  = variance of the Laplacian             (sharpness)
plus the mean absolute difference to the anchor frame's colour histogram (tone drift).
Reported per 21-frame block so runs with different chunk lengths line up on the same time axis.
"""
import sys, json
import av, numpy as np
from PIL import Image, ImageFilter

def load(p):
    c = av.open(p); fs = [x.to_ndarray(format="rgb24") for x in c.decode(video=0)]; c.close(); return fs

def luma(f):
    im = Image.fromarray(f).convert("L").resize((480, 272), Image.BILINEAR); return np.asarray(im, np.float32)

def measures(f):
    y = luma(f); im = Image.fromarray(y.astype(np.uint8))
    blur = np.asarray(im.filter(ImageFilter.GaussianBlur(2)), np.float32)
    hf = float(np.abs(y - blur).mean())
    lap = np.asarray(im.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1)), np.float32)
    return hf, float(lap.var())

def hist(f):
    h = np.concatenate([np.histogram(f[..., c], 32, (0, 255))[0] for c in range(3)]).astype(np.float64); return h / h.sum()

runs = dict(arg.split("=", 1) for arg in sys.argv[1:])
out = {}
for name, path in runs.items():
    fs = load(path); h0 = hist(fs[0])
    rows = []
    for i, f in enumerate(fs):
        hf, lap = measures(f); rows.append((i, hf, lap, float(np.abs(hist(f) - h0).sum())))
    blocks = {}
    for i, hf, lap, tone in rows:
        b = (i - 1) // 21 if i else 0
        blocks.setdefault(b, []).append((hf, lap, tone))
    out[name] = {b: [round(float(np.mean([r[j] for r in v])), 3) for j in range(3)] for b, v in blocks.items()}
    print(f"{name:12s} frames={len(fs)} " + "  ".join(f"b{b}:hf={m[0]:.2f}/lap={m[1]:.0f}/tone={m[2]:.2f}" for b, m in sorted(out[name].items())))
json.dump(out, open("/tmp/drift.json", "w"), indent=1)

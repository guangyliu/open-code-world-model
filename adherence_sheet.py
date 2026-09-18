"""Overlay the DUV proxy's own object boxes on a render, for several runs side by side.

This has been rebuilt by hand four times. It is the only adherence check in this project that
has ever held up: every statistical proxy tried -- cross-seed pixel difference, dark-blob
centroid tracking, gradient energy inside the boxes, Laplacian variance -- failed, because the
signal is small objects near the horizon and it is swamped by rain, wet stone and lane
markings. Drawing the proxy's own boxes on the frame and looking is direct, and it caught both
the half-canvas scale error and the native-DUV adherence collapse.

Crop at native resolution. A 560px-wide contact sheet once made distant vehicles vanish into
the downscale and produced a confident, wrong claim that 5-step sampling had dropped them.
"""
from __future__ import annotations

import argparse, math, os
import av
import numpy as np
from PIL import Image, ImageDraw

W, H = 336, 192


def frame_at(path, index):
    c = av.open(path)
    for i, f in enumerate(c.decode(video=0)):
        if i == index:
            a = f.to_ndarray(format="rgb24"); c.close(); return a
    c.close(); raise IndexError(index)


def proxy_boxes(npz, frame, cls=10, min_px=15, x_gap=6):
    """Bounding boxes of one semantic class in the proxy, normalised, with median depth."""
    z = np.load(npz); sem = z["semantic_id"]; dep = z["depth"]
    if dep.dtype != np.float32 or dep.max() > 1000:
        # uint16 files store fixed-log codes, not metres; the two are told apart by the
        # example's metadata, never by dtype alone.
        near, far = 0.3, 256.0
        sc = math.log(far) - math.log(near)
        dep = np.where(dep == 0, 0.0, np.exp(math.log(far) - dep.astype(np.float64) / 65535.0 * sc))
    dep = dep.astype(np.float32)
    m = sem[frame] == cls
    ys, xs = np.nonzero(m)
    if not len(xs):
        return []
    o = np.argsort(xs); xs, ys = xs[o], ys[o]
    out = []
    for s in np.split(np.arange(len(xs)), np.nonzero(np.diff(xs) > x_gap)[0] + 1):
        if len(s) < min_px:
            continue
        gx, gy = xs[s], ys[s]
        d = dep[frame][gy, gx]; d = d[d > 1e-3]
        out.append((gx.min() / W, gy.min() / H, (gx.max() + 1) / W, (gy.max() + 1) / H,
                    float(np.median(d)) if len(d) else float("nan")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="the proxy window the renders were made from")
    ap.add_argument("--frame", type=int, default=60)
    ap.add_argument("--video", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--out", default="adherence.png")
    ap.add_argument("--band", type=int, default=420, help="pixel rows to keep from the top")
    # Which class to box. 8 human, 9 animal, 10 vehicle, 11 prop -- so "did he actually get
    # into the car" and "did he actually end up on the camel" are different overlays, and
    # boxing vehicles for a camel shot silently draws nothing.
    ap.add_argument("--cls", type=int, default=10)
    ap.add_argument("--min-px", type=int, default=15)
    a = ap.parse_args()

    boxes = proxy_boxes(a.npz, a.frame, cls=a.cls, min_px=a.min_px)
    print(f"proxy frame {a.frame}: {len(boxes)} object boxes of class {a.cls}")
    if not boxes:
        print("  no boxes for this class in this frame; the overlay would be empty")
    for i, (x0, y0, x1, y1, d) in enumerate(boxes):
        print(f"  {i}: cx={(x0+x1)/2:.3f} cy={(y0+y1)/2:.3f} depth={d:.0f}m")

    tiles = []
    for spec in a.video:
        lab, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"  missing: {path}"); continue
        im = Image.fromarray(frame_at(path, a.frame)).convert("RGB")
        w, h = im.size
        d = ImageDraw.Draw(im)
        for x0, y0, x1, y1, dm in boxes:
            d.rectangle((x0 * w, y0 * h, x1 * w, y1 * h), outline=(255, 40, 40), width=3)
            d.text((x0 * w, max(0, y0 * h - 15)), f"{dm:.0f}m", fill=(255, 130, 130))
        d.rectangle((0, 0, w, 26), fill=(0, 0, 0))
        d.text((8, 7), f"{lab}   red = where the DUV proxy puts class {a.cls}",
               fill=(255, 255, 255))
        tiles.append(im.crop((0, 0, w, min(a.band, h))))

    if not tiles:
        print("nothing to draw"); return 1
    w, h = tiles[0].size
    sheet = Image.new("RGB", (w, h * len(tiles)), "black")
    for i, t in enumerate(tiles):
        sheet.paste(t, (0, i * h))
    sheet.save(a.out)
    print(f"wrote {a.out} ({sheet.size[0]}x{sheet.size[1]}, native resolution)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build docs/gallery/: for each gallery scene, the proxy (semantic over depth), the video
MiniMax-H3 made from it in its native reference mode (no LoRA), and the video it made with
the CWM LoRA -- same anchor, same DUV, same caption, same seed.

    python docs/mkgallery.py --gallery out/gallery --videos /path/to/portb/out --dst docs/gallery

Videos are re-encoded for the web (h264, faststart, 672 px wide). Nothing here is measured;
it is a page. The measurements are in each scene's meta.json and validate output.
"""
import argparse
import glob
import json
import os
import sys

import av
import numpy as np
from PIL import Image

COL = {0: (92, 92, 110), 1: (150, 200, 245), 2: (60, 130, 220), 4: (120, 110, 100),
       5: (55, 145, 70), 8: (255, 70, 70), 9: (255, 170, 40), 10: (250, 225, 60)}
LUT = np.zeros((256, 3), np.uint8); LUT[:] = (255, 0, 255)
for k, v in COL.items():
    LUT[k] = v

ORDER = ["boulevard", "causeway", "stream", "witchroad", "frostcourt", "monastery"]
BLURB = {
    "boulevard": ("Board and drive off", "He walks up the boulevard; a shuttle comes down it, halts beside him; he climbs in; it pulls away with him inside. Anchor: released example 40."),
    "causeway": ("A wagon passes, the camera eases back", "He runs up the shoulder; a wagon comes down the causeway on his right and passes; the boom lengthens from 3.75 to 6.5 m as it does. Anchor: example 19."),
    "stream": ("Turn away from the river", "He runs along the water's edge, then breaks right; the follow rig swings round behind him and the river leaves the frame. Anchor: example 4."),
    "witchroad": ("Stop, and the camera orbits", "He walks to the reliquary, stands for two seconds while the camera swings 40° round him, then walks on. Anchor: example 30."),
    "frostcourt": ("Two leaps", "He runs across the apron and leaps the curb twice; the caption carries the leaps because the JUMP state is read from the simulation. Anchor: example 25."),
    "monastery": ("Stop, push in, pull back", "He runs up the road, stops at the drums while the camera pushes in from 4 m to 2.2 m, then runs on as it drops back. Anchor: example 13."),
}


def depth_img(z):
    g = np.zeros(z.shape + (3,), np.uint8)
    hit = z > 0
    t = np.clip(np.log(np.maximum(z, 0.3) / 0.3) / np.log(256 / 0.3), 0, 1)
    v = (235 - 215 * t).astype(np.uint8)
    g[hit] = np.stack([v, v, v], -1)[hit]
    g[~hit] = (34, 38, 52)
    return g


def write(frames, path, fps=24):
    h, w = frames[0].shape[:2]
    c = av.open(path, mode="w", options={"movflags": "+faststart"})
    s = c.add_stream("libx264", rate=fps)
    s.width, s.height, s.pix_fmt = w, h, "yuv420p"
    s.options = {"crf": "24", "preset": "slow"}
    for f in frames:
        for pkt in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")):
            c.mux(pkt)
    for pkt in s.encode():
        c.mux(pkt)
    c.close()
    return os.path.getsize(path)


def transcode(src, dst, width=672):
    c = av.open(src)
    frames = []
    for f in c.decode(video=0):
        im = f.to_image()
        im = im.resize((width, round(im.height * width / im.width)))
        frames.append(np.array(im))
    fps = float(c.streams.video[0].average_rate or 24)
    c.close()
    return write(frames, dst, fps=round(fps)), len(frames)


def proxy_video(cond, dst):
    pngs = sorted(glob.glob(f"{cond}/*.semantic_id.png"))
    frames = []
    for p in pngs:
        a = np.array(Image.open(p)); a = a[..., 0] if a.ndim == 3 else a
        z = np.fromfile(p.replace("semantic_id.png", "depth.f32"), "<f4").reshape(192, 336)
        f = np.concatenate([LUT[a], depth_img(z)], 0)
        frames.append(np.repeat(np.repeat(f, 2, 0), 2, 1))
    return write(frames, dst), len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", default="out/gallery")
    ap.add_argument("--videos", required=True, help="dir holding gal_<scene>_lora.mp4 / _nolora.mp4")
    ap.add_argument("--dst", default="docs/gallery")
    a = ap.parse_args()
    os.makedirs(f"{a.dst}/video", exist_ok=True)
    cards, total = [], 0
    for n in ORDER:
        d = f"{a.gallery}/{n}"
        if not os.path.isdir(d):
            continue
        meta = json.load(open(f"{d}/meta.json"))
        Image.open(f"{d}/first_frame.png").convert("RGB").resize((672, 384)).save(f"{a.dst}/video/{n}_anchor.jpg", quality=88)
        sz, nf = proxy_video(f"{d}/conditions", f"{a.dst}/video/{n}_proxy.mp4"); total += sz
        have = {}
        for mode in ("nolora", "lora"):
            # The driver appends _nolora to the tag itself, so a no-LoRA run asked for as
            # gal_<n>_nolora lands as gal_<n>_nolora_nolora.mp4.
            cands = [f"{a.videos}/gal_{n}_{mode}.mp4", f"{a.videos}/gal_{n}_{mode}_{mode}.mp4"]
            src = next((c for c in cands if os.path.exists(c)), None)
            if src:
                sz, _ = transcode(src, f"{a.dst}/video/{n}_{mode}.mp4"); total += sz
                have[mode] = True
        t, f0 = meta["target"], meta["frame0"]
        title, blurb = BLURB[n]
        vid = lambda k, cap: (f'<figure><video controls muted loop playsinline preload="metadata" src="video/{n}_{k}.mp4"></video><figcaption>{cap}</figcaption></figure>'
                              if (k == "proxy" or have.get(k)) else f'<figure class="missing"><div>not rendered</div><figcaption>{cap}</figcaption></figure>')
        cards.append(f"""<section class="scene" id="{n}">
<header><h2>{title}</h2><p class="blurb">{blurb}</p></header>
<div class="row">
  <figure><img src="video/{n}_anchor.jpg" alt=""><figcaption>anchor: released first frame, example {meta['anchor_example']}</figcaption></figure>
  {vid("proxy", "our proxy — semantic over depth, 124 frames")}
  {vid("nolora", "MiniMax-H3, native reference mode (anchor + proxy video + caption, no LoRA)")}
  {vid("lora", "MiniMax-H3 + CWM LoRA — same anchor, proxy, caption and seed")}
</div>
<p class="meta">frame 0 vs anchor — subject x {f0['cx']} / {t['cx']}, y {f0['cy']} / {t['cy']}, lens→subject {f0['z8']} / {t['z8']} m, sky {f0['sky']}; {nf} frames, conditions clean</p>
</section>""")
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gallery — does the LoRA read the proxy?</title>
<style>
:root{{--bg:#f3f2ee;--panel:#fff;--ink:#1c1b1f;--mute:#5f5e66;--line:#dcdad3;--mono:ui-monospace,Menlo,Consolas,monospace;--sans:system-ui,-apple-system,"Segoe UI",sans-serif}}
@media (prefers-color-scheme:dark){{:root{{--bg:#17181d;--panel:#1f2027;--ink:#e8e7ee;--mute:#a4a3ad;--line:#33343d}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:15px;line-height:1.5;padding:32px clamp(16px,4vw,48px) 64px}}
main{{max-width:1240px;margin:0 auto}} h1{{font-size:clamp(22px,3vw,30px);margin:0 0 6px}} h2{{font-size:19px;margin:0 0 4px}}
.lede{{color:var(--mute);max-width:80ch;margin:0 0 28px}} .blurb{{color:var(--mute);max-width:90ch;margin:0 0 12px}}
.scene{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-bottom:26px}}
.row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:12px}}
figure{{margin:0}} video,img{{display:block;width:100%;max-width:100%;border-radius:8px;background:#000;aspect-ratio:672/384}}
figure video[src$="_proxy.mp4"]{{aspect-ratio:672/768}}
figcaption,.meta{{font:12.5px var(--mono);color:var(--mute);margin-top:6px}} .missing div{{aspect-ratio:672/384;border:1px dashed var(--line);border-radius:8px;display:grid;place-items:center;color:var(--mute)}}
</style></head><body><main>
<h1>Does the LoRA read the proxy?</h1>
<p class="lede">Six scenes. Each starts from a released first frame (for its style) and everything after frame 0 is ours: the geometry, the actions, the camera, and the caption compiled from the same simulation. MiniMax-H3 then generates each scene twice with the same anchor, proxy video, caption and seed — once in its native reference mode, once with the CWM LoRA. The only variable is whether the model was taught to read depth and semantics.</p>
{''.join(cards)}
<p class="lede">Made with <code>examples/gallery.py</code> and <code>docs/mkgallery.py</code>. Videos re-encoded for the web at 672 px; the conditions themselves are 336×192 float32 depth + uint8 semantic ids.</p>
</main></body></html>"""
    open(f"{a.dst}/index.html", "w").write(html)
    print(f"{len(cards)} scenes, {total/1e6:.1f} MB of video -> {a.dst}/index.html")


if __name__ == "__main__":
    main()

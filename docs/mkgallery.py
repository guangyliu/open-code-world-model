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
        im = im.resize((width, 2 * round(im.height * width / im.width / 2)))   # yuv420p needs even dims (960x544 -> 672x380)
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


# LynnReal-Omni driven chunk by chunk: 22 frames per chunk, frame 0 pinned to the last delivered
# frame, one resident process (lynnreal/chunk_drive.py). Tag -> label, in table order.
CHUNK_RUNS = [
    ("auto", "as released: cuDNN attention, official VAE, every module staged to CPU between phases"),
    ("auto_lightvae", "as released, lightweight VAE"),
    ("_flash_3", "FlashAttention-3"),
    ("_flash_3_lightvae", "FlashAttention-3, lightweight VAE"),
    ("fa3_pin_nostage", "FlashAttention-3, nothing staged, text encoder pinned on the GPU"),
    ("fa3_pin_nostage_lightvae", "same, lightweight VAE"),
    ("fa3_pin_nostage_textonly", "same, Qwen vision blocks dropped from the text context (946 rows instead of 8692)"),
    ("hist22", "same as pinned, plus the release's bounded FramePack latent history"),
    ("c39", "pinned, 39-frame chunks"),
    ("c56", "pinned, 56-frame chunks"),
    ("single124", "no chunking: the whole 124 frames in one call"),
]
CHUNK_VIDEOS = ["auto", "fa3_pin_nostage", "hist22", "c56", "single124"]
VIDEO_LABEL = {"auto": "22-frame chunks, as released", "fa3_pin_nostage": "22-frame chunks, nothing staged (same frames as released, faster)",
               "hist22": "22-frame chunks + latent history", "c56": "56-frame chunks", "single124": "no chunking, 124 frames in one call"}


def chunk_section(runs, proxy, dst):
    rows, have = [], {}
    for tag, label in CHUNK_RUNS:
        tj = f"{runs}/boulevard_540_{tag}/timing.json"
        if not os.path.exists(tj):
            continue
        t = json.load(open(tj))
        ch = t["chunks"][1:] or t["chunks"]
        med = lambda k: sorted(c[k] for c in ch)[len(ch) // 2]
        new = ch[0]["new_frames"]
        # A single call has no warm chunk to take the median of; its first-call conditioning number
        # includes staging the text encoder, so use the measured VLM forward instead.
        cond = med("vlm_forward_s") if len(t["chunks"]) == 1 and "vlm_forward_s" in ch[0] else med("conditioning_s")
        rows.append(f"<tr><td>{label}</td><td>{cond:.1f}</td><td>{med('generate_s'):.1f}</td><td>{med('dit_s'):.1f}</td>"
                    f"<td>{med('decode_s'):.2f}</td><td>{max(c['peak_gb'] for c in ch):.0f}</td><td>{(cond + med('generate_s')) / (new / 24):.0f}×</td></tr>")
        if tag in CHUNK_VIDEOS:
            transcode(f"{runs}/boulevard_540_{tag}/take.mp4", f"{dst}/video/chunked_{tag}.mp4")
            have[tag] = VIDEO_LABEL.get(tag, label)
    if not rows:
        return ""
    if proxy:
        transcode(proxy, f"{dst}/video/chunked_proxy.mp4")
    vids = [f'<figure><video controls muted loop playsinline preload="metadata" src="video/chunked_{k}.mp4"></video><figcaption>{v}</figcaption></figure>' for k, v in have.items()]
    if proxy:
        vids.insert(0, '<figure><video controls muted loop playsinline preload="metadata" src="video/chunked_proxy.mp4"></video><figcaption>the control video: our proxy as a flat-shaded game render, 960×544, cut into 22-frame chunks with a 21-frame stride</figcaption></figure>')
    return f"""<h2 class="part">Streaming, chunk by chunk</h2>
<section class="scene" id="chunked">
<header><h2>Boulevard, 540p, six chunks of 22 frames</h2><p class="blurb">The release has no entry that takes a frame-aligned control and streams, so this borrows its anchored structure-edit recipe: each chunk is a video-edit conditioned on the anchor and that chunk's proxy, with the last frame of the previous chunk pinned into the first latent slot. 21 new frames (0.88 s) per chunk, four denoiser steps, one H200, one resident process. Seams are as continuous as the frames inside a chunk; what drifts is texture: the road coarsens chunk by chunk because each chunk restarts from one frame. Carrying the release's latent history softens it; longer chunks nearly remove it, and one 124-frame call is both the cleanest and, per frame, the fastest.</p></header>
<div class="row">{''.join(vids)}</div>
<table class="timing"><thead><tr><th>configuration</th><th>conditioning s</th><th>generate s</th><th>of which DiT (4 steps)</th><th>decode s</th><th>peak GB</th><th>× slower than real time</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p class="meta">medians over the chunks after the first (single-call rows: that one call). Conditioning is the Qwen3-VL-32B text encoder reading the prompt, the anchor and the chunk's proxy; as released it is moved CPU→GPU→CPU every chunk. The staged runs move eight DiT blocks and the VAE decoder around every VAE phase. Pinning both on a 141 GB card removes 60 s per chunk; the four DiT steps at 3.35 s each are what is left.</p>
</section>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", default="out/gallery")
    ap.add_argument("--videos", required=True, help="dir holding gal_<scene>_lora.mp4 / _nolora.mp4")
    ap.add_argument("--dst", default="docs/gallery")
    ap.add_argument("--lynn", default=None, help="dir holding <scene>_game.mp4 from LynnReal-Omni video-edit (optional fourth column)")
    ap.add_argument("--chunk-runs", default=None, help="dir holding boulevard_540_<tag>/{take.mp4,timing.json} from lynnreal/chunk_drive.py (optional streaming section)")
    ap.add_argument("--chunk-proxy", default=None, help="the 540p proxy control video those runs were driven by")
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
        if a.lynn and os.path.exists(f"{a.lynn}/{n}_game.mp4"):
            sz, _ = transcode(f"{a.lynn}/{n}_game.mp4", f"{a.dst}/video/{n}_lynn.mp4"); total += sz
            have["lynn"] = True
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
  {vid("lynn", "LynnReal-Omni video-edit, zero-shot — the proxy as its frame-aligned control video, no LoRA") if have.get("lynn") else ""}
</div>
<p class="meta">frame 0 vs anchor — subject x {f0['cx']} / {t['cx']}, y {f0['cy']} / {t['cy']}, lens→subject {f0['z8']} / {t['z8']} m, sky {f0['sky']}; {nf} frames, conditions clean</p>
</section>""")
    # Long takes, generated the authors' way: 124 + 90 per window with Retake34 overlap,
    # stitched by portb/stitch.py. Looked for beside the gallery as out/gallery_4w.
    longs = []
    lg = f"{os.path.dirname(a.gallery.rstrip('/'))}/gallery_4w" if os.path.dirname(a.gallery.rstrip('/')) else "out/gallery_4w"
    for n in ORDER:
        d = f"{lg}/{n}"
        if not os.path.isdir(d):
            continue
        meta = json.load(open(f"{d}/meta.json"))
        sz, nf = proxy_video(f"{d}/conditions", f"{a.dst}/video/{n}_4w_proxy.mp4"); total += sz
        have = {}
        for mode in ("nolora", "lora"):
            src = f"{a.videos}/gal4w_{n}_{mode}.mp4"
            if os.path.exists(src):
                sz, _ = transcode(src, f"{a.dst}/video/{n}_4w_{mode}.mp4"); total += sz
                have[mode] = True
        vid = lambda k, cap: (f'<figure><video controls muted loop playsinline preload="metadata" src="video/{n}_4w_{k}.mp4"></video><figcaption>{cap}</figcaption></figure>'
                              if (k == "proxy" or have.get(k)) else f'<figure class="missing"><div>not rendered</div><figcaption>{cap}</figcaption></figure>')
        title = BLURB[n][0]
        longs.append(f"""<section class="scene" id="{n}-long">
<header><h2>{title} — {meta['windows']} windows, {nf} frames ({nf/24:.1f} s)</h2><p class="blurb">The same scene carried on: {meta['windows']} windows of 124 frames with 90 new frames each, Retake34 overlap pinned, one caption compiled per window, stitched with the overlap dropped.</p></header>
<div class="row">
  {vid("proxy", f"our proxy, {nf} frames")}
  {vid("nolora", "MiniMax-H3, native reference mode (no LoRA)")}
  {vid("lora", "MiniMax-H3 + CWM LoRA")}
</div></section>""")
    long_html = ("<h2 class=\"part\">Long takes, the authors' way</h2>" + "".join(longs)) if longs else ""
    chunk_html = chunk_section(a.chunk_runs, a.chunk_proxy, a.dst) if a.chunk_runs else ""
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gallery — does the LoRA read the proxy?</title>
<style>
:root{{--bg:#f3f2ee;--panel:#fff;--ink:#1c1b1f;--mute:#5f5e66;--line:#dcdad3;--mono:ui-monospace,Menlo,Consolas,monospace;--sans:system-ui,-apple-system,"Segoe UI",sans-serif}}
@media (prefers-color-scheme:dark){{:root{{--bg:#17181d;--panel:#1f2027;--ink:#e8e7ee;--mute:#a4a3ad;--line:#33343d}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:15px;line-height:1.5;padding:32px clamp(16px,4vw,48px) 64px}}
main{{max-width:1240px;margin:0 auto}} h1{{font-size:clamp(22px,3vw,30px);margin:0 0 6px}} h2{{font-size:19px;margin:0 0 4px}}
.lede{{color:var(--mute);max-width:80ch;margin:0 0 28px}} h2.part{{font-size:22px;margin:36px 0 14px}} .blurb{{color:var(--mute);max-width:90ch;margin:0 0 12px}}
.scene{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-bottom:26px}}
.row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:12px}}
figure{{margin:0}} video,img{{display:block;width:100%;max-width:100%;border-radius:8px;background:#000;aspect-ratio:672/384}}
figure video[src$="_proxy.mp4"]{{aspect-ratio:672/768}}
figcaption,.meta{{font:12.5px var(--mono);color:var(--mute);margin-top:6px}}
table.timing{{border-collapse:collapse;width:100%;margin-top:14px;font:13px var(--mono);font-variant-numeric:tabular-nums}} .timing th,.timing td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}} .timing td:not(:first-child),.timing th:not(:first-child){{text-align:right;white-space:nowrap}}
#chunked .row{{grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}} #chunked video{{aspect-ratio:960/544}} .missing div{{aspect-ratio:672/384;border:1px dashed var(--line);border-radius:8px;display:grid;place-items:center;color:var(--mute)}}
</style></head><body><main>
<h1>Does the LoRA read the proxy?</h1>
<p class="lede">Six scenes. Each starts from a released first frame (for its style) and everything after frame 0 is ours: the geometry, the actions, the camera, and the caption compiled from the same simulation. MiniMax-H3 then generates each scene twice with the same anchor, proxy video, caption and seed — once in its native reference mode, once with the CWM LoRA. The only variable is whether the model was taught to read depth and semantics. Where a fourth video is shown, it is LynnReal-Omni (a 32B H3-architecture model with native frame-aligned controls) given the same proxy as a control video, zero-shot: no LoRA, the same anchor, the same caption content rewritten into its six-section prompt.</p>
{''.join(cards)}
{long_html}
{chunk_html}
<p class="lede">Made with <code>examples/gallery.py</code> and <code>docs/mkgallery.py</code>. Videos re-encoded for the web at 672 px; the conditions themselves are 336×192 float32 depth + uint8 semantic ids.</p>
</main></body></html>"""
    open(f"{a.dst}/index.html", "w").write(html)
    print(f"{len(cards)} scenes, {total/1e6:.1f} MB of video -> {a.dst}/index.html")


if __name__ == "__main__":
    main()

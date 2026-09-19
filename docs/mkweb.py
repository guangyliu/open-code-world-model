"""Re-encode the release videos to web size for the project page.

The originals are 1344x768 and total ~57 MB, which is more than a repository should carry
for what is essentially a gallery. Halving the long edge and raising the CRF gets the whole
set under ~8 MB with no visible loss at the size they are actually played.

Run again after any regeneration; it is idempotent and skips nothing, so the page always
matches the current videos.
"""
import os
import sys

import av

SRC = "/mnt/weka/home/guangyi.liu/cwm/portb/out"
DST = sys.argv[1] if len(sys.argv) > 1 else "/mnt/weka/home/guangyi.liu/cwm/release_draft/video"
WIDE = 960          # long edge; the page never shows a tile wider than 640 css px
CRF = "30"

os.makedirs(DST, exist_ok=True)
names = ["v3"] + [f"{a}_ex{n}" for n in (18, 28, 30, 31, 34, 36) for a in ("rel", "our")]

total_in = total_out = 0
for name in names:
    src = f"{SRC}/{name}.mp4"
    if not os.path.exists(src):
        print(f"  MISSING {src}"); continue
    dst = f"{DST}/{name}.mp4"
    i = av.open(src)
    istream = i.streams.video[0]
    w, h = istream.codec_context.width, istream.codec_context.height
    ow = WIDE if w >= h else int(round(WIDE * w / h / 2)) * 2
    oh = int(round(ow * h / w / 2)) * 2
    o = av.open(dst, mode="w")
    # faststart so the browser can begin playing before the whole file arrives; the page
    # autoplays several of these at once and without it the first frames stall.
    os_ = o.add_stream("libx264", rate=int(istream.average_rate or 24))
    os_.width, os_.height, os_.pix_fmt = ow, oh, "yuv420p"
    os_.options = {"crf": CRF, "preset": "slow", "movflags": "+faststart"}
    n = 0
    for frame in i.decode(video=0):
        frame = frame.reformat(width=ow, height=oh, format="yuv420p")
        for pkt in os_.encode(frame):
            o.mux(pkt)
        n += 1
    for pkt in os_.encode():
        o.mux(pkt)
    o.close(); i.close()
    a, b = os.path.getsize(src), os.path.getsize(dst)
    total_in += a; total_out += b
    print(f"  {name:12s} {w}x{h} -> {ow}x{oh}  {n:3d}f  {a/1e6:5.1f} -> {b/1e6:4.1f} MB")

print(f"\ntotal {total_in/1e6:.1f} MB -> {total_out/1e6:.1f} MB into {DST}")

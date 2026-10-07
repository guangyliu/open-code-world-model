"""Turn a gallery scene into inputs for LTX-2.5's Layout-To-Render IC-LoRA: a grey clay
playblast of our simulation, the anchor as the art-direction first frame, and a short
prompt.

    CWM_EXAMPLES=... python examples/ltx_inputs.py boulevard [stream ...]

Writes out/ltx/<name>/{clay.mp4, first_frame.png, prompt.txt, clay_000.png}. The clay is
the same simulation and camera the gallery renders as depth + class ids, shaded grey
instead (`render.render_clay`). Vehicles are emitted solid: the wireframe is a convention
of the CWM proxy, and a model reading grey geometry would render twelve bars as a frame.

The LoRA card asks for one or two sentences that describe the first frame, and for none of
"clay", "3D" or "Unreal". The prompt is assembled from the scene's own content table.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("CWM_SOLID_VEHICLES", "1")      # before any worker imports compile

import av
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gallery as G                                           # noqa: E402
from compiler.render import Camera, render_clay               # noqa: E402
from world import compile as wc                               # noqa: E402
from world.gallery_content import CONTENT                     # noqa: E402
from world.state import WorldState                            # noqa: E402

# 480p-class output. 896x512 rather than 854x480: the LoRA needs both sides divisible by 64,
# and 1.75 is exactly the aspect of the 336x192 proxy, so the camera's field of view is the
# one the scenes were composed under. Rendered at the final (stage-2) size; the pipeline
# resizes the reference down for stage 1 (448x256).
W, H = int(os.environ.get("CWM_LTX_W", 896)), int(os.environ.get("CWM_LTX_H", 512))
OUT = Path(os.environ.get("CWM_LTX_OUT", "out/ltx"))


def _clay_one(job):
    k, sj, cam_t = job
    st = WorldState.from_json(sj)
    focal, pos, pitch, yaw, roll = cam_t
    cam = Camera(focal_px=focal, position=np.array(pos), pitch_deg=pitch, yaw_deg=yaw, roll_deg=roll)
    return k, render_clay(cam, wc.PrimitiveCache().build(st), width=W, height=H)


def write_mp4(frames, path, fps=24):
    c = av.open(str(path), mode="w")
    s = c.add_stream("libx264", rate=fps)
    s.width, s.height, s.pix_fmt = W, H, "yuv420p"
    s.options = {"crf": "12", "preset": "slow"}
    for f in frames:
        for p in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")):
            c.mux(p)
    for p in s.encode():
        c.mux(p)
    c.close()


def prompt_for(name: str) -> str:
    c = CONTENT[name]
    first = lambda t: re.split(r"(?<=[.])\s+", t.strip())[0]
    return f"{first(c['identity'])} The camera follows the protagonist across {c['place']}. {first(c['scene'])}"


def run(name: str) -> int:
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    plan, s, inv, sc = G.simulate(name, check=True)
    if inv.violations:
        print(f"{name}: invariants violated; not rendering")
        return 1
    import multiprocessing as mp
    t0 = time.time()
    n = int(os.environ.get("CWM_WORKERS", 24))
    with mp.get_context("spawn").Pool(n, initializer=G._init_worker) as pool:
        frames = dict(pool.imap_unordered(_clay_one, plan, chunksize=1))
    frames = [frames[k] for k in sorted(frames)]
    write_mp4(frames, out / "clay.mp4")
    Image.fromarray(frames[0]).save(out / "clay_000.png")
    shutil.copy(G.EXAMPLES / str(sc["anchor"]) / "first_frame.png", out / "first_frame.png")
    (out / "prompt.txt").write_text(prompt_for(name) + "\n", encoding="utf-8")
    print(f"{name}: {len(frames)} clay frames {W}x{H} in {time.time() - t0:.1f}s -> {out}")
    return 0


def main() -> int:
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(G.SCENES)
    bad = [n for n in names if n not in G.SCENES]
    if bad:
        print(f"unknown scene(s) {bad}; valid: {', '.join(G.SCENES)}", file=sys.stderr)
        return 2
    rc = 0
    for n in names:
        rc |= run(n)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

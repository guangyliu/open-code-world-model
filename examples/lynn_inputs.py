"""Turn a gallery scene into LynnReal-Omni video-edit inputs: a 1344x768 control video
rendered from our conditions (a flat-shaded "game render": class colour shaded by depth,
or depth alone), the anchor at the target canvas, and a six-section prompt in their
format compiled from the same caption our simulation produced.

    CWM_SOLID_VEHICLES=1 CWM_EXAMPLES=... python examples/gallery.py boulevard   # conditions
    python examples/lynn_inputs.py out/gallery/boulevard /path/to/lynn/inputs/boulevard

The prompt is a rewrite, not new authorship: subject_definitions and detailed_description
are lifted from our compiled caption (identity, scene, camera sentence and the timestamped
timeline), so what LynnReal is told is what H3 was told.
"""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image

COL = {0: (92, 92, 110), 1: (150, 200, 245), 2: (60, 130, 220), 4: (120, 110, 100),
       5: (55, 145, 70), 8: (255, 70, 70), 9: (255, 170, 40), 10: (250, 225, 60)}
LUT = np.zeros((256, 3), np.float32)
for k, v in COL.items():
    LUT[k] = v
W, H = 1344, 768


def write(frames, path):
    c = av.open(str(path), mode="w")
    s = c.add_stream("libx264", rate=24)
    s.width, s.height, s.pix_fmt = W, H, "yuv420p"
    s.options = {"crf": "12", "preset": "slow"}
    for f in frames:
        for p in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")):
            c.mux(p)
    for p in s.encode():
        c.mux(p)
    c.close()


def control_videos(cond: Path, dst: Path):
    game, depth = [], []
    for p in sorted(glob.glob(str(cond / "*.semantic_id.png"))):
        a = np.array(Image.open(p)); a = a[..., 0] if a.ndim == 3 else a
        z = np.fromfile(p.replace("semantic_id.png", "depth.f32"), "<f4").reshape(192, 336)
        t = np.clip(np.log(np.maximum(z, 0.3) / 0.3) / np.log(256 / 0.3), 0, 1)
        shade = (1.0 - 0.65 * t)[..., None]
        rgb = np.clip(LUT[a] * np.where((a == 1)[..., None], 1.0, shade), 0, 255).astype(np.uint8)
        game.append(np.array(Image.fromarray(rgb).resize((W, H), Image.BILINEAR)))
        g = (255 * (1 - t)).astype(np.uint8); g[z <= 0] = 0
        depth.append(np.array(Image.fromarray(np.stack([g, g, g], -1)).resize((W, H), Image.BILINEAR)))
    write(game, dst / "proxy_game.mp4")
    write(depth, dst / "proxy_depth.mp4")
    return len(game)


def six_sections(caption: str) -> str:
    """Our three-section caption -> LynnReal's six sections, same facts."""
    body = caption.split("Appearance and scene:", 1)[1]
    scene_block, rest = body.split("Protagonist action timeline:", 1)
    timeline, others = rest.split("Other visible actors:", 1)
    paras = [p.strip() for p in scene_block.strip().split("\n\n") if p.strip()]
    identity = paras[0]
    cam = next((s for s in re.split(r"(?<=[.])\s+", " ".join(paras)) if "camera" in s.lower()), "")
    setting = " ".join(p for p in paras[1:] if "camera" not in p.lower()) or paras[-1]
    subject = identity.split(".")[0] + "."
    return f"""subject_definitions:
<Picture 1> defines the target appearance: {subject} {setting}
<Video 1> defines the source scene geometry and its evolving camera and object motion: a flat-shaded structural render in which the red figure is the protagonist, yellow bodies are vehicles, grey ground is the walkable surface, green masses are vegetation, blue is water, and the lens motion is the shot's camera.

summary:
[video editing] Render the source action in the materials, lighting and visual style of <Picture 1>, preserving every position, silhouette, contact and camera move of <Video 1>.

retention_analysis:
<Picture 1>: attribute_transfer - transfer the protagonist's appearance, the ground, walls, vegetation, vehicles and lighting of the picture onto the corresponding source surfaces; the picture is the exact first frame.
<Video 1>: partially_preserved - preserve the protagonist's silhouette, position and timing, every other actor's path, relative scale, contacts and occlusion order, and the camera trajectory; replace the flat proxy colours and box-like shapes with real materials and real bodies.

detailed_description:
[Shot 1] {cam} {identity} {timeline.strip()} Other visible actors: {others.strip()} Textures stay attached to surfaces as the camera and objects move; shadows follow actual contacts.

overall_soundscape:
Quiet environmental ambience and contact sounds that follow the visible action. No dialogue.

non_diegetic_music:
N/A
"""


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    n = control_videos(src / "conditions", dst)
    Image.open(src / "first_frame.png").convert("RGB").resize((W, H)).save(dst / "anchor.png")
    cap = (src / "prompts" / "000.txt").read_text(encoding="utf-8")
    (dst / "prompt_edit.txt").write_text(six_sections(cap), encoding="utf-8")
    print(f"{src.name}: {n} frames -> {dst}")


if __name__ == "__main__":
    main()

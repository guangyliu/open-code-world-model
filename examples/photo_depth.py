"""Per-photo geometry for a photo-anchored clay proxy: metric depth, focal length, and
person / sky masks. Runs on GPU in the LTX venv (needs torch + transformers).

    python examples/photo_depth.py /path/to/img out/photo

Writes out/photo/<stem>/{image.png, geom.npz} with
    depth   float32 (H, W) metres, Depth Pro
    focal   focal length in pixels at the photo's resolution, Depth Pro's own estimate
    labels  uint8 (H, W) ADE20K class ids from Mask2Former
    person, sky   the ADE ids of those two classes
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (AutoImageProcessor, DepthProForDepthEstimation,
                          Mask2FormerForUniversalSegmentation)

DEPTH = "apple/DepthPro-hf"
SEG = "facebook/mask2former-swin-large-ade-semantic"


def main() -> int:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dev = torch.device("cuda")
    dp = AutoImageProcessor.from_pretrained(DEPTH)
    dm = DepthProForDepthEstimation.from_pretrained(DEPTH, torch_dtype=torch.float16).to(dev).eval()
    sp = AutoImageProcessor.from_pretrained(SEG)
    sm = Mask2FormerForUniversalSegmentation.from_pretrained(SEG).to(dev).eval()
    lab = {v: int(k) for k, v in sm.config.id2label.items()}
    for path in sorted(p for p in src.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")):
        im = Image.open(path).convert("RGB")
        H, W = im.height, im.width
        with torch.no_grad():
            x = dp(images=im, return_tensors="pt").to(dev)
            out = dm(pixel_values=x["pixel_values"].half())
            post = dp.post_process_depth_estimation(out, target_sizes=[(H, W)])[0]
            depth = post["predicted_depth"].float().cpu().numpy()
            focal = float(post["focal_length"])
            y = sp(images=im, return_tensors="pt").to(dev)
            labels = sp.post_process_semantic_segmentation(sm(**y), target_sizes=[(H, W)])[0]
            labels = labels.cpu().numpy().astype(np.uint8)
        o = dst / path.stem
        o.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, o / "image.png")
        np.savez_compressed(o / "geom.npz", depth=depth.astype(np.float32), focal=focal, labels=labels,
                            person=lab["person"], sky=lab["sky"])
        print(f"{path.stem}: {W}x{H} focal {focal:.0f}px  depth p5/p50/p95 "
              f"{np.percentile(depth, 5):.1f}/{np.median(depth):.1f}/{np.percentile(depth, 95):.1f} m  "
              f"person {(labels == lab['person']).mean():.3f} sky {(labels == lab['sky']).mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

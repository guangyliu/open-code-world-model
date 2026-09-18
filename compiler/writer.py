"""Write rendered proxy frames in the exact layout the released inference consumes.

Contract from INFERENCE.md and cwm_h3_inference/duv.py:
  <condition_root>/NNNNNN.depth.f32        headerless C-order little-endian float32,
                                           192x336 metres, exactly 258048 bytes
  <condition_root>/NNNNNN.semantic_id.png  8-bit grayscale L, 336x192, values in [0, 11]
Windows read 124 consecutive ordinals starting at `start_frame`, advancing by 90.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

WINDOW_FRAMES = 124
STRIDE_FRAMES = 90
EXPECTED_BYTES = 336 * 192 * 4
FAR_M = 256.0


def write_frame(root: Path, ordinal: int, depth_m: np.ndarray, semantic: np.ndarray) -> dict:
    """Validate everything first, then write both files. Returns a small report.

    Order matters: writing the depth before checking the semantic contract left a
    .depth.f32 on disk with no matching .png, which verify() then reported as a missing
    ordinal rather than as the contract violation it was -- and a stale .png from an
    earlier run would have paired with it and passed.
    """
    if depth_m.shape != (192, 336) or semantic.shape != (192, 336):
        raise ValueError(f"frame must be 192x336, got {depth_m.shape} / {semantic.shape}")
    # Range-check BEFORE the uint8 cast: 256 wraps to 0 and would be written as
    # void_unknown without complaint.
    if int(np.min(semantic)) < 0 or int(np.max(semantic)) > 11:
        raise ValueError(f"semantic IDs must be in [0, 11], got "
                         f"[{int(np.min(semantic))}, {int(np.max(semantic))}]")
    d = np.ascontiguousarray(depth_m, dtype="<f4")
    if not np.isfinite(d).all() or (d < 0).any():
        raise ValueError("depth must be finite and non-negative")
    # SPEC section 2: the released float32 producer does NOT clip above the far plane.
    # We clamp because the fixed-log encoder maps anything >= FAR to code 0, the
    # invalid/sky code -- so the clamped fraction is reported rather than hidden.
    far_frac = float((d > FAR_M).mean())
    if far_frac:
        d = np.minimum(d, np.float32(FAR_M))
    payload = d.tobytes()
    if len(payload) != EXPECTED_BYTES:
        raise ValueError(f"expected {EXPECTED_BYTES} bytes, got {len(payload)}")

    s = np.ascontiguousarray(semantic, dtype=np.uint8)
    sky = s == 1
    if (d[sky] != 0).any() or (d[~sky] <= 0.001).any():
        raise ValueError("depth must be zero exactly on sky pixels and positive elsewhere")

    root.mkdir(parents=True, exist_ok=True)
    (root / f"{ordinal:06d}.depth.f32").write_bytes(payload)
    Image.fromarray(s, "L").save(root / f"{ordinal:06d}.semantic_id.png")
    return {"far_clamped_frac": far_frac}


def window_starts(n_frames: int) -> list[int]:
    """Start ordinals for a take of `n_frames`; the released rule is 124 + 90*(k-1)."""
    if n_frames < WINDOW_FRAMES or (n_frames - WINDOW_FRAMES) % STRIDE_FRAMES:
        raise ValueError(f"{n_frames} frames is not 124 + 90*k")
    return list(range(0, n_frames - WINDOW_FRAMES + 1, STRIDE_FRAMES))


def write_config(path: Path, *, condition_root: str, anchor_image: str, prompts: list[str],
                 seed: int, cache_dir: str, output: str, models: dict) -> None:
    starts = [i * STRIDE_FRAMES for i in range(len(prompts))]
    cfg = {
        "models": models,
        "anchor_image": anchor_image,
        "condition_root": condition_root,
        "cache_dir": cache_dir,
        "output": output,
        "windows": [{"prompt": p, "start_frame": s} for p, s in zip(prompts, starts)],
        "seed": seed,
        "steps": 20,
        "attention_mode": "flash3",
        "blocks_to_swap": 48,
        "expected_lora_modules": 200,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def verify(root: Path, n_frames: int) -> dict:
    """Re-read the written bytes and assert the contract on them.

    Checking only file size and PNG mode asserted far less than the verdicts implied: it
    could not catch a semantic ID out of range, a sky pixel with nonzero depth, or a file
    left over from an earlier run.
    """
    bad = []
    for i in range(n_frames):
        f = root / f"{i:06d}.depth.f32"
        q = root / f"{i:06d}.semantic_id.png"
        if not f.exists() or not q.exists():
            bad.append(f"missing ordinal {i}")
            continue
        if f.stat().st_size != EXPECTED_BYTES:
            bad.append(f"{f.name}: {f.stat().st_size} bytes")
            continue
        d = np.fromfile(f, dtype="<f4").reshape(192, 336)
        with Image.open(q) as im:
            if im.mode != "L" or im.size != (336, 192):
                bad.append(f"{q.name}: {im.mode}/{im.size}")
                continue
            sem = np.asarray(im, dtype=np.uint8)
        if not np.isfinite(d).all() or (d < 0).any():
            bad.append(f"{f.name}: non-finite or negative depth")
        if sem.max() > 11:
            bad.append(f"{q.name}: semantic ID {int(sem.max())} out of range")
        sky = sem == 1
        if (d[sky] != 0).any():
            bad.append(f"ordinal {i}: sky pixel with nonzero depth")
        if (d[~sky] <= 0.001).any():
            bad.append(f"ordinal {i}: non-sky pixel with invalid depth")
        if len(bad) > 20:
            bad.append("... further errors suppressed")
            break
    extra = sorted(x.name for x in root.glob("*.depth.f32") if int(x.name[:6]) >= n_frames)
    bad += [f"unexpected ordinal {e}" for e in extra[:5]]
    return {"frames": n_frames, "windows": len(window_starts(n_frames)), "errors": bad}

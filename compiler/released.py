"""Read released CWM example conditions, whichever family they are in.

Both families store float32 arrays, so the dtype tells you nothing. `metadata.json` carries
`depth_format`, and that is the only correct discriminator. This module reads it and hands
back metric depth either way, so callers never have to care.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .render import decode_gray16

METRIC = "float32_le_metric_first_surface_metres"
GRAY16 = "uint16_le_fixed_log_gray16_codes"


def family(example_dir) -> str:
    """"metric" or "gray16", from metadata.json. Never inferred from the array."""
    fmt = json.loads((Path(example_dir) / "metadata.json").read_text())["depth_format"]
    if fmt == METRIC:
        return "metric"
    if fmt == GRAY16:
        return "gray16"
    raise ValueError(f"unknown depth_format {fmt!r} in {example_dir}")


def read_chunk(example_dir, chunk: str = "condition/000.npz"):
    """Return (depth_metres, semantic_ids, family) for one condition chunk.

    Depth comes back in metres for both families: gray16 codes are decoded on the way out,
    so a caller comparing examples across families is comparing the same quantity.
    """
    d = Path(example_dir)
    fam = family(d)
    z = np.load(d / chunk)
    depth, ids = z["depth"], z["semantic_id"]
    if fam == "gray16":
        depth = decode_gray16(depth).astype(np.float32)
    return depth, ids, fam

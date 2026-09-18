"""Step 4: class occupancy, depth distribution, and sky/invalid encoding conventions."""
from __future__ import annotations

import numpy as np

from .proxy import CLASSES, SKY, VALID_EPS_M, VOID, Window


def class_occupancy(w: Window) -> np.ndarray:
    """Fraction of pixels per class, averaged over the window."""
    counts = np.bincount(w.semantic.reshape(-1), minlength=12).astype(np.float64)
    return counts / counts.sum()


def depth_by_class(w: Window) -> dict[int, np.ndarray]:
    """Percentiles of valid metric depth per class: [p1, p50, p99, max]."""
    valid = w.valid()
    out = {}
    for cid in range(12):
        sel = (w.semantic == cid) & valid
        if sel.sum() < 64:
            continue
        d = w.depth[sel]
        out[cid] = np.array([*np.percentile(d, [1, 50, 99]), d.max()])
    return out


def invalid_profile(w: Window) -> dict:
    """How invalid depth (<= 1mm) relates to semantic class -- reveals the sky convention."""
    invalid = ~w.valid()
    total_invalid = int(invalid.sum())
    per_class = np.bincount(w.semantic[invalid].reshape(-1), minlength=12) if total_invalid else np.zeros(12, int)
    sky = w.semantic == SKY
    return {
        "invalid_frac": total_invalid / w.depth.size,
        "invalid_by_class": per_class,
        "sky_frac": float(sky.mean()),
        "sky_invalid_frac": float(invalid[sky].mean()) if sky.any() else float("nan"),
        "sky_depth_p50": float(np.percentile(w.depth[sky & w.valid()], 50)) if (sky & w.valid()).any() else float("nan"),
        "void_frac": float((w.semantic == VOID).mean()),
    }

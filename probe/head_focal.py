"""Focal length from the head ellipsoid, with the shoulder contamination removed.

The first version of this estimator took the widest row in the top 2-18% of the
figure. On a rig whose shoulders sit at 0.82 of standing height that band reaches
the shoulders, and `max` is not robust: one shoulder row doubles the measured width
and therefore doubles f. Round-tripping against a synthetic scene exposed a +104%
bias. This version uses a tighter band and a percentile of row widths.
"""
from __future__ import annotations

import numpy as np

from . import proxy
from .geometry import blobs

BAND = (0.02, 0.12)
WIDTH_PCT = 75


def head_focal(depth: np.ndarray, semantic: np.ndarray, *, min_px: int = 150,
               min_h: int = 20, band=BAND, width_pct: float = WIDTH_PCT):
    out = []
    for v, u in blobs(semantic == proxy.HUMAN, min_px=min_px):
        y0, y1 = int(v.min()), int(v.max())
        h = y1 - y0 + 1
        if h < min_h or y0 == 0:
            continue
        m = (v >= y0 + band[0] * h) & (v <= y0 + band[1] * h)
        if m.sum() < 15:
            continue
        du, dv, dd = u[m], v[m], depth[v[m], u[m]].astype(np.float64)
        if dd.min() <= 0.5:
            continue
        widths = [int(du[dv == r].max() - du[dv == r].min() + 1) for r in np.unique(dv)]
        w_px = float(np.percentile(widths, width_pct))
        span = float(np.percentile(dd, 98) - np.percentile(dd, 2))
        if w_px < 3 or span <= 1e-3:
            continue
        out.append({"focal_px": w_px * float(np.median(dd)) / (2 * span),
                    "head_D_m": 2 * span, "z": float(np.median(dd)), "fig_h_px": float(h)})
    return out

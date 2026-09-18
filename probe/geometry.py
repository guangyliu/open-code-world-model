"""Steps 2-3: recover the proxy camera's intrinsics and ground-plane pose.

The released conditions give metric depth but no camera parameters. A pinhole
camera with a centred principal point and square pixels has one free intrinsic,
the focal length f. We recover it by exploiting the fact that road_paved pixels
lie on a physical plane: back-project them for a candidate f and keep the f whose
back-projection is flattest. The same fit then yields camera height and pitch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .proxy import H, W


@dataclass
class PlaneFit:
    focal_px: float
    rms_m: float          # RMS point-to-plane residual at this focal length
    normal: np.ndarray    # unit plane normal in camera coords, pointing "up"
    height_m: float       # camera distance to the plane
    pitch_deg: float      # downward tilt of the optical axis below the plane
    roll_deg: float
    n_points: int


def backproject(u: np.ndarray, v: np.ndarray, d: np.ndarray, focal_px: float, *, euclidean: bool) -> np.ndarray:
    """Pixel centres + depth -> camera-space XYZ. `euclidean` selects range vs z-depth."""
    x = (u + 0.5 - W / 2.0) / focal_px
    y = (v + 0.5 - H / 2.0) / focal_px
    ray = np.stack([x, y, np.ones_like(x)], axis=-1)
    if euclidean:
        ray = ray / np.linalg.norm(ray, axis=-1, keepdims=True)
    return ray * d[..., None]


def _fit_plane(pts: np.ndarray, iters: int = 5, keep: float = 0.85) -> tuple[np.ndarray, float, float, int]:
    """Trimmed least-squares plane fit; returns (unit normal, offset, rms, n_used)."""
    p = pts
    for _ in range(iters):
        c = p.mean(0)
        _, _, vt = np.linalg.svd(p - c, full_matrices=False)
        n = vt[-1]
        r = (pts - c) @ n
        thresh = np.quantile(np.abs(r), keep)
        p = pts[np.abs(r) <= thresh]
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    n = vt[-1]
    rms = float(np.sqrt(np.mean(((p - c) @ n) ** 2)))
    return n, float(-c @ n), rms, len(p)


def fit_ground(depth: np.ndarray, mask: np.ndarray, *, euclidean: bool,
               focal_range=(80.0, 1200.0), samples: int = 240,
               max_depth_m: float = 80.0, max_points: int = 60000,
               rng: np.random.Generator | None = None) -> PlaneFit | None:
    """Sweep focal length, keeping the value whose back-projected ground is flattest."""
    rng = rng or np.random.default_rng(0)
    sel = mask & (depth > 0.5) & (depth < max_depth_m)
    v, u = np.nonzero(sel)
    if len(u) < 2000:
        return None
    if len(u) > max_points:
        idx = rng.choice(len(u), max_points, replace=False)
        u, v = u[idx], v[idx]
    d = depth[v, u].astype(np.float64)

    best: PlaneFit | None = None
    for f in np.geomspace(*focal_range, samples):
        pts = backproject(u.astype(np.float64), v.astype(np.float64), d, f, euclidean=euclidean)
        n, off, rms, used = _fit_plane(pts)
        # Scale-invariant: a larger f inflates X/Y, so compare residual per unit depth.
        score = rms / d.mean()
        if best is None or score < best.rms_m:
            if n[1] < 0:            # make the normal point up (camera +Y is down)
                n, off = -n, -off
            pitch = float(np.degrees(np.arcsin(np.clip(-n[2], -1, 1))))
            roll = float(np.degrees(np.arctan2(n[0], -n[1])))
            best = PlaneFit(float(f), score, n, abs(off), pitch, roll, used)
    return best


def blobs(mask: np.ndarray, min_px: int = 120):
    """Connected components of a boolean mask (4-connectivity), as index arrays."""
    from scipy import ndimage

    lab, n = ndimage.label(mask)
    for k in range(1, n + 1):
        idx = np.nonzero(lab == k)
        if len(idx[0]) >= min_px:
            yield idx


def focal_from_round_primitive(depth: np.ndarray, mask: np.ndarray, *, min_px: int = 200):
    """Focal length implied by treating each blob as a rotationally symmetric primitive.

    Changing f rescales transverse (X, Y) extent but never the measured depth extent,
    so a primitive that is circular in plan view pins f down: its metric width equals
    twice the depth span of its visible front surface.
    """
    out = []
    for v, u in blobs(mask, min_px):
        d = depth[v, u].astype(np.float64)
        if d.min() <= 0.5:
            continue
        z = float(np.median(d))
        # Robust extents: percentiles resist single-pixel silhouette noise.
        w_px = float(np.percentile(u, 97.5) - np.percentile(u, 2.5))
        span = float(np.percentile(d, 97.5) - np.percentile(d, 2.5))
        h_px = float(np.percentile(v, 99) - np.percentile(v, 1))
        if w_px < 3 or span <= 1e-3:
            continue
        out.append({
            "z": z, "w_px": w_px, "h_px": h_px, "depth_span_m": span,
            "n_px": len(u), "focal_px": w_px * z / (2.0 * span),
        })
    return out


def human_parts(depth: np.ndarray, semantic: np.ndarray, *, human_id: int = 8, min_h: int = 30):
    """Per-figure capsule measurements, split into head / torso / legs row bands.

    Depth spans are absolutely metric and independent of focal length, so they are
    the right quantity for asking whether every example shares one rig. Pixel widths
    are not -- they scale with f -- so they are reported separately.
    """
    out = []
    for v, u in blobs(semantic == human_id, min_px=300):
        y0, y1 = int(v.min()), int(v.max())
        h = y1 - y0 + 1
        if h < min_h or y0 == 0:          # need an unclipped head
            continue
        d_all = depth[v, u].astype(np.float64)
        if d_all.min() <= 0.5:
            continue
        rec = {"z": float(np.median(d_all)), "fig_h_px": float(h), "n_px": len(u),
               "clipped_bottom": bool(y1 == depth.shape[0] - 1)}
        # Row bands as fractions of figure height, chosen from the rendered anatomy.
        for name, lo, hi in (("head", 0.02, 0.18), ("torso", 0.28, 0.50), ("leg", 0.72, 0.92)):
            m = (v >= y0 + lo * h) & (v <= y0 + hi * h)
            if m.sum() < 30:
                continue
            du, dv, dd = u[m], v[m], depth[v[m], u[m]].astype(np.float64)
            widest = max((dv == r).sum() and int(du[dv == r].max() - du[dv == r].min() + 1)
                         for r in np.unique(dv))
            rec[f"{name}_span_m"] = float(np.percentile(dd, 98) - np.percentile(dd, 2))
            rec[f"{name}_w_px"] = float(widest)
        out.append(rec)
    return out


def bar_thickness(depth: np.ndarray, mask: np.ndarray, focal_px: float, *, max_run: int = 40):
    """Metric cross-section of a wireframe's bars, from short runs along scanlines.

    A wireframe cage is mostly thin bars, so the run-length histogram of mask pixels
    along rows and columns is dominated by the bar cross-section. Runs touching the
    blob's bounding box are dropped: those cross a bar lengthwise, not across.
    """
    runs = []
    for axis in (0, 1):
        m = mask if axis == 0 else mask.T
        d = depth if axis == 0 else depth.T
        for r in range(m.shape[0]):
            row = m[r]
            if not row.any():
                continue
            idx = np.flatnonzero(np.diff(np.concatenate(([0], row.view(np.int8), [0]))))
            for a, b in zip(idx[::2], idx[1::2]):
                n = b - a
                if n < 2 or n > max_run or a == 0 or b == len(row):
                    continue
                z = float(np.median(d[r, a:b]))
                if z > 0.5:
                    runs.append(n * z / focal_px)
    return np.array(runs)


def blobs_depth_aware(mask: np.ndarray, depth: np.ndarray, *, min_px: int = 120,
                      rel_jump: float = 0.15, abs_jump_m: float = 0.5):
    """Connected components that also split at depth discontinuities.

    Plain 2-D labelling cannot separate two objects of the same class at different
    depths that merely touch in the image -- a near bush and a far treeline become one
    blob whose depth span is meaningless. Cutting the mask at large depth gradients
    first keeps each component on one surface.

    A jump counts as a boundary when it exceeds `abs_jump_m` AND `rel_jump` of the
    local depth, so the threshold scales with distance the way depth precision does.
    """
    from scipy import ndimage

    d = np.where(mask, depth, np.nan)
    cut = np.zeros_like(mask, dtype=bool)
    for axis in (0, 1):
        a = np.diff(d, axis=axis)
        base = np.minimum(np.take(d, range(d.shape[axis] - 1), axis=axis),
                          np.take(d, range(1, d.shape[axis]), axis=axis))
        with np.errstate(invalid="ignore"):
            big = (np.abs(a) > abs_jump_m) & (np.abs(a) > rel_jump * base)
        big = np.nan_to_num(big.astype(float), nan=0.0).astype(bool)
        pad = [(0, 0), (0, 0)]
        pad[axis] = (0, 1)
        cut |= np.pad(big, pad, constant_values=False)
        pad[axis] = (1, 0)
        cut |= np.pad(big, pad, constant_values=False)

    lab, n = ndimage.label(mask & ~cut)
    for k in range(1, n + 1):
        idx = np.nonzero(lab == k)
        if len(idx[0]) >= min_px:
            yield idx

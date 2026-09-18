"""Focal length from "people stand upright on the ground".

Needs no shape prior, unlike the head-ellipsoid estimator. Back-projection depends
on f, and it distorts the ground normal and the body axis differently, so the value
of f that makes a standing figure perpendicular to the ground it stands on is unique.

Validate on a synthetic render before trusting it on released data.
"""
from __future__ import annotations

import numpy as np

from . import proxy
from .geometry import blobs


def ground_coeffs(depth, road):
    """Affine fit of 1/z over road pixels. The coefficients do NOT depend on f:
    for z-depth, a plane always makes 1/z affine in (u, v). Fit once, sweep f after."""
    m = road & (depth > 0.5) & (depth < 60)
    v, u = np.nonzero(m)
    if len(u) < 1500:
        return None
    q = 1.0 / depth[v, u].astype(np.float64)
    uu, vv = u + 0.5 - proxy.W / 2.0, v + 0.5 - proxy.H / 2.0
    A = np.stack([uu, vv, np.ones_like(uu, dtype=float)], 1)
    coef = None
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(A, q, rcond=None)
        with np.errstate(all="ignore"):
            r = np.abs(q - A @ coef)
        keep = r <= np.quantile(r, 0.85)
        A, q = A[keep], q[keep]
    return coef


def _normal_from_coeffs(coef, f):
    a, b, c = coef
    off = 1.0 / np.sqrt(f * f * (a * a + b * b) + c * c)
    n = -np.array([a * f * off, b * f * off, c * off])
    return n / np.linalg.norm(n) * (-1.0 if n[1] > 0 else 1.0)


def body_scatter(depth, human):
    """3x3 scatter of the figure in f-free coordinates (X, Y, Z) = (z*u', z*v', z).

    Back-projection is P = diag(1/f, 1/f, 1) @ (X, Y, Z), so the covariance at any f
    is S C0 S with S = diag(1/f, 1/f, 1). Computing C0 once turns the focal sweep from
    one PCA over every pixel into one 3x3 eigendecomposition per sample.
    """
    v, u = np.nonzero(human)
    if len(u) < 200:
        return None
    z = depth[v, u].astype(np.float64)
    X = z * (u + 0.5 - proxy.W / 2.0)
    Y = z * (v + 0.5 - proxy.H / 2.0)
    P = np.stack([X, Y, z], 1)
    P -= P.mean(0)
    return P.T @ P / len(P)


def _axis_from_scatter(c0, f):
    s = np.array([1.0 / f, 1.0 / f, 1.0])
    c = c0 * np.outer(s, s)
    w, vec = np.linalg.eigh(c)
    a = vec[:, int(np.argmax(w))]
    return a * (-1.0 if a[1] > 0 else 1.0)


def focal_from_upright(depth, semantic, *, focal_range=(60.0, 600.0), samples=420,
                       min_blob=400):
    """Sweep f; keep the value where the body axis is most parallel to the ground normal."""
    road = semantic == proxy.ROAD
    bs = sorted(blobs(semantic == proxy.HUMAN, min_px=min_blob), key=lambda b: -len(b[0]))
    if not bs or road.sum() < 1500:
        return None
    v, u = bs[0]
    human = np.zeros_like(road)
    human[v, u] = True
    if depth[human].min() <= 0.5:
        return None
    coef = ground_coeffs(depth, road)
    c0 = body_scatter(depth, human)
    if coef is None or c0 is None:
        return None
    best = None
    for f in np.geomspace(*focal_range, samples):
        n = _normal_from_coeffs(coef, f)
        a = _axis_from_scatter(c0, f)
        ang = np.degrees(np.arccos(np.clip(abs(a @ n), -1, 1)))
        if best is None or ang < best[1]:
            best = (float(f), float(ang))
    return best

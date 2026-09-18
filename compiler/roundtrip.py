"""Acceptance test: measure our own render with the probe estimators.

The estimators in probe/geometry.py were built blind against the released examples.
Running them on a scene whose ground truth we set is the only way to tell an
estimator's bias apart from a genuine property of the released data.
"""
from __future__ import annotations

import numpy as np

from probe import geometry, proxy
from . import rig
from .render import Camera, Plane, render_frame, wireframe_box


def build_scene(cam_f=206.0, cam_h=1.67, pitch=-8.0, human_z=5.0, box_at=(4.0, 15.0)):
    cam = Camera(focal_px=cam_f, position=np.array([0.0, -cam_h, 0.0]), pitch_deg=pitch)
    prims = [Plane(np.zeros(3), np.array([0.0, -1.0, 0.0]), class_id=proxy.ROAD)]
    half = np.array([2.2, 1.0, 2.4])
    prims += wireframe_box([box_at[0], -half[1], box_at[1]], half)
    prims += rig.Human(position=np.array([0.0, 0.0, human_z]), facing_deg=180.0,
                       pose=rig.walk_pose(0.2)).primitives()
    return cam, prims, half


def measure(z, ids, f_assumed):
    """Recover focal, camera pose and bar ratio the same way the probe did."""
    out = {}
    # focal from the head ellipsoid (SPEC 6)
    fs = []
    for v, u in geometry.blobs(ids == proxy.HUMAN, min_px=150):
        y0, y1 = int(v.min()), int(v.max())
        h = y1 - y0 + 1
        if h < 20 or y0 == 0:
            continue
        m = (v >= y0 + 0.02 * h) & (v <= y0 + 0.18 * h)
        if m.sum() < 20:
            continue
        du, dv, dd = u[m], v[m], z[v[m], u[m]].astype(np.float64)
        widest = max(int(du[dv == r].max() - du[dv == r].min() + 1) for r in np.unique(dv))
        span = float(np.percentile(dd, 98) - np.percentile(dd, 2))
        if span > 1e-3:
            fs.append(widest * float(np.median(dd)) / (2 * span))
            out["head_D_m"] = 2 * span
            out["fig_h_px"] = float(h)
    out["focal_px"] = float(np.median(fs)) if fs else np.nan

    # camera height and pitch from the road plane (SPEC 6.2)
    m = (ids == proxy.ROAD) & (z > 0.5) & (z < 60)
    v, u = np.nonzero(m)
    q = 1.0 / z[v, u].astype(np.float64)
    uu, vv = u + 0.5 - proxy.W / 2.0, v + 0.5 - proxy.H / 2.0
    A = np.stack([uu, vv, np.ones_like(uu, dtype=float)], 1)
    coef, *_ = np.linalg.lstsq(A, q, rcond=None)
    a, b, c = coef
    off = 1.0 / np.sqrt(f_assumed ** 2 * (a * a + b * b) + c * c)
    n = -np.array([a * f_assumed * off, b * f_assumed * off, c * off])
    if n[1] > 0:
        n = -n
    out["cam_height_m"] = float(off)
    out["pitch_deg"] = float(np.degrees(np.arcsin(np.clip(n[2], -1, 1))))

    # vehicle bar ratio (SPEC 5.2)
    vm = ids == proxy.VEHICLE
    if vm.any():
        vy, vx = np.nonzero(vm)
        zz = float(np.median(z[vm]))
        runs = geometry.bar_thickness(z, vm, f_assumed)
        runs = runs[runs > 0]
        W = (vx.max() - vx.min() + 1) * zz / f_assumed
        H = (vy.max() - vy.min() + 1) * zz / f_assumed
        D = float(np.percentile(z[vm], 98) - np.percentile(z[vm], 2))
        out["bar_m"] = float(np.median(runs)) if len(runs) else np.nan
        out["box_size_m"] = float((W * H * D) ** (1 / 3))
        out["bar_ratio"] = out["bar_m"] / out["box_size_m"]
    return out

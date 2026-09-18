"""Analytic ray-primitive rasteriser producing CWM proxy conditions.

The proxy is tiny (672x384 before pooling), so closed-form ray intersection in numpy
beats a GL context: it yields exact metric z-depth and integer class IDs with no
driver, no EGL, and no depth-buffer precision to reason about.

Camera convention matches the probe analysis: +X right, +Y down, +Z forward.
Depth written is z along the optical axis, in metres -- never euclidean range.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SRC_W, SRC_H = 672, 384        # render resolution, before the 2x2 pooling (see pool())
OUT_W, OUT_H = 336, 192
NEAR_M = 0.3        # released fixed-log near plane; anything closer cannot be encoded
FAR_M = 256.0
MISS = np.inf


# --------------------------------------------------------------------------- camera

@dataclass
class Camera:
    """Pinhole camera. `focal_px` is quoted at the 336x192 output scale."""

    focal_px: float = 206.0
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, -1.67, 0.0]))
    pitch_deg: float = -8.0        # negative = tilted down, per the probe convention
    yaw_deg: float = 0.0
    roll_deg: float = 0.0

    def basis(self) -> np.ndarray:
        """Rows are the camera axes (right, down, forward) expressed in world coords."""
        cy, sy = np.cos(np.radians(self.yaw_deg)), np.sin(np.radians(self.yaw_deg))
        cp, sp = np.cos(np.radians(self.pitch_deg)), np.sin(np.radians(self.pitch_deg))
        cr, sr = np.cos(np.radians(self.roll_deg)), np.sin(np.radians(self.roll_deg))
        # Yaw about world up (-Y), then pitch about the camera's right axis, then roll.
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
        return (ry @ rx @ rz).T

    def rays(self, width: int = SRC_W, height: int = SRC_H):
        """Unit ray directions in world space, plus the z-axis they project onto."""
        f = self.focal_px * (width / OUT_W)
        u = (np.arange(width) + 0.5 - width / 2.0) / f
        v = (np.arange(height) + 0.5 - height / 2.0) / f
        uu, vv = np.meshgrid(u, v)
        d = np.stack([uu, vv, np.ones_like(uu)], axis=-1)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        R = self.basis()
        return d @ R, R[2]      # world-space directions, camera forward axis

    def to_camera(self, pts: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(pts) - self.position) @ self.basis().T


# ----------------------------------------------------------------------- primitives

class Primitive:
    """A shape that reports, per ray, the euclidean distance to its first surface."""

    class_id: int = 0

    def intersect(self, origin: np.ndarray, d: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def bound(self) -> tuple[np.ndarray, float] | None:
        """Bounding sphere (centre, radius), or None for unbounded shapes."""
        return None


@dataclass
class Plane(Primitive):
    """Half-infinite ground: point + normal, optionally bounded to a radius."""

    point: np.ndarray
    normal: np.ndarray
    class_id: int = 4
    radius: float | None = None

    def intersect(self, o, d):
        n = self.normal / np.linalg.norm(self.normal)
        denom = d @ n
        t = np.where(np.abs(denom) > 1e-9, ((self.point - o) @ n) / np.where(denom == 0, 1e-9, denom), MISS)
        t = np.where(t > 1e-4, t, MISS)
        if self.radius is not None:
            hit = o + t[..., None] * d
            t = np.where(np.linalg.norm(hit - self.point, axis=-1) <= self.radius, t, MISS)
        return t

    def bound(self):
        return None if self.radius is None else (np.asarray(self.point, float), float(self.radius))


@dataclass
class Rect(Primitive):
    """Axis-aligned horizontal rectangle at world height `y`, spanning x0..x1 by z0..z1.

    Bounded on purpose: `bound()` returns a real sphere so the screen-space cull can drop
    it, unlike `Plane`, which answers None and costs a full-frame intersect on every frame
    it appears in.
    """

    y: float = 0.0
    x0: float = -1.0
    x1: float = 1.0
    z0: float = -1.0
    z1: float = 1.0
    class_id: int = 2

    def intersect(self, o, d):
        dy = d[..., 1]
        t = np.where(np.abs(dy) > 1e-9, (self.y - o[1]) / np.where(dy == 0, 1e-9, dy), MISS)
        t = np.where(t > 1e-4, t, MISS)
        hit = o + t[..., None] * d
        inside = ((hit[..., 0] >= self.x0) & (hit[..., 0] <= self.x1)
                  & (hit[..., 2] >= self.z0) & (hit[..., 2] <= self.z1))
        return np.where(inside, t, MISS)

    def bound(self):
        c = np.array([(self.x0 + self.x1) / 2.0, self.y, (self.z0 + self.z1) / 2.0])
        r = float(np.hypot(self.x1 - self.x0, self.z1 - self.z0) / 2.0)
        return c, max(r, 1e-6)


@dataclass
class Sphere(Primitive):
    centre: np.ndarray
    radius: float
    class_id: int = 0
    scale: np.ndarray | None = None     # per-axis radii for an ellipsoid

    def intersect(self, o, d):
        s = np.ones(3) if self.scale is None else np.asarray(self.scale, float)
        oc = (o - self.centre) / s
        dd = d / s
        a = np.einsum("...i,...i->...", dd, dd)
        b = 2.0 * np.einsum("...i,...i->...", oc, dd)
        c = np.einsum("...i,...i->...", oc, oc) - self.radius ** 2
        disc = b * b - 4 * a * c
        ok = disc > 0
        sq = np.sqrt(np.where(ok, disc, 0.0))
        t0 = (-b - sq) / (2 * a)
        t1 = (-b + sq) / (2 * a)
        t = np.where(t0 > 1e-4, t0, t1)
        return np.where(ok & (t > 1e-4), t, MISS)

    def bound(self):
        s = 1.0 if self.scale is None else float(np.max(self.scale))
        return np.asarray(self.centre, float), float(self.radius) * s


@dataclass
class Capsule(Primitive):
    """Swept sphere from `a` to `b` -- the building block of the human rig."""

    a: np.ndarray
    b: np.ndarray
    radius: float
    class_id: int = 8

    def intersect(self, o, d):
        ba = self.b - self.a
        oa = o - self.a
        baba = float(ba @ ba)
        bard = d @ ba
        baoa = oa @ ba
        oad = np.einsum("...i,...i->...", oa, d)
        oaoa = np.einsum("...i,...i->...", oa, oa)
        a_ = baba - bard * bard
        b_ = baba * oad - baoa * bard
        c_ = baba * oaoa - baoa * baoa - self.radius ** 2 * baba
        h = b_ * b_ - a_ * c_
        t = np.full(d.shape[:-1], MISS)
        ok = (h > 0) & (np.abs(a_) > 1e-12)
        sq = np.sqrt(np.where(ok, h, 0.0))
        tc = (-b_ - sq) / np.where(np.abs(a_) > 1e-12, a_, 1.0)
        y = baoa + tc * bard
        body = ok & (y > 0) & (y < baba) & (tc > 1e-4)
        t = np.where(body, tc, t)
        # Spherical caps close the ends.
        for centre in (self.a, self.b):
            oc = o - centre
            b2 = np.einsum("...i,...i->...", oc, d)
            c2 = np.einsum("...i,...i->...", oc, oc) - self.radius ** 2
            h2 = b2 * b2 - c2
            okc = h2 > 0
            tcap = -b2 - np.sqrt(np.where(okc, h2, 0.0))
            t = np.where(okc & (tcap > 1e-4) & (tcap < t), tcap, t)
        return t

    def bound(self):
        a, b = np.asarray(self.a, float), np.asarray(self.b, float)
        return (a + b) / 2.0, float(np.linalg.norm(b - a) / 2.0 + self.radius)


@dataclass
class Box(Primitive):
    """Oriented box via the slab method. `axes` rows are unit, `half` are extents."""

    centre: np.ndarray
    half: np.ndarray
    axes: np.ndarray = field(default_factory=lambda: np.eye(3))
    class_id: int = 10

    def intersect(self, o, d):
        oc = (o - self.centre) @ self.axes.T
        dl = d @ self.axes.T
        inv = 1.0 / np.where(np.abs(dl) < 1e-12, 1e-12, dl)
        t1 = (-self.half - oc) * inv
        t2 = (self.half - oc) * inv
        tmin = np.max(np.minimum(t1, t2), axis=-1)
        tmax = np.min(np.maximum(t1, t2), axis=-1)
        t = np.where(tmin > 1e-4, tmin, tmax)
        return np.where((tmax >= np.maximum(tmin, 0.0)) & (t > 1e-4), t, MISS)

    def bound(self):
        return np.asarray(self.centre, float), float(np.linalg.norm(np.abs(self.half) @ np.abs(self.axes)))


@dataclass
class SphereCluster(Primitive):
    """Many spheres of one class, intersected in a single vectorised pass.

    A vegetation clump is ~11 lobes. As separate primitives that is 11 numpy calls per
    clump per frame, and a street scene with 40 clumps spends all its time in per-call
    overhead rather than arithmetic. Geometry is unchanged -- the result is the per-pixel
    minimum over the same spheres -- so this is a pure speedup.
    """

    centres: np.ndarray                 # (n, 3)
    radii: np.ndarray                   # (n,)
    class_id: int = 5

    def intersect(self, o, d):
        oc = o[None, :] - self.centres                       # (n, 3)
        b = np.einsum("...i,ni->...n", d, oc)                # (..., n)
        c = np.einsum("ni,ni->n", oc, oc) - self.radii ** 2  # (n,)
        disc = b * b - c
        ok = disc > 0
        sq = np.sqrt(np.where(ok, disc, 0.0))
        t0 = -b - sq
        t1 = -b + sq
        t = np.where(t0 > 1e-4, t0, t1)
        t = np.where(ok & (t > 1e-4), t, MISS)
        return t.min(axis=-1)

    def bound(self):
        c = self.centres.mean(0)
        return c, float(np.max(np.linalg.norm(self.centres - c, axis=1) + self.radii))


def wireframe_box(centre, half, axes=None, *, bar_ratio: float = 0.060, class_id: int = 10):
    """The 12 edges of a box as square-section bars -- the measured vehicle proxy.

    Bar cross-section scales with the box, at the measured ratio of ~6% of the
    geometric-mean dimension; a fixed absolute thickness does not match the data.
    """
    centre = np.asarray(centre, float)
    half = np.asarray(half, float)
    axes = np.eye(3) if axes is None else np.asarray(axes, float)
    r = bar_ratio * float(np.prod(2 * half) ** (1 / 3)) / 2.0
    out = []
    for axis in range(3):
        i, j = [k for k in range(3) if k != axis]
        for si in (-1, 1):
            for sj in (-1, 1):
                h = np.zeros(3)
                h[axis] = half[axis]
                h[i] = h[j] = r
                off = np.zeros(3)
                off[i] = si * half[i]
                off[j] = sj * half[j]
                out.append(Box(centre + off @ axes, h, axes, class_id=class_id))
    return out


# -------------------------------------------------------------------------- render

def render(camera: Camera, prims: list[Primitive], *, width=SRC_W, height=SRC_H, cull=True):
    """Return (z-depth in metres, class IDs) at the render resolution.

    `cull=False` disables the screen-space bounding-sphere reject. It exists so the fast
    path can be proved pixel-identical to the brute-force one -- a culler that drops
    geometry produces a plausible-looking image, not an error.
    """
    with np.errstate(all="ignore"):
        return _render(camera, prims, width, height, cull)


def screen_bbox(camera: Camera, centre: np.ndarray, radius: float, width: int, height: int):
    """Conservative pixel bounds of a bounding sphere, or None when it cannot be bounded.

    Uses the exact projected extent of a sphere rather than projecting its centre and
    padding: near the frame edge a centre-plus-padding box is wrong, and a wrong box
    silently drops geometry instead of merely costing time.
    """
    c = camera.basis() @ (np.asarray(centre, float) - camera.position)
    z = float(c[2])
    # Behind the camera and straddling the eye plane used to return the same `None`, and the
    # renderer answers `None` by intersecting against every ray in the frame. A sphere whose
    # *furthest* point is still nearer than the near plane cannot produce a hit the renderer
    # would accept -- it only takes t > t_near, i.e. z > NEAR_M -- so it is not merely
    # unboundable, it is dead, and the full-frame intersect for it is pure waste. On a hot
    # frame 836 primitives were taking that path, all of them 400-680 m *behind* the camera.
    if z + radius <= NEAR_M:
        return ()                   # cannot reach the near plane; cull
    if z <= radius + 1e-6:          # genuinely straddles the eye plane: cannot be bounded
        return None
    f = camera.focal_px * (width / OUT_W)
    denom = z * z - radius * radius
    out = []
    for axis, extent, half in ((0, width, width / 2.0), (1, height, height / 2.0)):
        a = float(c[axis])
        root = radius * np.sqrt(max(a * a + z * z - radius * radius, 0.0))
        lo = (a * z - root) / denom
        hi = (a * z + root) / denom
        p0 = int(np.floor(f * lo + half)) - 1
        p1 = int(np.ceil(f * hi + half)) + 1
        if p1 < 0 or p0 >= extent:
            return ()               # fully offscreen
        out.append((max(p0, 0), min(p1 + 1, extent)))
    return out[1] + out[0]          # (y0, y1, x0, x1)


def _render(camera, prims, width, height, cull=True):
    d, fwd = camera.rays(width, height)
    o = camera.position
    best = np.full((height, width), MISS)
    ids = np.zeros((height, width), np.uint8)
    # Near-plane clip. Intersections are accepted at t > 1e-4, which is a z-depth of
    # ~8e-5 m -- below the 0.001 m the released format treats as invalid, so a camera that
    # grazes or enters a primitive would emit "not sky but zero depth" and break the
    # contract. Reject anything nearer than the encoder's own near plane instead; the ray
    # then takes whatever lies behind, or sky.
    t_near = NEAR_M / np.maximum(d @ fwd, 1e-6)

    order = prims
    if cull:
        # Front-to-back, so `best` tightens early and the occlusion test below can fire.
        def key(p):
            b = p.bound()
            return -np.inf if b is None else float(np.linalg.norm(b[0] - o) - b[1])
        order = sorted(prims, key=key)

    for p in order:
        b = p.bound() if cull else None
        box = None
        near_bound = 0.0
        if b is not None:
            box = screen_bbox(camera, b[0], b[1], width, height)
            if box == ():
                continue            # nothing of this primitive reaches the frame
            near_bound = float(np.linalg.norm(b[0] - o) - b[1])
        if box:
            y0, y1, x0, x1 = box
            view = best[y0:y1, x0:x1]
            # Occlusion reject: if the nearest point the primitive could possibly occupy
            # is already behind everything recorded in its own screen box, it cannot win
            # any pixel. Conservative -- compares against the box maximum, so exact.
            if near_bound > 0.0 and near_bound >= view.max():
                continue
            sub = d[y0:y1, x0:x1]
            t = p.intersect(o, sub)
            closer = (t < view) & (t > t_near[y0:y1, x0:x1])
            view[closer] = t[closer]
            ids[y0:y1, x0:x1][closer] = np.uint8(p.class_id)
            continue
        t = p.intersect(o, d)
        closer = (t < best) & (t > t_near)
        best = np.where(closer, t, best)
        ids = np.where(closer, np.uint8(p.class_id), ids)
    z = best * (d @ fwd)                    # euclidean t -> z along the optical axis
    hit = np.isfinite(best)
    z = np.where(hit, z, 0.0).astype(np.float32)
    ids = np.where(hit, ids, np.uint8(1)).astype(np.uint8)   # misses are sky
    return z, ids


def pool(z: np.ndarray, ids: np.ndarray):
    """The metric family's downsample: minimum positive depth per 2x2, semantic by copy.

    NOT "the" published downsample. The released examples come in two families, and only
    15 of the 40 use this one. The other 25 store uint16 fixed-log codes and pool depth by
    box-mean in log space with a categorical-mode semantic tiebreak, which this does not
    implement. `metadata.json` carries the family in its `depth_format` field -- read it;
    the array dtype is float32 either way and will not tell you.
    """
    h, w = z.shape
    zb = z.reshape(h // 2, 2, w // 2, 2).transpose(0, 2, 1, 3).reshape(h // 2, w // 2, 4)
    ib = ids.reshape(h // 2, 2, w // 2, 2).transpose(0, 2, 1, 3).reshape(h // 2, w // 2, 4)
    masked = np.where(zb > 0, zb, np.inf)
    k = np.argmin(masked, axis=-1)
    zo = np.take_along_axis(masked, k[..., None], -1)[..., 0]
    io = np.take_along_axis(ib, k[..., None], -1)[..., 0]
    allsky = ~np.isfinite(zo)
    return np.where(allsky, 0.0, zo).astype(np.float32), np.where(allsky, np.uint8(1), io).astype(np.uint8)


def render_frame(camera: Camera, prims: list[Primitive], *, cull: bool = True):
    """Full path to a published-format frame: render, pool, clamp beyond the far plane."""
    return pool(*render(camera, prims, cull=cull))


def vegetation_clump(centre, radius: float, *, aspect: float = 1.0, lobes: int = 18,
                     spread: float = 1.10, seed: int = 0, class_id: int = 5):
    """A plant crown as overlapping spheres.

    Step `vegetation` measured the released blobs and ruled out a single ellipsoid:
    an ellipsoid fills pi/4 = 0.785 of its bounding box with almost no variance, while
    released vegetation fills 0.610 with CV 0.21. A lobed cluster reproduces the
    lumpiness; `lobes`, `spread` and the radius range below are the fitted values.

    Re-fitted to 18 lobes / spread 1.10 once the released reference was rebuilt from
    per-example percentiles (veg_fill p10-p90 = 0.555-0.724, p50 0.612). The earlier
    8-14 / 0.75 fit targeted a looser band and rendered at fill ~0.76, above p90.
    """
    rng = np.random.default_rng(seed)
    centre = np.asarray(centre, float)
    # Kept as individual spheres on purpose. Merging them into one SphereCluster was
    # measured slower: the cluster's bounding sphere covers every lobe, so its screen box
    # is far larger than any single lobe's, and the vectorised cost grows as area x lobes.
    offs = rng.normal(0.0, radius * spread, (lobes, 3))
    offs[:, 1] *= aspect
    radii = radius * rng.uniform(0.40, 0.70, lobes)
    return [Sphere(centre + offs[i], float(radii[i]), class_id=class_id) for i in range(lobes)]

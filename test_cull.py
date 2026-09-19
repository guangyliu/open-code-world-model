"""Guard the screen-space cull: it must be a pure speed-up, never a change in pixels.

`screen_bbox` folding "straddles the eye plane" and "entirely behind the camera" into the
same `None` cost a full-frame intersect per primitive -- on one frame, 836 primitives sitting
400-680 m behind the camera, each tested against all 258,048 rays. The fix returns `()` for
the dead case. Because a wrong cull *drops geometry* rather than merely costing time, the
real assertion here is not the branch but the pixels: `cull=False` skips `screen_bbox`
entirely, so it is an exact oracle for whatever the culled path produces.
"""
from __future__ import annotations

import sys

import numpy as np

from compiler.render import NEAR_M, Camera, Sphere, screen_bbox, render


def t_branches() -> bool:
    """The three cases must stay distinguishable: dead, unboundable, boundable."""
    cam = Camera()                      # at y=-1.67, looking down +Z
    ok = True
    behind = screen_bbox(cam, np.array([0.0, -1.67, -500.0]), 8.0, 672, 384)
    if behind != ():
        print(f"  behind-camera sphere returned {behind!r}, expected ()"); ok = False
    # Just far enough back that its nearest point still cannot reach the near plane.
    edge = screen_bbox(cam, np.array([0.0, -1.67, NEAR_M - 5.0]), 4.0, 672, 384)
    if edge != ():
        print(f"  sphere short of the near plane returned {edge!r}, expected ()"); ok = False
    # Enclosing the eye: genuinely unboundable, must stay None so the renderer tests it.
    straddle = screen_bbox(cam, np.array([0.0, -1.67, 0.5]), 4.0, 672, 384)
    if straddle is not None:
        print(f"  eye-enclosing sphere returned {straddle!r}, expected None"); ok = False
    return ok


def t_pixels() -> bool:
    """Culled and uncensored renders must agree bit for bit, including behind the camera."""
    rng = np.random.default_rng(7)
    prims = []
    for i in range(60):                 # in front, the ones that should survive
        c = rng.uniform([-20, -6, 4], [20, 0, 90])
        prims.append(Sphere(centre=c, radius=float(rng.uniform(0.4, 3.0)), class_id=int(i % 12)))
    for i in range(400):                # the ones the old code paid a whole frame for
        c = rng.uniform([-400, -60, -680], [400, 0, -400])
        prims.append(Sphere(centre=c, radius=float(rng.uniform(0.4, 9.0)), class_id=int(i % 12)))
    ok = True
    for yaw, pitch in ((0.0, -8.0), (37.0, -8.0), (150.0, 2.0), (180.0, -20.0)):
        cam = Camera(yaw_deg=yaw, pitch_deg=pitch)
        za, ia = render(cam, prims, cull=True)
        zb, ib = render(cam, prims, cull=False)
        dz = np.abs(np.where(np.isfinite(za) | np.isfinite(zb), za - zb, 0.0))
        bad_z = float(np.nanmax(np.where(np.isfinite(dz), dz, 0.0)))
        bad_id = int((ia != ib).sum())
        if bad_z != 0.0 or bad_id:
            print(f"  yaw={yaw} pitch={pitch}: depth maxdiff {bad_z}, {bad_id} id pixels differ")
            ok = False
    return ok


def t_cost() -> bool:
    """Count the primitives that still force a full-frame intersect.

    This is the check that would have failed before the fix. `t_pixels` would not have:
    a sphere behind the camera produces no accepted hit either way, so the old code was
    bit-correct and merely slow -- which is exactly why it survived so long. Counting the
    unbounded path turns "slow" into something an assertion can see, without timing
    anything and without the flakiness that would bring.
    """
    rng = np.random.default_rng(7)
    cam = Camera(yaw_deg=180.0, pitch_deg=-8.0)     # turned around: everything is behind
    prims = [Sphere(centre=rng.uniform([-400, -60, -680], [400, 0, -400]),
                    radius=float(rng.uniform(0.4, 9.0)), class_id=int(i % 12))
             for i in range(400)]
    unbounded = sum(1 for p in prims
                    if screen_bbox(cam, *p.bound(), 672, 384) is None)
    if unbounded:
        print(f"  {unbounded}/400 primitives behind the camera still take the full-frame path")
        return False
    return True


def t_gray16_roundtrip():
    """encode -> decode must return the depths it was given, across the whole range."""
    from compiler.render import (FAR_M, GRAY16_MAX, NEAR_M, decode_gray16,
                                 encode_gray16)

    ok = True
    z = np.array([0.3, 0.5, 1.0, 2.5, 10.0, 47.3, 120.0, 255.0, 256.0])
    back = decode_gray16(encode_gray16(z))
    rel = np.abs(back - z) / z
    if rel.max() > 2e-4:
        print("     round-trip error %.2e at z=%.1f" % (rel.max(), z[rel.argmax()]))
        ok = False

    # Near is WHITE: the near plane is the largest code, the far plane is 0. Getting this
    # backwards is the error that takes flat-plane fits from 0.3% to 10.8-21.9%.
    if not abs(float(encode_gray16(np.array([NEAR_M]))[0]) - GRAY16_MAX) < 1.0:
        print("     near plane does not encode to white")
        ok = False
    if float(encode_gray16(np.array([FAR_M]))[0]) != 0.0:
        print("     far plane does not encode to 0")
        ok = False

    # A no-hit pixel encodes to the far sentinel and decodes back to the far plane.
    if float(encode_gray16(np.array([0.0]))[0]) != 0.0:
        print("     no-hit did not encode to the far sentinel")
        ok = False
    if float(decode_gray16(np.array([0.0]))[0]) != FAR_M:
        print("     code 0 did not decode to the far plane")
        ok = False

    print("     round-trip clean, max rel err %.1e over 0.3-256 m" % rel.max())
    return ok


def t_gray16_blended_sky():
    """The signature that tells the two families apart, on a render of our own.

    Under minimum-positive-depth a 2x2 that straddles a silhouette resolves to ONE of the
    two surfaces: a pixel labelled sky had no hit in all four samples, so its depth is
    exactly 0. A box MEAN in code space cannot do that -- it returns a code belonging to
    neither surface, so some sky-labelled pixels carry a finite blended depth. That is
    measured in the released data: 100% of metric-family sky pixels are exactly 0, while
    0.1-1.1% of gray16 sky pixels sit at 43-256 m.

    This reproduces the signature on a scene we control, which is the only way to be sure
    the implementation has it rather than the data merely showing it.
    """
    from compiler.render import (FAR_M, Box, decode_gray16, pool, pool_gray16,
                                 render)

    ok = True
    # A wall standing against open sky: half the frame is a silhouette edge, so the 2x2
    # blocks along it are exactly the straddling case.
    cam = Camera(focal_px=206.0, position=np.array([0.0, -1.67, 0.0]), pitch_deg=0.0)
    prims = [Box(np.array([0.0, -3.0, 14.0]), np.array([3.0, 3.0, 0.4]), class_id=0)]
    z, ids = render(cam, prims)

    zm, im = pool(z, ids)
    cg, ig = pool_gray16(z, ids)
    zg = decode_gray16(cg)

    # metric: every sky pixel is exactly zero, with no exceptions.
    metric_bad = int(((im == 1) & (zm > 0)).sum())
    if metric_bad != 0:
        print("     pool() produced %d sky pixels with non-zero depth" % metric_bad)
        ok = False

    # gray16: some sky pixels carry a blended depth, and it lands in the released band.
    blended = (ig == 1) & (cg > 0)
    n_blend = int(blended.sum())
    if n_blend == 0:
        print("     pool_gray16 produced no blended sky pixels; it is not a box mean")
        ok = False
    else:
        band = zg[blended]
        if not (band.min() >= 43.0 and band.max() <= FAR_M + 1e-3):
            print("     blended sky depths %.1f-%.1f m are outside the released 43-256 m"
                  % (band.min(), band.max()))
            ok = False

    # Negative case: wiring the gray16 path to minimum-positive must fail this test. If it
    # ever does get wired that way, `blended` above goes to zero and so does this.
    zmin, imin = pool(z, ids)
    if int(((imin == 1) & (zmin > 0)).sum()) != 0:
        print("     the negative case is not staged: min-positive blended a sky pixel")
        ok = False
    if n_blend and int(((imin == 1) & (zmin > 0)).sum()) >= n_blend:
        print("     min-positive matched the box mean; the signature does not separate them")
        ok = False

    print("     gray16 %d blended sky px at %.0f-%.0f m; pool() %d"
          % (n_blend, zg[blended].min() if n_blend else 0,
             zg[blended].max() if n_blend else 0, metric_bad))
    return ok


if __name__ == "__main__":
    tests = [t_branches, t_pixels, t_cost,
             t_gray16_roundtrip, t_gray16_blended_sky]
    results = []
    for t in tests:
        r = bool(t())
        print(f"{'ok  ' if r else 'FAIL'} {t.__name__}")
        results.append(r)
    n_ok = sum(results)
    print(f"\n{n_ok}/{len(results)} cull checks clean")
    sys.exit(0 if n_ok == len(results) else 1)

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


if __name__ == "__main__":
    tests = [t_branches, t_pixels, t_cost]
    results = []
    for t in tests:
        r = bool(t())
        print(f"{'ok  ' if r else 'FAIL'} {t.__name__}")
        results.append(r)
    n_ok = sum(results)
    print(f"\n{n_ok}/{len(results)} cull checks clean")
    sys.exit(0 if n_ok == len(results) else 1)

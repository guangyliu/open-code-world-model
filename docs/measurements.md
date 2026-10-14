# Measurements: the released format, and how far our proxies are from theirs

This page holds the measured facts behind the README's one-line claims. Every number here
can be re-derived from a script in this repository; the script is named next to the number.

## What this is, and what it is not

This project produces **proxy conditions in the released format, at the released camera
intrinsics**. It is *not* a reconstruction of the authors' scenes.

We tried to establish the stronger claim and could not. We built an inverse renderer
(`inverse/`), recovered their camera, unprojected their depth maps to metric point clouds, and
fitted our primitives to the result. Their water surface is exactly a plane and their road is a
plane to within 1.4%, so those two invert cleanly. Everything else does not: a single-frame
*self-fit* with roughly a thousand free parameters lands at 2.47% median depth error but 13.3% at
p90, with human IoU 0.62 and vegetation IoU 0.50. Their content is film-grade mesh; our
primitives approximate its visible surface and nothing more.

The cleanest way to state the gap: on a frame where the camera is stationary, re-rendering
*their own* recovered point cloud costs 0.03% depth error, and re-rendering our fitted primitive
scene costs 0.60%. **Our primitives are about 20× worse than perfect geometry at the same
camera.** That is the honest size of the difference between a primitive proxy and film-grade
mesh.

The measurements, including the methods that were tried and retired (a plane-fit focal
estimator that is analytically degenerate, a vegetation lobe-count sweep that fails on its own
control, and primitive ranking from a single view, which scores `Plane` as the best model of a
human body), are in [`inverse/FINDINGS_inverse_probe.md`](../inverse/FINDINGS_inverse_probe.md).

## The condition format

Each frame is metric depth plus a semantic class id, rendered at 672×384 and pooled 2×2 to
336×192.

| field | value |
|---|---|
| resolution | 336×192, from 672×384 |
| depth | float32, metres, **first surface**, 0 = no hit |
| near / far | 0.3 m / 256 m |
| focal length | **206.6 px** at 336×192 |
| semantic ids | `{0, 1, 2, 4, 5, 8, 9, 10}` |

The focal length is measured, not assumed. (206 px is also exactly a 50.0° vertical field of
view: 96/tan 25° = 205.87. Offered as a hypothesis about where the number came from, not a
finding.) Pitch and camera height are **not** fixed: they vary per example and within a take.
Camera height ranges from 1.6 m to 231 m across the 40 examples, and one example's horizon moves
100 rows in 124 frames. There is no single rig to recover.

We recovered the focal length from the released depth maps with an estimator validated against
known-answer synthetic data before use, cross-checked by a second independent mechanism:
**206.6 px, IQR [204, 208] across 97 estimates**. Re-derive it with `python inverse/focal_all.py`.

## The semantic vocabulary is smaller than the ontology

The ontology has 12 ids. Across all 40 released examples, **only eight ever appear**:

| id | meaning | share of pixels | examples using it |
|---|---|---|---|
| 0 | everything not otherwise classed | 56.5% | 40/40 |
| 1 | sky, and "no hit" | 14.3% | 40/40 |
| 2 | water | 6.1% | 23/40 |
| 4 | road | 8.5% | 36/40 |
| 5 | vegetation | 8.4% | 40/40 |
| 8 | **the protagonist** | 3.2% | 40/40 |
| 9 | animal | 0.003% | 29/40 |
| 10 | vehicle | 2.9% | 30/40 |

Ids 3, 6, 7 and 11 appear in none of them.

Class 0 is worth a warning: it is **not** a ground plane. It is a catch-all for everything not
otherwise classed, and its planarity across examples runs from 1.2% to 55%. Their road *is* a
plane, but it is not the class-0 plane: measured tilt 1.4°–70.9° and offset 0.5–16.6 m between
the two. Modelling road as a quad lying on a shared class-0 ground plane is wrong.
`RELEASED_CLASSES` in `world/state.py` declares this set and a test fails any take that emits
outside it.

**Class 8 is not "human", it is "the protagonist".** In example 31 the protagonist is a white
stag and it is class 8, while class 9 sits at 0.02%. That is why class 9 is nearly empty in the
released data: when an animal is the subject, it takes 8.

## There are two condition families, not one

This matters if you parse the released examples, and it is not documented anywhere we could find:

| family | examples | depth encoding | depth pooling | semantic pooling |
|---|---|---|---|---|
| metric | 1, 2, 28–40 (15) | float32 metres, first surface | min positive depth 2×2 | copy |
| gray16 | 3–27 (25) | uint16 fixed-log codes **stored in a float32 array** | box mean 2×2, in log space | categorical mode 2×2 |

The gray16 family decodes as `z = 0.3 * (256/0.3) ** (1 - v/65535)`: near is *white*. Read it
as linear disparity and roads span 1.5–7282 m; read it near-black and flat-plane error is
10.8–21.9% instead of 0.30–1.42%. We verified the decode by falsification rather than by
assumption.

`compiler/render.py` implements the **metric** family's pooling. For the gray16 family the depth
pooling and the semantic tiebreak both differ; this is a known gap, not a silent one.

(Also: examples 32 and 33 ship byte-identical conditions with different first frames. Anything
averaged "over 40 examples" is really over 39 distinct geometries.)

## Throughput

A 484-frame take renders in about 25 s on CPU. Throwing cores at it does not help: measured on
one take, 8 workers 24 s, 32 workers 28 s, 100 workers 42 s, 200 workers 75 s. The ray tracing
is not the bottleneck; the serial simulation ahead of it is. Set `CWM_WORKERS` accordingly.

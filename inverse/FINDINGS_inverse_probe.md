# Inverse-rendering probe: can we recover the authors' scene geometry and re-render it?

**Question asked.** Recover the authors' scene geometry from their released conditions,
re-render with `compiler/render.py`, measure the error — the test of whether we may
write "reproduce" in tomorrow's release.

**Verdict: no, not as "reproduce".** Three things did come out of it:

1. **The camera is recovered.** focal = **206.6 px at 336x192** (IQR [204,207] over 97
   estimates), from an estimator validated against ground truth before use, corroborated
   by an independent mechanism, and surviving two falsification tests designed to kill it.
2. **Their water surface is exactly a plane** — flat to the limit the released format can
   express — and their road is a plane to 0.02–1.4%. Those two we can invert.
3. **Everything else is a per-frame surface refit, not a scene.** Single-frame self-fit
   across 18 examples: 2.47% median depth error, but 13.3% at p90 and 82% worst-case, with
   human IoU 0.62 and vegetation IoU 0.50 — using ~1000 free parameters per frame.

The defensible claim for the release is **"proxy conditions in the released format, at the
released camera intrinsics"**. Not "reproduction of the released scenes".

One thing the probe could *not* settle: temporal stability. Our ego-motion recovery is the
binding error beyond ~8 frames (see 4b), and a control proved it, so any claim either way
about geometry persisting over time is unsupported by this data.

All work in `probe_src/inverse/`. Nothing outside that directory was modified.

---

## 0. Correction to the brief, found before any geometry was done

The brief states every `condition/*.npz` holds `depth float32 (124,192,336)` in metres,
first surface. **That is true for 15 of the 40 examples, not all 40.**

| family | examples | `depth_format` | depth resample | semantic resample |
|---|---|---|---|---|
| metric | 1, 2, 28–40 (15) | `float32_le_metric_first_surface_metres` | `minimum_positive_depth_2x2` | copy |
| gray16 | 3–27 (25) | `uint16_le_fixed_log_gray16_codes` | `box_mean_2x2_fixed_log_gray16_round_half_up_v1` | `categorical_mode_2x2_nearest_center_tiebreak_ontology0_11_v1` |

The gray16 family stores integer codes in a float32 array (values 0–65535, all integral),
mapped by `fixed_log_view_z_0p3m_256m_near_white_gray16_v1`. Decode:

```
z = 0.3 * (256/0.3) ** (1 - v/65535)      # near = white = 65535
```

Verified falsifiably, not assumed: road-class pixels decode to a flat plane at 0.30–1.42%
median relative error under `near_white`, versus 10.8–21.9% under `near_black` and
physically absurd ranges (road spanning 1.5–7282 m) under a linear-disparity reading.

**Consequence for `compiler/render.py` (reported, not fixed — another agent owns that file):**
`pool()` implements min-positive-depth + semantic-by-copy. That matches the 15 metric
examples exactly and is the *wrong* resample for the other 25, where depth is box-mean in
gray16 log space and semantics are categorical-mode with a nearest-centre tiebreak.
`NEAR_M = 0.3` / `FAR_M = 256.0` in that file are exactly this family's log endpoints, so
the constants are right and only the pooling differs.

Confirmed empirically rather than read off the metadata (`poolproof.py`). Under
min-positive + copy, a sky pixel is either sky-at-zero or replaced wholesale by the near
surface; a blended value in between is impossible. In the metric family **100%** of sky
pixels have depth exactly 0 (ex1 17325/17325, ex2 22459/22459, ex36 16901/16901). In the
gray16 family 0.1–1.1% of sky pixels carry a finite blended depth of 43–256 m (ex3 29 px,
ex5 7 px, ex20 50 px, ex23 63 px). Only a box **mean** across a silhouette produces those.

Also: **examples 32 and 33 have byte-identical `condition/000.npz` but different
`first_frame.png`** — one geometry, two appearances. Deliberate, not a duplicate bug, but
anything averaging "over 40 examples" is really over 39 distinct geometries.

The content itself is film-grade CG video (`first_frame.png`: a photoreal cobblestone
lane with a hooded woman and a giant winged creature; an anime diver over a reef). It is
not primitive-rendered, which sets the ceiling for everything below.

---

## 1. Camera recovery

### 1a. The requested method is analytically degenerate — do not use it

The brief asks for the `(focal_px, pitch_deg, height)` that makes ground pixels unproject
to a single flat plane. **That has a one-parameter family of solutions.** For a plane
`n·p = d` seen by a pinhole of focal `f`, with `u=(i-cx)/f`, `v=(j-cy)/f` and `p = z(u,v,1)`:

```
1/z = (n_x/(d f)) (i-cx) + (n_y/(d f)) (j-cy) + n_z/d
```

which is **affine in pixel coordinates for every f**. Planarity is exactly f-invariant.
Fitting a plane and reading off a focal recovers whatever focal you seeded. This is the
shape of the earlier 2.04x bias, and it would have reproduced it.

What the plane fit *does* establish, f-free: the projection really is a pinhole. Class-4
road pixels fit affine-inverse-depth at median relative depth error **0.22%** (ex28,
40041 px), 0.02% (ex36), 0.41% (ex40), 0.30% (ex19). A fisheye, or a monocular depth
estimate, does not do that over 40k pixels.

Camera height and pitch are *per-example and time-varying*, so there is no rig to recover:
height above the fitted ground runs 1.6 m (ex36) to 231 m (ex34), and within ex2 the
horizon row moves from 105 to 3 inside 124 frames. Each example is its own cinematic shot.

### 1b. Focal from two-frame rigidity

Unprojecting with a wrong focal applies `S = diag(f0/f, f0/f, 1)` to the true cloud, and
`S R S^-1` is a rotation only when `R` commutes with `S`. So **camera rotation, not
translation, is what makes focal observable**; pairs with pure translation carry no
information and a planar scene carries none either.

Method: for each candidate `f`, run point-to-plane ICP entirely in that `f`'s own geometry
(static classes 0/2/4/5 only; depth-discontinuity pixels dropped), score
`median point-to-plane residual / cloud RMS radius`, take the minimum.

**Validated against ground truth before being pointed at the data** (`synth_check.py`,
`synth3.py`): scenes rendered by our own renderer at f = 140, 206, 330, 480 recover to
141.8, 206.0, 330.1, 479.6 — within 1.3%, with the minimum 30–700x below its neighbours.
An earlier initialisation-chained version of the same estimator was biased toward its seed
(returned 356 for a true 480) and was discarded; only the global-search version is used.

**Result on the released data** (`focal_all.py`), 97 well-conditioned (example, frame-pair)
estimates across 30 of the 40 examples:

```
focal_px at 336x192   median 207   IQR [204, 208]   p10-p90 [182, 222]
```

Per-example medians sit at 202–210 for fifteen examples, with per-example spread under 10%
in ten of them (ex3 2%, ex9 2%, ex15 2%, ex23 3%, ex26 1%, ex39 1%).

### 1c. Chasing the coincidence

207 lands within 0.5% of `render.py`'s hard-coded default of 206.0, and the pair-selection
step ran its gating ICP at 206. That had to be killed or confirmed.

- **Test A — gate dependence.** Re-ran the whole pipeline with the pair-selection ICP
  gated at f = 120, 206, 350 and 500. Recovered medians: **206.5, 206.6, 206.9, 206.7**
  (IQR [204,207] in every case). The answer does not move with the seed — a 4.2x change in
  the gate moves the estimate by 0.4 px. The estimate is not an echo of the default.
- **Test B — injected zoom on their own data.** A central crop by k, resized back to
  336x192, *is* the same scene through a camera of focal `k·f`. Fed the estimator k = 1.5
  and k = 2.0 versions of the released frames; it must return 1.5x and 2.0x the baseline.

  | zoom k | n | recovered focal | observed ratio | expected |
  |---|---|---|---|---|
  | 1.0 | 29 | 206.6 | 1.00 | 1.0 |
  | 1.5 | 16 | **309.4** | **1.50** | 1.5 |
  | 2.0 | 10 | **412.7** | **2.00** | 2.0 |

  Exact, to three significant figures, on the authors' own frames with a ground-truth
  change injected into them. (The crop-and-resize is nearest-neighbour, which blocks the
  depth gradients the normals are built from, and the estimator still recovered the ratio
  exactly — so it is not riding on interpolation smoothness either.)
- **Test C — independent mechanism.** Class-8 blob height above the fitted ground plane,
  set to 1.70 m. Across examples with a good ground fit and a single upright figure
  (image-plane w/h < 1.0) this gives f = 137–228, median ~190 — consistent with 207,
  though much looser, and confounded by real variation in character height (these are
  stylised characters, and camera heights vary 100x across examples).
- **Test D — is 206 a structural constant?** 96/tan(25°) = 205.87, i.e. f = 206 px is
  exactly a **50.0° vertical field of view** (78.6° horizontal at 336 px). That is a
  plausible authored default rather than a numerical accident, and it is *not* a power of
  two, a frame dimension, or a pixel count. Offered as a hypothesis, not a claim.

**Conclusion: focal_px = 207 ± 3 at 336x192 (p10–p90 182–222 across examples).** Our
existing default of 206.0 is right, and it is right for a reason we can now state rather
than inherit.

---

## 2. Segmentation: what the scenes actually decompose into

Unprojecting at f = 206 and running 3D connected components (0.30 m voxels, >=120 points)
over frame 0 of all 40 examples:

| class | 3D components | notes |
|---|---|---|
| 0 terrain | 196 | **not "the ground"** — a catch-all, planarity 1.2–55% |
| 2 water | 44 | exactly planar (see 3b) |
| 4 road | 27 | planar to 0.02–1.4%, but on its own plane |
| 5 vegetation | 139 | metric bbox 1.45x2.71x3.46 m up to 53.8x24.9x26.9 m |
| 8 human | 44 | image aspect w/h median 0.63, CV 0.42 — consistently upright figures |
| 10 vehicle | 17 | heterogeneous |

Example 28, frame 0, decomposes into 10 objects: the cobbled lane (11.6 x 5.2 x 8.3 m at
4.2 m) plus a second road patch, two terrain banks, one small vegetation clump
(1.6 x 0.5 x 1.0 m), one standing human (0.95 x 1.54 x 1.09 m at 1.9 m — a plausible
adult, an independent sanity check on the focal), and four class-10 pieces of which the
largest is 5.4 x 4.0 x 5.4 m (the winged creature, which their own labeller calls
"vehicle").

---

## 3. Primitive fitting

### 3a. A methodological trap, reported so nobody repeats it

Ranking primitives by point-to-surface residual **is invalid from a single view** and
produced a confidently wrong answer: it ranked `Plane` as the best fit for the human body
(3.4 cm RMS) and for the cart (10.7 cm). A single depth frame shows only the visible
*surface sheet* of an object, and a plane fits a sheet. Every residual computed that way
is a lower bound on true shape error with no upper bound at all.

The valid metric is to instantiate each candidate, render it at the released settings, and
score silhouette IoU *and* depth together. All numbers below are render-scored.

Second trap, also reported: under render scoring, `SphereCluster` with 24 lobes won for
terrain, road, vegetation, human *and* vehicle alike. That is a capacity artifact — 96 free
parameters against a box's 9 — and it means **the winning primitive type carries no
information about the content.** Their content is not made of our primitives; the ranking
just selects whichever approximator has the most parameters.

### 3b. Water — surface exactly ours, outline not

**Water is an analytic plane.** Fitting a single plane to class-2 pixels:

| example | px | depth span | median rel. residual | median abs. residual |
|---|---|---|---|---|
| 21 | 34076 | 8–138 m | 2.7e-05 | 1.31 mm |
| 18 | 35523 | 1.5–213 m | 2.6e-05 | 0.07 mm |
| 12 | 9510 | 9.9–253 m | 3.2e-05 | 1.57 mm |
| 16 | 4138 | 27–135 m | 3.5e-05 | 1.57 mm |
| 8 | 1967 | 23–241 m | 3.1e-05 | 1.76 mm |
| 5 | 1280 | 37–242 m | 1.9e-05 | 2.61 mm |

3e-05 is suspiciously small, so it was chased. **It is the encoder's own quantisation
floor.** The 16-bit fixed-log code step is `ln(256/0.3)/65535 = 1.03e-04` relative; uniform
quantisation error has std `step/sqrt(12) = 2.97e-05`. Observed: 1.9–3.5e-05. The residual
*is* the quantiser — meaning the underlying surface is exactly planar, to the limit the
released format can express. Cross-checked with an unrelated statistic (residual to a
smooth quadratic in inverse depth): water 0.001%, road 0.544%, our renderer's exactly flat
`Box` face 0.000%, our `Sphere` 0.532%.

Water is also horizontal: water-plane vs road-plane tilt is **0.43°** (ex8) and **0.99°**
(ex5). Caveat: all six water examples with enough pixels are in the gray16 family, so this
is demonstrated for that family only.

**But the assembled scene scores water IoU 0.000 on three examples, which contradicts the
above — so one of my own results had to be wrong.** Scoring the water segment alone
resolves it, and the resolution is a finding rather than a bug:

| example | water px | Plane depth error | Plane IoU alone | IoU in assembled scene |
|---|---|---|---|---|
| 18 | 35523 | **0.40%** | 0.813 | 0.887 |
| 21 | 34076 | 1.15% | 0.528 | 0.512 |
| 12 | 9510 | 4.67% | 0.251 | 0.211 |
| 16 | 4138 | 0.90% | 0.064 | 0.267 |
| 8 | 1967 | 1.80% | 0.034 | 0.000 |
| 5 | 1280 | 3.77% | 0.044 | 0.000 |

The *surface* is reproduced to a fraction of a percent at every size. What collapses is the
**outline**: an unbounded plane spills past the shoreline, so IoU tracks how much of the
frame the water fills, not how well the surface is modelled. Where water fills the frame
(ex18) IoU is 0.89; where it is a lake seen past a bank (ex5, ex8) it is ~0.

So the accurate statement is: **their water surface is exactly a plane, and our primitive
set has no way to express its boundary.** `Rect` is an *axis-aligned rectangle*, and a
natural shoreline is not one — fitting `Rect` to these segments costs 15.9–65.0% depth
error against `Plane`'s 0.40–4.67%, because in a camera-aligned frame `Rect`'s horizontal
constraint is simply wrong. Reproducing water needs plane + boundary polygon, which is one
new primitive, not a re-fit.

### 3b2. Road — a plane, but not the class-0 plane

**Road is a quad, but not on a shared ground plane.** Road planarity is excellent — median relative
depth error to a single plane is 0.02% (ex36), 0.22% (ex28), 0.30% (ex19), 0.41% (ex40),
0.82% (ex2), 0.87% (ex22), 1.00% (ex3), 1.13% (ex8), 1.22% (ex5), 1.42% (ex20), with only
ex38 an outlier at 4.05%; that is ~10–50x better than class 0 in the same frames, but the road plane is *not* the class-0 plane: tilt
between them is 1.4°–70.9° and the median offset is 0.5–16.6 m. So class 0 is not terrain
in the "ground" sense — it is everything that is not road, including walls, cliffs and
buildings. Modelling road as a quad on a single world ground plane shared with class 0 is
wrong; it needs its own plane.

### 3c. Vegetation — the long-open question, answered partly, with one test retired

**The lobe-count sweep does not work and should not be run again.** Sweeping sphere-cluster
lobe count k and looking for saturation fails on the *control*: our own `vegetation_clump`,
where the answer is known to be 18 lobes, does not saturate at 18 either (k=1 24.2% depth
error, k=64 1.9%, monotone). From a single view k-means tiles the visible surface rather
than recovering the generating lobes, so the curve is a generic approximator curve for
every input, ground truth included.

**What the data does say, with f-invariant descriptors:** their class-5 segments are a
heterogeneous shape population, **not one primitive scaled**:

- metric bounding boxes span 1.45x2.71x3.46 m to 53.8x24.9x26.9 m — a 20x linear range;
- image-plane aspect ratio (f-invariant) p10–p90 = 0.68–2.69, CV 0.95;
- surface roughness — median relative depth residual to a smooth quadratic-in-inverse-depth
  model, dimensionless and f-invariant — p10–p90 = 0.20%–1.90%, CV 0.92, spanning the range
  from a single smooth ellipsoid (control 0.53%) to a lobed cluster.

**Honest limit, stated because it cuts against the tidy answer:** at *matched apparent
size* their vegetation roughness (1.2–5.0% for 4500–12400 px blobs) **overlaps** our
sphere-cluster proxy's (2.8–4.1% at 5200–10300 px). So this evidence does **not** reject
sphere-cluster as a per-blob proxy. What it rejects is the fixed parameterisation: a single
18-lobe / spread-1.10 clump cannot cover a population whose size varies 20x and whose
smoothness varies 10x. The actionable finding is about our renderer, not theirs.

---

## 4. Re-render and score

Scene assembled from the per-segment render-scored best primitive, rendered by
`compiler/render.py` at the released settings (672x384 -> 2x2 pool -> 336x192), camera at
the origin with zero rotation so the fitted primitives live directly in camera coordinates.

**18 examples, frame 0, focal 206, 2–18 primitives each:**

```
 ex  prims   depth_med   depth_p90    abs_med    abs_p90   cover   road   veg  human terrain vehicle
  1     10      9.44%      23.46%     3.832m     9.885m   59.1%   0.00  0.05  0.67   0.61    -
  2     14      2.29%      10.64%     0.183m     1.594m   81.1%   0.90  0.00  0.62   0.55    -
  3     10      2.86%      10.36%     0.233m     1.865m   86.4%   0.88  0.52  0.60   0.70   0.78
  5     13      4.32%      58.13%     0.256m     4.645m   93.9%   0.69  0.28  0.51   0.49    -
  8     16      1.39%       7.47%     0.135m     1.952m   87.9%   0.90  0.32  0.51   0.35   0.39
 15      2      5.98%      15.13%     0.330m     1.042m  100.0%    -     -    0.56   0.99    -
 16     14      2.08%      11.47%     0.128m     2.582m   94.4%    -    0.86  0.54   0.88    -
 19     17      0.97%      10.35%     0.066m     2.081m   92.5%   0.91  0.65  0.53   0.36   0.52
 21     17      4.81%      81.96%     0.894m    18.333m   70.8%    -     -    0.65   0.64    -
 22     16      2.66%      35.37%     0.264m     6.335m   92.5%   0.84  0.57  0.53   0.70    -
 28     10      0.61%      55.91%     0.020m     2.567m   99.6%   0.82  0.79  0.64   0.56   0.56
 29     13      3.11%       8.02%     0.138m     0.729m   87.6%    -    0.67  0.65   0.87    -
 30     11      9.61%      43.08%     0.482m     5.686m   87.4%   0.00  0.20  0.62   0.82    -
 32     14      1.09%       8.10%     0.060m     2.119m   94.1%   0.00  0.42  0.64   0.93   0.27
 35     18      3.88%      24.34%     0.182m     3.290m   95.0%    -    0.54  0.67   0.83   0.66
 36     16      0.74%      17.82%     0.037m     1.870m   88.8%   0.86  0.48  0.61   0.44   0.00
 39     15      1.85%      11.49%     0.114m     1.309m   91.9%    -    0.69  0.67   0.80   0.45
 40     12      0.61%       4.50%     0.035m     0.802m   94.8%   0.95  0.40  0.56   0.62   0.48
```

**Depth: median-of-medians 2.47%, median p90 13.3%, worst p90 82.0%.**
Per-class IoU across the 18: road **0.850** (min 0.00, max 0.95), terrain 0.671, human 0.616,
vegetation 0.501, vehicle 0.476, sky 0.385, animal 0.000 (2 examples, both missed entirely).

Read this correctly. The 2.47% median is a **self-fit**: the primitives were fitted to this
exact frame's depth map, with roughly a thousand free parameters per frame. It is the
ceiling of what our primitive set can express about one view, not a measure of recovered
geometry. The p90 is the honest half of it — on ex28 the median is 0.61% while the p90 is
55.9%, i.e. a tenth of the frame is wrong by more than half its depth. And the median
human IoU of 0.616 with a range of only 0.506–0.671 across 18 wildly different characters
is what a blob approximation of a body looks like: never badly wrong, never right.

### 4b. Temporal stability: could not be answered, and why

Fit on frame 0, recover the camera motion to frame t by ICP on their own depth, re-render,
score. Baselines: NULL = reuse their frame 0 unchanged; ORACLE = refit on frame t.

```
ex28   t      rot     OURS med     NULL med     (ORACLE at t=64: 0.57%)
        1   0.00 deg     0.60%        0.02%
        8   0.76 deg     2.46%        0.78%
       32   6.20 deg    35.14%       19.06%
       64  17.36 deg    29.39%       18.12%
```

The obvious reading — "we are refitting depth maps, not recovering geometry" — is **wrong,
and the control that caught it is worth recording.** Warping *their own* frame-0 point
cloud by the same recovered motion (geometry perfect by construction) gives 1.85% at t=8,
**33.82%** at t=32 and 45.33% at t=64. That is as bad as our primitive scene. **Beyond
~8 frames the error is dominated by our ego-motion recovery, not by the primitives**, so
the temporal experiment says nothing about geometry transfer.

What survives the control is the zero-motion comparison. At t=1 on ex28, with the camera
genuinely stationary (rot 0.00°), warping their own geometry costs 0.03% while our fitted
primitive scene costs 0.60% — **our primitives are ~20x worse than perfect geometry at the
same camera.** That is the correctly attributed statement about primitive fidelity.

Answering temporal stability properly needs ego-motion far better than depth-only ICP,
which needs the RGB video. Only `first_frame.png` is released, so it cannot be done from
this data.

---

## 5. Verdict, and what the word "reproduce" can carry

**Geometric round-trip is not viable as a reproduction claim.**

Safe to say:
- we match the released **condition format** exactly for the 15 metric examples (and know
  precisely what differs for the other 25);
- we have the **camera intrinsics**: focal **206.6 px at 336x192** (IQR [204,207]),
  recovered by an estimator validated against ground truth at f = 140/206/330/480 before
  use, invariant to a 4.2x change in its own seed, returning exactly 1.50x and 2.00x on
  injected zooms of the authors' own frames, and corroborated by an unrelated mechanism
  (human height);
- their **water surface is exactly a plane**, to the limit the format can express (depth
  round-trip 0.40% on 35523 px) — though its *outline* is beyond our primitive set;
- their **road is a plane** to 0.02–1.4% — our `Plane` is the right model for it.

Not safe to say:
- that we recover their *scenes*. Single-frame self-fit is 2.47% median depth error but
  13.3% at p90 (82% worst), with human IoU 0.62 and vegetation IoU 0.50 — and that is a
  self-fit with ~1000 free parameters per frame, not a prediction;
- anything about temporal stability, which this data cannot settle;
- that vegetation, humans, terrain or vehicles are our primitives. They are film-grade
  meshes; our primitive set approximates their visible surface and nothing more.

Suggested wording for the release: *"proxy conditions in the released format, at the
released intrinsics"* — not *"reproduction of the released scenes"*.

---

## 6. Probe -> full inverse rendering: what it would take

| step | work | why |
|---|---|---|
| ego-motion | 1–2 weeks | The blocker. Depth-only ICP hits 34% error by t=32. Needs either the source RGB video (not released) or a proper depth-odometry stack with keyframing, loop closure and bundle adjustment over the 124-frame chunks. |
| multi-view fusion | 1 week | Fit primitives to a *fused* cloud over many frames, not one view. Removes the visible-sheet degeneracy that made `Plane` beat everything, and is the only way any residual becomes an upper bound on shape error. Depends entirely on ego-motion. |
| render-in-the-loop fitting | 1–2 weeks | Replace least-squares fitting with direct optimisation of the rendered depth + semantic loss (finite differences over `render.py` is fine on CPU; the frame is 64512 px). Fixes the DOF artifact by adding an explicit complexity penalty. |
| class-specific models | 2–3 weeks | Water and road are solved (`Rect`, `Plane`). Humans need the articulated rig in `compiler/rig.py` driven by a pose fit, not blobs. Vegetation needs a size- and roughness-parameterised clump family rather than the fixed 18-lobe/spread-1.10 one. |
| gray16 pooling path | 1–2 days | `pool()` needs the second resample (box-mean in log space + categorical mode) to cover 25 of the 40 examples. Independent of everything else and worth doing regardless. |

Realistic total: **6–9 weeks**, and the first two items are strictly sequential. The
gray16 pooling path is the only item that pays off immediately.

---

## Files

```
probe_src/inverse/
  focal2.py        focal from two-frame rigidity (global search)   <- the estimator
  synth_check.py   initialisation-chained version -- BIASED, kept as the negative control
  synth3.py        ground-truth validation of focal2 at f=140/206/330/480
  focal_all.py     focal across all 40 examples
  falsify.py       gate-dependence and injected-zoom falsification
  human_f.py       independent focal cue from human height
  plane.py motion.py cond.py   ground-plane and conditioning diagnostics
  decodeB.py       gray16 decode, verified by falsification
  water.py         water-is-a-plane, with the quantisation-floor check
  veg.py           roughness v1 -- WRONG (quadratic in z); kept, the water control caught it
  veg2.py          roughness v2 (quadratic in 1/z) + renderer controls
  fitlib.py        unprojection, 3D components, raw primitive fits
  step4.py         render-scored primitive selection + full-scene score
  temporal.py      fit-on-frame-0, predict-frame-t, vs NULL and ORACLE
  attribute.py     warp-their-own-cloud control that overturned the temporal reading
  water_iou.py     resolved the water IoU 0.000 vs 'water is an exact plane' contradiction
  headline.py      the headline round-trip table
  poolproof.py     empirical proof the two families use different 2x2 pooling
```

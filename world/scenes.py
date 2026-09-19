"""One definition of the street take, shared by every caller.

The scene used to be written out separately in three harness steps, in batch.py and in
the tuning script. They drifted: the tuner measured head_diameter 0.219 while the step
that was supposed to be measuring the same thing read 0.244, because they were not the
same scene. Everything now builds through `street_take`, so a parameter tuned in one
place is the parameter used everywhere.

Defaults are the operating point fitted against out/released_reference.json.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from compiler.follow import FollowCamera
from .state import Entity, WorldState

FPS = 24
ACTIVE = ["waypoints", "locomotion", "separation", "fatigue"]


@dataclass
class StreetSpec:
    """Everything that varies between takes. Ranges are documented in batch.sample_spec."""

    seed: int = 33
    # The released intrinsic, measured from their depth maps by an estimator validated on
    # known-answer synthetic data and cross-checked by a second mechanism: 206.6 px at
    # 336x192, IQR [204, 207] over 97 estimates. This used to be 202.0, carried over from
    # an operating point fitted against a reference file that is not part of this
    # repository; 206.0 matches `compiler.render.Camera` and the measurement.
    focal_px: float = 206.0
    cam_height: float = 1.83
    aim_height: float = 1.20
    boom_len: float = 2.40
    # Where the subject sits across the frame. 0.5 centres it, which is what a follow boom
    # does on its own; the released prompts often say otherwise ("left of center", "lower
    # center-left"), and `world.shot` reads that out of the prompt.
    aim_screen_x: float = 0.5
    # Fitted on the shared scene against out/released_reference.json. head_diameter_m is
    # noticeably sensitive to these: the same rig reads 0.244 at tau 0.45 and 0.225 at
    # tau 0.70, because boom damping changes how much of the figure fills the frame.
    boom_tau: float = 0.70
    zeta: float = 0.50
    # Flight framing. 1.0 / 0.0 is a boom rigidly attached in Y, which is what made a climb
    # to 14 m look exactly like a walk; see FollowCamera._desired.
    # Fitted by sweeping both against the two criteria that already existed: the
    # protagonist inside the frustum (invariants.screen_bbox) and the fraction of depth
    # pixels that hit geometry (validate_conditions' 'valid', which fails under 5%).
    # A rigid boom (1.0 / 0.0) left 38 frames of the climb at 2% valid -- 98% empty sky.
    # pull_back 2.5 is the smallest that clears every frame with margin (min 9.5%) while
    # keeping the protagonist at a median 38 px of the 192 px DUV, ~154 px once the take
    # is rendered at 1344x768.
    rise_gain: float = 0.30
    pull_back: float = 2.50
    rise_knee: float = 2.50
    # Optical-axis clamp. The default +-35 deg suits a follow rig; an aerial shot needs to
    # look much further down, and their example 34 is a bird's-eye over terrain.
    pitch_min: float = -35.0
    pitch_max: float = 35.0
    base_speed: float = 2.5
    # Water. Class 2 is 6.144% of the released pixels and appears in 23 of their 40
    # examples; we emitted none of it until now. Terraces step away from the causeway on
    # the side opposite the road; the channel is the runoff strip along the near kerb.
    # n_terraces = 0 turns water off entirely for a dry scene.
    # 3, not 5: the 4th and 5th terraces change the water share by 0.06 percentage points
    # (8.70% vs 8.76%) because they sit behind the nearer ones in almost every frame. They
    # are dropped for expressing nothing extra, NOT for cost -- water's effect on take wall
    # time measured smaller than run-to-run variance.
    n_terraces: int = 3
    terrace_lat0: float = 11.0       # metres from the path to the first terrace edge
    terrace_step: float = 13.0       # lateral gap between successive terraces
    terrace_w: float = 11.0
    terrace_len: float = 440.0
    water_y: float = -0.02           # +Y is down, so this is 2 cm ABOVE the road plane
    channel_w: float = 1.1
    channel_lat: float = 3.1
    has_road: bool = True
    road_x: float = 7.0
    road_half_width: float = 3.5
    n_vehicles: int = 14
    n_peds: int = 3
    tree_step: float = 12.0          # metres between roadside clumps, NOT a count
    tree_lat: float = 9.0
    n_buildings: int = 5
    bldg_step: float = 60.0
    bldg_lat: tuple[float, float] = (18.0, 28.0)
    bldg_h: tuple[float, float] = (4.0, 9.0)
    n_backdrop: int = 11
    backdrop_dist: tuple[float, float] = (320.0, 900.0)
    backdrop_h: tuple[float, float] = (45.0, 130.0)
    path_len: float = 420.0
    waypoints: list = field(default_factory=lambda: [[0.0, 0.0, 60.0], [3.0, 0.0, 110.0],
                                                     [0.0, 0.0, 400.0]])


def build(spec: StreetSpec) -> WorldState:
    rng = np.random.default_rng(spec.seed)
    s = WorldState(seed=spec.seed)
    s.params = {"base_speed": spec.base_speed, "stride_m": 1.5, "separation_m": 1.6,
                "run_threshold": 2.0}
    if spec.has_road:
        s.params["road"] = {"half_width": spec.road_half_width, "half_length": 400.0,
                            "centre_z": 300.0, "x": spec.road_x}
    s.add(Entity("courier", "human", [0.0, 0.0, 0.0], speed=spec.base_speed,
                 attrs={"waypoints": [list(w) for w in spec.waypoints]}))
    for k in range(spec.n_peds):
        x = (-1) ** k * 2.2
        s.add(Entity(f"ped{k}", "human", [x, 0.0, 30.0 + 40.0 * k], heading_deg=180.0,
                     speed=1.2, attrs={"waypoints": [[x, 0.0, -40.0]]}))
    for k in range(spec.n_vehicles):
        s.add(Entity(f"veh{k}", "vehicle",
                     [spec.road_x if spec.has_road else 6.5, 0.0, 20.0 + 16.0 * k],
                     size=[4.2, 1.7, 4.6]))
    # Vegetation by spacing: a dense line pushes veg_frac past the released p90 whatever
    # the total count, because a short stretch of path is all the camera ever sees.
    for k in range(int(spec.path_len / spec.tree_step)):
        side = -1 if k % 2 else 1
        s.add(Entity(f"tree{k}", "vegetation", [side * spec.tree_lat, 0.0,
                                                5.0 + spec.tree_step * k],
                     size=[3.2, 3.8, 3.2]))
    for k in range(spec.n_buildings):
        side = -1 if k % 2 else 1
        s.add(Entity(f"bldg{k}", "scenery",
                     [side * float(rng.uniform(*spec.bldg_lat)), 0.0, 4.0 + spec.bldg_step * k],
                     size=[float(rng.uniform(6.0, 11.0)), float(rng.uniform(*spec.bldg_h)),
                           float(rng.uniform(6.0, 10.0))]))
    # Flooded terraces, stepping away from the causeway on the side away from the road, and
    # one runoff channel along the near kerb. Both are Rect sheets at class 2.
    side = -1.0 if spec.has_road and spec.road_x > 0 else 1.0
    for k in range(spec.n_terraces):
        lat = spec.terrace_lat0 + spec.terrace_step * k
        s.add(Entity(f"terrace{k}", "water",
                     [side * lat, spec.water_y, spec.path_len / 2.0],
                     size=[spec.terrace_w, 0.0, spec.terrace_len]))
    if spec.channel_w > 0:
        s.add(Entity("runoff", "water",
                     [side * spec.channel_lat, spec.water_y, spec.path_len / 2.0],
                     size=[spec.channel_w, 0.0, spec.terrace_len]))

    # Distant backdrop. An infinite ground plane puts only ~0.8% of the frame past 256 m
    # from a 1.83 m camera, against a released far-clamp fraction of 3-18%: the released
    # scenes must contain tall geometry beyond the far plane.
    for k in range(spec.n_backdrop):
        ang = 2 * np.pi * k / max(spec.n_backdrop, 1)
        dist = float(rng.uniform(*spec.backdrop_dist))
        s.add(Entity(f"far{k}", "scenery",
                     [dist * float(np.sin(ang)), 0.0, dist * float(np.cos(ang))],
                     size=[float(rng.uniform(60.0, 160.0)), float(rng.uniform(*spec.backdrop_h)),
                           float(rng.uniform(60.0, 160.0))]))
    return s


def camera(spec: StreetSpec) -> FollowCamera:
    return FollowCamera(distance=spec.boom_len, boom_tau=spec.boom_tau, zeta=spec.zeta,
                        aim_tau=0.15, focal_px=spec.focal_px, height=spec.cam_height,
                        aim_height=spec.aim_height, rise_gain=spec.rise_gain,
                        pull_back=spec.pull_back, rise_knee=spec.rise_knee,
                        pitch_min=spec.pitch_min, pitch_max=spec.pitch_max,
                        aim_screen_x=spec.aim_screen_x)


def street_take(spec: StreetSpec | None = None):
    spec = spec or StreetSpec()
    return build(spec), camera(spec), spec

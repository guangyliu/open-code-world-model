"""Compile executable world state into the two channels the video model consumes.

Section 3.2 of the paper: the proxy expresses coarse spatiotemporal state that the
observation must obey, while text carries identity, appearance and action semantics.
Both are derived deterministically from the same state -- nothing is invented here.
"""
from __future__ import annotations

import zlib

import numpy as np

from compiler.caption import Caption, WINDOW_SECONDS
from compiler.render import (Box, Capsule, Plane, Rect, Sphere, SphereCluster,
                             vegetation_clump, wireframe_box)
from compiler import rig
from .state import KIND_TO_CLASS, WorldState

ROAD_LIFT = 0.02      # metres the road surface sits above the ground plane


class PrimitiveCache:
    """Rebuild only the entities whose transform changed.

    Static scenery and parked vehicles never move, so their primitives are built once and
    reused; only entities whose transform changed are rebuilt.

    The docstring used to claim rebuilding cost "roughly three times the ray tracing
    itself". Measured on the shipped scene it is about 1% -- see the figure in the commit
    that corrected this. The class is still worth having (it is free, and the saving grows
    with clump count), but it was being constructed fresh at every call site, so it had
    never cached anything at all.
    """

    def __init__(self):
        self._sig: dict[str, tuple] = {}
        self._prims: dict[str, list] = {}

    @staticmethod
    def _signature(e) -> tuple:
        """Everything `_entity_shapes` and the class override read.

        `kind` and `attrs["class_id"]` belong here because both change what is built
        without changing the transform: `_entity_shapes` branches on kind, and the
        override decides the semantic class. Leaving them out was harmless only while the
        cache was constructed fresh at every call and therefore never cached anything.
        Once it persists, an entity whose ROLE changes while it sits still keeps its old
        class forever -- geometry right, depth right, semantic channel stale -- and that is
        precisely the case the override exists for: a prop that becomes a conveyance when
        it is ridden, or a quadruped emitting class 8 because it is the protagonist.
        """
        return (round(e.position[0], 4), round(e.position[1], 4), round(e.position[2], 4),
                round(e.heading_deg, 3), round(float(e.attrs.get("gait_phase", 0.0)), 4),
                round(float(e.attrs.get("idle_phase", 0.0)), 4),
                round(float(e.attrs.get("scale", 1.0)), 4), tuple(e.size), e.alive,
                round(e.speed, 4), e.kind, e.attrs.get("class_id"))

    def build(self, s: WorldState, *, ground_class: int = 0) -> list:
        out = list(_ground(s, ground_class))
        live = set()
        for eid, e in s.entities.items():
            if not e.alive:
                continue
            live.add(eid)
            sig = self._signature(e)
            if self._sig.get(eid) != sig:
                self._sig[eid] = sig
                self._prims[eid] = _entity_primitives(e)
            out.extend(self._prims[eid])
        for dead in set(self._prims) - live:
            self._prims.pop(dead, None)
            self._sig.pop(dead, None)
        return out


def _ground(s: WorldState, ground_class: int) -> list:
    out = [Plane(np.zeros(3), np.array([0.0, -1.0, 0.0]), class_id=ground_class)]
    road = s.params.get("road")
    if road:
        # Top face flush with y = 0, the plane every entity's feet sit on. Centring the
        # slab at -0.03 with half-height 0.06 lifted its surface 0.09 m above the ground,
        # which is exactly the 1.83 -> 1.74 m error the camera-height descriptor showed.
        half = np.array([road["half_width"], 0.06, road["half_length"]])
        # +Y is DOWN. Centring at +half sinks the slab so its top face sits at y = 0 --
        # but exactly coplanar with the ground plane, which z-fights: the road broke into
        # 306 connected components, 297 of them 1-3 px of speckle. Lift it by ROAD_LIFT so
        # it wins unambiguously. 2 cm is 1% of the camera height, far below the 9 cm error
        # the original floating slab caused.
        out.append(Box(np.array([road.get("x", 0.0), half[1] - ROAD_LIFT, road["centre_z"]]),
                       half, class_id=4))
    return out


def _entity_primitives(e) -> list:
    """Primitives for one entity, with an optional semantic-class override.

    The override used to apply to props only. It is general because class is about ROLE,
    not shape: released example 31 has a white stag as its protagonist and still classes it
    8, so class 8 encodes "the protagonist" rather than "a human". Reproducing that scene
    needs a quadruped emitting 8.
    """
    prims = _entity_shapes(e)
    cid = e.attrs.get("class_id")
    if cid is not None:
        for q in prims:
            q.class_id = int(cid)
    return prims


def _entity_shapes(e) -> list:
    p = e.pos.copy()
    if e.kind == "human":
        # p[1] is altitude (+Y down), so a rider or a flyer sits above the road. Ground
        # entities carry p[1] == 0 and are unchanged.
        walking = abs(e.speed) > 1e-9
        idle = float(e.attrs.get("idle_phase", 0.0))
        pose = (rig.walk_pose(e.attrs.get("gait_phase", 0.0), run=e.speed > 2.0)
                if walking else rig.idle_pose(idle))
        facing = e.heading_deg + (0.0 if walking else rig.idle_turn(idle))
        return rig.Human(position=np.array([p[0], p[1], p[2]]), facing_deg=facing,
                         pose=pose,
                         scale=float(e.attrs.get("scale", 1.0))).primitives()
    if e.kind == "vehicle":
        half = np.asarray(e.size, float) / 2.0
        th = np.radians(e.heading_deg)
        axes = np.array([[np.cos(th), 0, -np.sin(th)], [0, 1, 0], [np.sin(th), 0, np.cos(th)]])
        # p[1] is altitude: +Y is down, so a body flying `alt` metres up has p[1] = -alt.
        # Ground entities carry p[1] == 0 and land exactly where they did before.
        return wireframe_box([p[0], p[1] - half[1], p[2]], half, axes)
    if e.kind == "scenery":
        half = np.asarray(e.size, float) / 2.0
        th = np.radians(e.heading_deg)
        axes = np.array([[np.cos(th), 0, -np.sin(th)], [0, 1, 0], [np.sin(th), 0, np.cos(th)]])
        return [Box(np.array([p[0], p[1] - half[1], p[2]]), half, axes, class_id=0)]
    if e.kind == "animal":
        # A quadruped from the same swept-sphere vocabulary as the human rig: barrel body,
        # four legs, angled neck, head, and a hump that makes the silhouette read as a camel
        # rather than a horse.
        L, Hh, Wd = float(e.size[2]), float(e.size[1]), float(e.size[0])
        th = np.radians(e.heading_deg)
        fwd = np.array([np.sin(th), 0.0, np.cos(th)])
        side = np.array([np.cos(th), 0.0, -np.sin(th)])
        base = np.array([p[0], p[1], p[2]])
        back = base + np.array([0.0, -Hh * 0.62, 0.0])
        body_r = Wd * 0.34
        out = [Capsule(back - fwd * L * 0.28, back + fwd * L * 0.22, body_r, class_id=9),
               Sphere(back + fwd * L * 0.02 + np.array([0.0, -body_r * 0.85, 0.0]),
                      body_r * 0.78, class_id=9)]
        # Diagonal walk: the near-fore and off-hind legs swing together, a quarter cycle
        # ahead of the other pair. Amplitude 0.11*L puts a 2.5 m stag's hoof through about
        # a 0.55 m fore-aft arc, which is the stride such an animal covers at 4.2 m/s. The
        # branch had no animation term at all before: the only trig in it was heading.
        gp = 2 * np.pi * float(e.attrs.get("gait_phase", 0.0))
        for sx in (-1.0, 1.0):
            for fz, lean in ((0.20, 0.03), (-0.24, -0.03)):
                diag = 0.0 if (sx > 0) == (fz > 0) else np.pi
                swing = 0.11 * L * float(np.sin(gp + diag))
                top = back + fwd * L * fz + side * (Wd * 0.28 * sx)
                foot = top + np.array([0.0, Hh * 0.62, 0.0]) + fwd * (lean * L + swing)
                out.append(Capsule(top, foot, body_r * 0.30, class_id=9))
        # At rest the legs are still, so the neck and head carry the idle instead.
        nod = np.radians(4.0) * float(np.sin(2 * np.pi * float(e.attrs.get("idle_phase", 0.0))))
        neck0 = back + fwd * L * 0.30
        neck1 = neck0 + fwd * L * 0.16 + np.array([0.0, -Hh * (0.34 + 0.06 * nod), 0.0])
        out.append(Capsule(neck0, neck1, body_r * 0.42, class_id=9))
        out.append(Sphere(neck1 + fwd * L * 0.07, body_r * 0.40, class_id=9))
        return out
    if e.kind == "prop":
        # A slender shaft along the entity's heading -- the broom.
        L = float(e.size[2])
        th = np.radians(e.heading_deg)
        fwd = np.array([np.sin(th), 0.0, np.cos(th)])
        c = np.array([p[0], p[1], p[2]])
        # class 11 was ours alone -- 0 of the released 40 use it. A ridden prop is a
        # conveyance (class 10 in their vocabulary); anything else falls back to 0.
        cid = KIND_TO_CLASS["prop"]
        return [Capsule(c - fwd * L * 0.5, c + fwd * L * 0.5, float(e.size[0]) * 0.5, class_id=cid),
                SphereCluster(np.stack([c + fwd * L * (0.5 + 0.03 * k)
                                        + np.array([0.0, 0.02 * ((k % 3) - 1), 0.0]) for k in range(6)]),
                              float(e.size[0]) * 0.9, class_id=cid)]
    if e.kind == "water":
        # size is (width, _, length); position is the centre of the sheet. The y of the
        # entity is the surface height, so a scene can sink a canal or raise a flooded
        # terrace without a new kind.
        # Tiled along Z rather than emitted as one long sheet. The screen-space cull works
        # on a bounding sphere, and a single 11 x 440 m terrace has a ~220 m radius that no
        # frame can reject. Measured, tiling changes nothing at this scale (a back-to-back
        # A/B put water's whole effect on take wall time inside run-to-run variance:
        # dry 23.8/17.0 s against wet 19.1/19.3 s). It is kept because the loose bound is a
        # property of the shape, not of this scene, and a longer river would feel it.
        w, L = float(e.size[0]), float(e.size[2])
        cid = KIND_TO_CLASS["water"]
        n = max(1, int(round(L / 40.0)))
        step = L / n
        z0 = p[2] - L / 2.0
        return [Rect(y=float(p[1]), x0=p[0] - w / 2.0, x1=p[0] + w / 2.0,
                     z0=z0 + step * k, z1=z0 + step * (k + 1), class_id=cid)
                for k in range(n)]
    if e.kind == "vegetation":
        return vegetation_clump([p[0], p[1] - e.size[1] / 2.0, p[2]], e.size[0] / 2.0,
                                aspect=float(e.size[1] / max(e.size[0], 1e-6)),
                                seed=_stable_seed(e.eid))
    return []


def _stable_seed(eid: str) -> int:
    """Deterministic across processes. Python's hash() is salted per interpreter, so it
    made every rendered frame differ run to run, while the determinism check -- which
    compared only the state JSON -- kept passing."""
    return zlib.crc32(eid.encode("utf-8")) % 10_000



# One slow idle cycle per four seconds. The window is 124 frames at 24 fps, so a planted
# subject moves through a little over one cycle -- continuous motion rather than a loop the
# viewer can count.
#
# Recorded, not acted on: three of their five planted windows never repeat a frame at all,
# where ours repeats once (frames 96-123 duplicate 0-27, which is why ex36 renders 96
# distinct frames rather than 124). A real divergence, but not a rule -- their ex37 repeats
# at 52 frames. An aperiodic idle, drifting the phase or running two incommensurate beats,
# would close it and is a change for its own day.
IDLE_HZ = 0.25


def advance_gait(s: WorldState) -> None:
    """Keep each walker's gait phase in step with the distance it covered, and give a
    subject that is standing still an idle phase so it never renders as a still frame.

    Animals were omitted entirely: this iterated `of_kind("human")`, so a quadruped never
    received a phase at all and ex31's stag walked 21.5 m with rigid legs."""
    for e in list(s.of_kind("human")) + list(s.of_kind("animal")):
        stride = s.params.get("stride_m", 1.5)
        e.attrs["gait_phase"] = float(e.attrs.get("gait_phase", 0.0)) + e.speed * s.dt / stride
        # A planted subject covers no distance, so gait gives it nothing and it renders as
        # one frame for the whole window. The idle channel runs on wall time instead, and
        # ONLY at rest: a walking subject never receives an idle phase, which is what keeps
        # every moving take bit-identical.
        if abs(e.speed) <= 1e-9:
            e.attrs["idle_phase"] = float(e.attrs.get("idle_phase", 0.0)) + IDLE_HZ * s.dt


def caption_for(s: WorldState, appearance: dict, actions: list[str],
                *, continuation: bool) -> Caption:
    """Assemble one window's caption. Prose comes from the agent; timing comes from state."""
    protagonist = appearance["protagonist"]
    if continuation:
        protagonist = "Continuing without a cut, " + protagonist[0].lower() + protagonist[1:]
    others = describe_others(s)
    return Caption(protagonist=protagonist, scene=appearance["scene"], actions=actions,
                   held_objects=appearance.get("held_objects", ""),
                   other_actors_appearance=appearance.get("other_actors_appearance", ""),
                   other_actors=others)


def describe_others(s: WorldState, hero: str = "courier") -> str:
    """Timeline entry for non-protagonists, split in half like the released captions."""
    movers = [e for e in s.entities.values()
              if e.alive and e.eid != hero and e.kind in ("human", "vehicle", "animal") and e.speed]
    statics = [e for e in s.entities.values()
               if e.alive and e.eid != hero and e.kind == "vehicle" and not e.speed]
    half = round(WINDOW_SECONDS / 2, 2)
    dash = "\u2013"
    parts = []
    if movers:
        parts.append(
            f"0.00{dash}{half:.2f}s, {len(movers)} other moving actors: they hold their own lines "
            f"along the street at a steady, unhurried pace, neither converging on the protagonist "
            f"nor reacting to her passage; their spacing relative to one another stays roughly "
            f"constant and none of them changes direction within the window.")
    if statics:
        parts.append(
            f"{(half if movers else 0.0):.2f}{dash}{WINDOW_SECONDS:.2f}s, {len(statics)} parked "
            f"vehicles: they remain stationary along the kerb with no motion of their own, their "
            f"wheels still and their bodies square to the roadway, passing out of frame behind the "
            f"protagonist one after another as she advances down the street.")
    if not parts:
        return (f"0.00{dash}{WINDOW_SECONDS:.2f}s, no other visible actors: the protagonist remains "
                f"alone throughout, with only ambient environmental motion in the vegetation and "
                f"suspended dust, and no other person, creature, craft, or independently moving "
                f"identity enters the frame at any point in the window.")
    return " ".join(parts)

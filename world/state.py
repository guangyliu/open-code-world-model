"""Executable world state: the half of the model the coding agent owns.

The paper splits state into S_exe (entities, rules, relations, history -- operated on
directly by code) and S_vis (appearance and motion, produced by the video model). This
module is S_exe. It is deliberately serialisable: persistence across windows, including
for entities that are off-screen, is the claim that separates this from a video model
that only sees what the camera saw.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# Semantic classes the proxy can express (SPEC section 4: only 8 of 12 are ever used).
# The released 40 use exactly {0, 1, 2, 4, 5, 8, 9, 10}; 3, 6, 7 and 11 appear in none of
# them. Nothing here may map outside that set, and `prop` defaults to 0 rather than the 11
# it used to emit. An entity may override with attrs["class_id"] when its role differs from
# its shape -- a ridden broom is a conveyance, so class 10.
KIND_TO_CLASS = {"human": 8, "vehicle": 10, "vegetation": 5, "animal": 9,
                 "road": 4, "water": 2, "scenery": 0, "prop": 0}
RELEASED_CLASSES = frozenset({0, 1, 2, 4, 5, 8, 9, 10})


@dataclass
class Entity:
    """One addressable thing. `attrs` carries whatever the agent's rules need."""

    eid: str
    kind: str
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    heading_deg: float = 0.0
    speed: float = 0.0
    size: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    alive: bool = True
    attrs: dict = field(default_factory=dict)

    @property
    def pos(self) -> np.ndarray:
        return np.asarray(self.position, float)

    @pos.setter
    def pos(self, v) -> None:
        self.position = [float(x) for x in v]

    def forward(self) -> np.ndarray:
        th = np.radians(self.heading_deg)
        return np.array([np.sin(th), 0.0, np.cos(th)])


@dataclass
class Event:
    """A thing that happened. History is what lets consequences outlive the frame."""

    t: float
    kind: str
    subject: str
    detail: dict = field(default_factory=dict)


class WorldState:
    """Entities, rule parameters, and event history, advanced by code at a fixed step."""

    def __init__(self, seed: int = 0, dt: float = 1.0 / 24.0):
        self.t = 0.0
        self.dt = dt
        self.seed = seed
        self.entities: dict[str, Entity] = {}
        self.events: list[Event] = []
        self.params: dict = {}
        self.rng = np.random.default_rng(seed)

    # -- entities -----------------------------------------------------------
    def add(self, e: Entity) -> Entity:
        if e.eid in self.entities:
            raise ValueError(f"duplicate entity id {e.eid}")
        self.entities[e.eid] = e
        return e

    def of_kind(self, kind: str) -> list[Entity]:
        return [e for e in self.entities.values() if e.kind == kind and e.alive]

    def get(self, eid: str) -> Entity:
        return self.entities[eid]

    def log(self, kind: str, subject: str, **detail) -> None:
        self.events.append(Event(round(self.t, 3), kind, subject, detail))

    def since(self, t0: float) -> list[Event]:
        return [e for e in self.events if e.t >= t0]

    # -- persistence --------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps({"t": self.t, "dt": self.dt, "seed": self.seed,
                           "rng": self.rng.bit_generator.state,
                           "params": self.params,
                           "entities": {k: asdict(v) for k, v in self.entities.items()},
                           "events": [asdict(e) for e in self.events]},
                          indent=1, default=str)

    @classmethod
    def from_json(cls, text: str) -> "WorldState":
        d = json.loads(text)
        s = cls(seed=d["seed"], dt=d["dt"])
        s.t = d["t"]
        s.params = d["params"]
        s.entities = {k: Entity(**v) for k, v in d["entities"].items()}
        s.events = [Event(**e) for e in d["events"]]
        s.rng.bit_generator.state = _coerce_state(d["rng"])
        return s

    def save(self, path: Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "WorldState":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def _coerce_state(st: dict) -> dict:
    """json round-trips numpy's bit-generator ints as strings; put them back."""
    out = dict(st)
    inner = dict(out.get("state", {}))
    for k, v in inner.items():
        if isinstance(v, str) and v.isdigit():
            inner[k] = int(v)
    if inner:
        out["state"] = inner
    for k in ("has_uint32", "uinteger"):
        if isinstance(out.get(k), str) and out[k].isdigit():
            out[k] = int(out[k])
    return out

"""The coding-agent boundary.

The paper's loop: the agent reads world state, then either invokes existing mechanisms
or *modifies the world program*, and code advances the dense state in between. The
second half is what makes this a coding agent rather than a controller, so the interface
has to accept new executable rules, not just parameter changes.

`ScriptedAgent` implements the interface deterministically so the whole pipeline is
testable without credentials. An LLM-backed agent is a drop-in replacement: same
`decide()` signature, prose and rule source produced by the model instead.

SECURITY: `AddRule` executes agent-authored Python. With an LLM on the other end that is
arbitrary model-generated code execution. The namespace below is restricted, which is a
speed bump and not a sandbox -- run the real thing in a container with no network and no
credentials mounted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from . import rules
from .state import Entity, WorldState


@dataclass
class SetParam:
    key: str
    value: float


@dataclass
class SetGoal:
    eid: str
    waypoints: list[list[float]]


@dataclass
class Spawn:
    entity: Entity


@dataclass
class AddRule:
    """New executable mechanism, as source. This is the agent editing the world program."""

    name: str
    source: str


@dataclass
class EnableRule:
    name: str
    enabled: bool = True


Action = SetParam | SetGoal | Spawn | AddRule | EnableRule


class Agent(Protocol):
    def decide(self, s: WorldState, window: int) -> list[Action]: ...
    def narrate(self, s: WorldState, window: int, segments: int) -> list[str]: ...


SAFE_BUILTINS = {"abs": abs, "min": min, "max": max, "round": round, "len": len,
                 "range": range, "float": float, "int": int, "enumerate": enumerate,
                 "sorted": sorted, "sum": sum, "list": list, "dict": dict}


def apply(s: WorldState, actions: list[Action], active: list[str],
          ruleset: "rules.RuleSet") -> list[str]:
    """Apply the agent's decisions; returns the updated active-rule list.

    `ruleset` is required, and is where AddRule puts its rule. It used to register into a
    process-global dict, which leaked every rule into the next take in the same process.
    """
    active = list(active)
    for a in actions:
        if isinstance(a, SetParam):
            s.params[a.key] = a.value
            s.log("param_changed", "world", key=a.key, value=a.value)
        elif isinstance(a, SetGoal):
            s.get(a.eid).attrs["waypoints"] = a.waypoints
            s.get(a.eid).attrs["wp_index"] = 0
            s.log("goal_changed", a.eid, n=len(a.waypoints))
        elif isinstance(a, Spawn):
            s.add(a.entity)
            # `log(kind, subject, **detail)` -- the entity's own kind cannot ride in as
            # `kind=`, it collides with the event kind. This path had never been run.
            s.log("spawned", a.entity.eid, entity_kind=a.entity.kind)
        elif isinstance(a, AddRule):
            ns = {"__builtins__": SAFE_BUILTINS, "np": __import__("numpy"),
                  "WorldState": WorldState, "Entity": Entity}
            exec(compile(a.source, f"<agent:{a.name}>", "exec"), ns)      # noqa: S102
            fn = ns.get(a.name)
            if not callable(fn):
                raise ValueError(f"rule source must define a callable named {a.name!r}")
            ruleset.add(a.name, fn)
            if a.name not in active:
                active.append(a.name)
            s.log("rule_added", "world", name=a.name)
        elif isinstance(a, EnableRule):
            if a.enabled and a.name not in active:
                active.append(a.name)
            elif not a.enabled and a.name in active:
                active.remove(a.name)
            s.log("rule_toggled", "world", name=a.name, enabled=a.enabled)
    return active


@dataclass
class ScriptedAgent:
    """A deterministic stand-in for the LLM, so the loop is testable end to end."""

    appearance: dict
    plan: dict[int, list[Action]] = field(default_factory=dict)

    def decide(self, s: WorldState, window: int) -> list[Action]:
        return self.plan.get(window, [])

    def narrate(self, s: WorldState, window: int, segments: int) -> list[str]:
        """Turn the window's state and events into per-segment action prose."""
        hero = s.get("courier")
        fat = float(hero.attrs.get("fatigue", 0.0))
        pace = ("drives forward at a full run" if hero.speed > 2.2 else
                "runs at a steady, economical pace" if hero.speed > 1.4 else
                "slows to a jog, her stride shortening")
        tired = (" Her shoulders have begun to drop and her arm swing is looser than before."
                 if fat > 0.45 else "")
        out = []
        for k in range(segments):
            out.append(
                f"The courier {pace} along the paving, her satchel swinging back against her left "
                f"hip as her trailing leg extends behind her; her arms drive in opposition close to "
                f"her ribs, her head stays level and fixed on the line ahead, and the awning shadows "
                f"slide across her back in bands.{tired}")
        return out

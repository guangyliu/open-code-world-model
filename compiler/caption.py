"""Compile world state into the released caption format.

Step `caption_schema` measured all 412 released prompts: every one uses the same three
sections in the same order, spans exactly 5.17 s, and starts its timeline at 0.00 s --
including continuation windows, whose timeline is window-local, not take-global. The
bundled config.multiwindow.example.json shows "[3.75s-8.92s]"; that is an author-written
placeholder, not the format the model was trained on.

The compiler owns the structure. Content -- identities, appearance, action semantics --
comes from the coding agent, which is exactly the split the paper describes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

WINDOW_FRAMES = 124
FPS = 24
WINDOW_SECONDS = round(WINDOW_FRAMES / FPS, 2)      # 5.17, as in every released prompt

SECTIONS = ("Appearance and scene", "Protagonist action timeline", "Other visible actors")
# Character budgets from the released distribution (median, p10, p90).
BUDGET = {"Appearance and scene": (3072, 2432, 4236),
          "Protagonist action timeline": (2097, 1465, 2752),
          "Other visible actors": (307, 209, 876)}


@dataclass
class Caption:
    """Structured content for one window. Every field is prose written by the agent."""

    protagonist: str                 # build, identity-stable features, starting posture
    scene: str                       # setting, lighting, palette, and the camera's behaviour
    actions: list[str] = field(default_factory=list)   # one per timeline segment
    held_objects: str = ""           # objects carried or directly manipulated
    other_actors_appearance: str = ""  # or an explicit statement that there are none
    other_actors: str = ""           # timeline entry for everyone who is not the protagonist

    def segments(self) -> list[tuple[float, float]]:
        """Split the window into len(actions) equal spans; released uses 6-10, mode 7."""
        n = len(self.actions)
        if not 4 <= n <= 12:
            raise ValueError(f"timeline needs 4-12 segments, got {n}")
        edges = [round(WINDOW_SECONDS * k / n, 2) for k in range(n + 1)]
        edges[-1] = WINDOW_SECONDS
        return list(zip(edges[:-1], edges[1:]))

    def render(self) -> str:
        """Emit the exact three-section layout, CRLF-normalised by the caller."""
        para = [self.protagonist]
        if self.held_objects:
            para.append(self.held_objects)
        para.append(self.other_actors_appearance or
                    "No other person, creature, craft, or independently moving identity is visible.")
        para.append(self.scene)

        timeline = "; ".join(
            f"{a:.2f}–{b:.2f}s: {text.rstrip().rstrip('.')}."
            for (a, b), text in zip(self.segments(), self.actions))

        others = self.other_actors or (
            f"0.00–{WINDOW_SECONDS:.2f}s, no other visible actors: the protagonist "
            f"remains alone throughout, with only ambient environmental motion.")

        return (f"{SECTIONS[0]}:\n\n" + "\n\n".join(para) +
                f"\n\n{SECTIONS[1]}:\n\n{timeline}\n\n"
                f"{SECTIONS[2]}:\n\n{others}")


def to_prompt(cap: Caption) -> str:
    """Released prompts are consumed with CRLF line endings (see materialize script)."""
    return cap.render().replace("\r\n", "\n").replace("\n", "\r\n")


def budget_report(cap: Caption) -> dict:
    """Per-section length against the released distribution, so drift is visible."""
    text = cap.render()
    out = {}
    for k, name in enumerate(SECTIONS):
        start = text.index(name + ":") + len(name) + 1
        end = text.index(SECTIONS[k + 1] + ":") if k + 1 < len(SECTIONS) else len(text)
        n = len(text[start:end].strip())
        med, lo, hi = BUDGET[name]
        out[name] = {"chars": n, "median": med, "in_p10_p90": lo <= n <= hi}
    return out

"""Read the shot out of a released prompt, so the scene author cannot ignore it.

The proxy used to place every protagonist at horizontal centroid 0.49-0.50 with essentially
zero variance across six reproduced examples, while the released six sit at 0.350, 0.370,
0.358, 0.418, 0.499, 0.365. That was not a tuning error. A follow boom solves exactly one
constraint -- keep the subject projected at frame centre -- so 0.49 is that equation's
fixed point. The camera moved constantly; it moved to cancel the subject's motion.

Meanwhile the prompts say where the subject goes. "She stands left of center", "his large
lower center-left placement", "keeping the animal low and central", "the sentinel centered
prominently". Nothing read those words.

WHAT IS ACTUALLY PARSEABLE, scored over all 40 released window-0 prompts rather than the
six in the figures, because a pattern that only fires on the figures is a per-example hack
wearing a regex:

    camera "... begins ..." sentence      40/40   universal anchor
    subject placement (centred/off-centre) 27/40  the dominant signal
    vertical placement (lower/upper)       14/40
    height (shoulder / elevated / high)    13/40
    hold the opening layout                 6/40  minority, and the one we were asked for
    aerial / downward-looking               3/40
    push in / advance                       7/40
    rise / crane                            7/40
    orbit or arc                            1/40
    follow / track named explicitly         2/40
    pull back                               0/40

So the shot clause is mostly STATIC COMPOSITION, not camera choreography: where the subject
sits, how high the lens is, what stays in view. The motion vocabulary the camera modes were
built for is rare in window 0. That is a finding about the released data, not about us.

The placement words map to standard framing positions, NOT to the measured centroids:
"centred" is the middle, "centre-left" and "left of centre" are the left rule-of-thirds
line at 1/3. Choosing 1/3 from the words is what keeps this honest -- their six land at
0.35-0.37, which corroborates the mapping without having set it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Screen fractions for the composition words. Rule of thirds, chosen from the vocabulary.
CENTRE = 0.50
THIRD_LEFT = 1.0 / 3.0
THIRD_RIGHT = 2.0 / 3.0
# Vertical, same rule and same reasoning: "lower center", "lower center-left" name the
# lower rule-of-thirds line. Chosen from the vocabulary, not fitted to a measurement.
LOWER_THIRD = 2.0 / 3.0
UPPER_THIRD = 1.0 / 3.0


def aim_y_for(sh: "Shot") -> float:
    """Where the subject sits vertically in frame, as a screen fraction.

    `subject_low` parses for 14 of the 40 prompts and used to be wired to nothing. Scenes
    faked it with `aim_height`, which is a different quantity -- the point ON THE SUBJECT
    the lens looks at -- and dragging that to the feet tips the optical axis into its clamp
    and stops the framing responding at all. ex28 and ex31 sat at pitch -35.00, the clamp,
    for all 124 frames, which is also why neither contained any sky.
    """
    return LOWER_THIRD if sh.subject_low else CENTRE


@dataclass
class Shot:
    """What a prompt says about the shot. `None` means the prompt did not say."""

    subject_x: float | None = None      # target horizontal screen fraction
    subject_low: bool = False           # "lower ..." / "low and central"
    hold: bool = False                  # camera preserves the opening layout
    hold_then_follow: bool = False      # "... before following ..."
    aerial: bool = False                # steep downward-looking view
    elevated: bool = False              # camera above the subject
    shoulder_height: bool = False       # explicit "shoulder height"
    source: str = ""                    # the sentence the parse came from

    def parsed(self) -> list[str]:
        out = []
        if self.subject_x is not None:
            out.append("subject_x=%.3f" % self.subject_x)
        for f in ("subject_low", "hold", "hold_then_follow", "aerial", "elevated",
                  "shoulder_height"):
            if getattr(self, f):
                out.append(f)
        return out


def mode_for(sh: "Shot") -> str:
    """The camera mode a parsed shot asks for.

    "preserving / holding the opening layout" reads as a frozen lens and is not one. The
    released conditions hold the subject at a constant distance through exactly that span
    -- ex30 1.72-1.76 m and ex28 1.75-1.86 m over the first 31 frames -- so the layout
    holds because the camera moves WITH the subject. ex30's prompt says it outright: "the
    camera follows at a nearly constant distance with a gentle walking bob", in the same
    sentence group as "preserving his large lower center-left placement".

    So the clause maps to `framed`: a follow that holds stand-off and screen position with
    the boom azimuth frozen. `lock` remains for clauses that describe a genuinely fixed
    lens, such as a steady vantage held while the subject rotates in place.

    This lives here, rather than in the scene script, so the routing is covered by
    `t_framed_holds_the_opening_layout` instead of being a line nothing tests.
    """
    return "framed" if sh.hold else "follow"


_SCENE = ("Appearance and scene:", "Protagonist action timeline:")

# Ordered: the more specific phrasing wins. All are anchored on the subject or the camera
# sentence, never on a bare direction word, because "screen-left" appears constantly as a
# property of scenery.
_PLACEMENT = [
    (r"\b(?:left of cent(?:er|re)|cent(?:er|re)-left)\b", THIRD_LEFT),
    (r"\b(?:right of cent(?:er|re)|cent(?:er|re)-right)\b", THIRD_RIGHT),
    (r"\bcent(?:er|re)ed\b|\bcentral\b|\bcent(?:er|re) foreground\b", CENTRE),
]
_HOLD = (r"\b(?:preserv\w+|holding|keeping)\b[^.]{0,90}?"
         r"\b(?:layout|placement|composition|framing|ordering)\b")
_HOLD_EXACT = r"\b(?:preserv\w+|holding|keeping)\s+(?:the\s+)?(?:exact|opening|initial)\b"


def scene_block(prompt: str) -> str:
    if _SCENE[0] in prompt:
        prompt = prompt.split(_SCENE[0], 1)[1]
    if _SCENE[1] in prompt:
        prompt = prompt.split(_SCENE[1], 1)[0]
    return prompt


def camera_sentence(prompt: str) -> str:
    """The '... camera ... begins ...' sentence, which all 40 released prompts have."""
    for s in re.split(r"(?<=[.])\s+", scene_block(prompt)):
        low = s.lower()
        if "camera" in low and "begins" in low:
            return s.strip()
    return ""


def parse_shot(prompt: str) -> Shot:
    """Extract the shot. Only what the text states; anything unsaid stays None/False."""
    scene = scene_block(prompt)
    cam = camera_sentence(prompt)
    low_all = scene.lower()
    sh = Shot(source=cam)

    # Placement: prefer the camera sentence, fall back to the wider scene block, because
    # some prompts state the placement while describing the subject rather than the lens.
    for hay in (cam.lower(), low_all):
        for pat, val in _PLACEMENT:
            if re.search(pat, hay):
                sh.subject_x = val
                break
        if sh.subject_x is not None:
            break

    sh.subject_low = bool(re.search(r"\blow(?:er)?\b[^.]{0,40}\b(cent|left|right|fore)",
                                    low_all)
                          or re.search(r"\blow and (?:wide|central)\b", low_all))
    sh.hold = bool(re.search(_HOLD, low_all) or re.search(_HOLD_EXACT, low_all))
    sh.hold_then_follow = bool(re.search(r"\bbefore\s+following\b", low_all))
    sh.aerial = bool(re.search(r"\b(?:aerial|downward-looking|looking down)\b", low_all))
    sh.elevated = bool(re.search(r"\b(?:elevated|very high|above)\b[^.]{0,60}\bcamera\b",
                                 low_all)
                       or re.search(r"\bcamera\b[^.]{0,60}\b(?:above|elevated)\b", low_all))
    sh.shoulder_height = "shoulder height" in low_all
    return sh

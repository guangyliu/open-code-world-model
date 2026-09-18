"""Scenario prose for the rainkeeper take: the parts that do not depend on the action.

Compiled captions need three things no simulation can supply -- who he is, where this is,
and what the ambient traffic looks like. They are lifted verbatim from the captions that
were hand-written for this take; only the sentence asserting his posture was removed,
because `world/narrate.py` now derives that from the simulation.

Everything describing WHAT HE DOES lives in narrate.py and is compiled per window. Nothing
here may state an action, or the drift this whole change removes comes straight back.
"""

PLACE = 'the elevated stone spillway above the flooded terraces'

IDENTITY = (
    'The protagonist is the reed-masked rainkeeper, an adult humanlike figure with a lean '
    'build concealed beneath layered weather gear. His smooth jade-green masklike face, '
    'shadowed eyes, and broad conical woven-reed hat are stable identity features. A dark '
    'indigo scarf wraps his neck beneath a long mantle of rain-blackened reeds that falls '
    'in ragged strands from his shoulders to his calves. Layered indigo robes cross '
    'beneath the mantle and are secured by a coarse rope belt. Wrapped dark gloves cover '
    'his hands, while loose black trousers and sturdy wrapped boots complete his '
    'clothing. Rain beads across the reeds, saturates the cloth to nearly black blue, and '
    'gives the mask a cold muted sheen.')

SETTING = (
    'The broad wet causeway runs horizontally behind him, its dark rectangular paving '
    'stones divided by long pale channels of streaming runoff. A low moss-dark parapet '
    'borders the far edge. Beyond it, flooded emerald terraces descend through the valley '
    'in layered steps, each ledge spilling thin silver curtains of rainwater. Small tiled '
    'shrine roofs rise from the terraces, while tall weathered rain-conduit pillars and '
    'taut dark cables cross the middle distance. Mountain ridges recede beneath a dense '
    'ceiling of charcoal storm clouds. Slanting rain, soft gray daylight, and restrained '
    'highlights across the flooded masonry maintain moody classical oil-painting realism.')

OTHERS_APPEARANCE = (
    'Low lacquered rain-carts occasionally traverse the distant causeway. Each has a '
    'compact enclosed body, dark wooden panels polished by rain, rounded metal-rimmed '
    'wheels, narrow amber lamps, and a low roof shaped to shed water. They remain '
    'secondary to the rainkeeper and preserve the roadway scale and direction of travel.')

# eid -> how the caption names it in "Other visible actors".
ACTOR_NAMES = {
    "taxi": "the lacquered rain-cart",
    "camel": "the tall pale-coated camel",
    "gunship": "the grey gunship",
    "ord0": "the falling ordnance",
}


def content() -> dict:
    return {"identity": IDENTITY, "scene": SETTING, "place": PLACE,
            "others_appearance": OTHERS_APPEARANCE, "actor_names": ACTOR_NAMES}

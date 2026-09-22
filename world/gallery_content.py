"""Scenario prose for the six gallery scenes: who he is, where this is, how the captions
phrase each state. Nothing here states an action -- `world/narrate.py` compiles those from
the simulation, and each phrase table below carries the same SIGNATURES the gate reads back.

The identity and setting paragraphs are condensed from the released prompts of the
examples whose first frames anchor these scenes (Apache-2.0, attributed in NOTICE). The
geometry, the actions and the camera are ours.
"""
from world.narrate import (STAND, WALK, JUMP, BOARD, RIDE_VEHICLE, DRIVE, MOUNT, RIDE_MOUNT,
                           FLY, A_HALT, A_NEAR, A_AWAY, A_DESC, A_FALL, A_CARRY, A_ALONG,
                           PHRASES as _RAIN_PHRASES, OPENING as _RAIN_OPENING)


def _phrases(stand, walk, jump, board=None, ride=None, drive=None):
    """Fill the states a scene does not use from the rainkeeper tables so the audit still
    sees every signature exactly once per variant."""
    out = dict(_RAIN_PHRASES)
    out[STAND], out[WALK], out[JUMP] = stand, walk, jump
    if board: out[BOARD] = board
    if ride: out[RIDE_VEHICLE] = ride
    if drive: out[DRIVE] = drive
    return out


def _opening(place_stand, place_walk, place_jump, board=None, ride=None, drive=None):
    out = dict(_RAIN_OPENING)
    out[STAND], out[WALK], out[JUMP] = place_stand, place_walk, place_jump
    if board: out[BOARD] = board
    if ride: out[RIDE_VEHICLE] = ride
    if drive: out[DRIVE] = drive
    return out


NO_ACTORS = ("0.00–5.17s: No other visible person, animal, vessel, or independently "
             "moving identity appears; only ambient environmental motion continues.")


# ----------------------------------------------------------------------- 1. boulevard
BOULEVARD = dict(
    place="the sunlit palace boulevard",
    identity=(
        "The protagonist is a pearl-ceramic grounds courier with a slender, athletic humanoid "
        "automaton build. His smooth white cranial shell, narrow black facial visor, segmented "
        "gold-metal neck, pearl shoulder and torso plates, black flexible seams, brushed-gold "
        "joint housings, and articulated four-fingered hands are stable identity features. A "
        "cropped ivory technical shell follows his chest and back above light pleated utility "
        "shorts, and slim pearl leg casings leave gold-and-black knee and ankle mechanisms "
        "visible above compact white feet. Hard midday sunlight gives the ceramic a warm ivory "
        "sheen and sharp gold glints at every joint. He is seen from behind and slightly above, "
        "left of centre, walking away up the boulevard."),
    scene=(
        "Charcoal asphalt with paired gold centre lines and white lane marks runs ahead between "
        "pale curbs toward warm ivory garden walls, clipped jade hedges, palms, dark-glass "
        "residences and patterned stone screens. A pearl-ceramic imperial utility roadster stands "
        "at the lower screen-left with its near door open, and an autonomous ceremonial carriage "
        "with a tall pearl body and a cyan light strip waits ahead-left. Golden-white midday light "
        "produces crisp shadows in clear air."),
    others_appearance=(
        "The glass-roof court shuttle is a compact pearl transit vehicle with a low curved "
        "black-glass cabin, gold roof rails, concealed wheel housings and a narrow cyan light "
        "strip; it moves on the boulevard itself. A copper-faced garden guard automaton and two "
        "ivory protocol androids stand on the right-hand sidewalk and do not move."),
    actor_names={"shuttle": "the glass-roof court shuttle"},
    camera_ground=(
        "The elevated trailing camera follows him from behind and slightly above, holding him "
        "left of centre at close range as the boulevard and its vehicles move past."),
    phrases=_phrases(
        stand=["He holds his mark on the asphalt, pearl feet planted beside the gold centre line",
               "He keeps his mark where he stopped, visor turned toward the halted shuttle",
               "He stays on his mark, joints still, the shuttle's open side beside him"],
        walk=["The courier walks away up the boulevard on foot, arms swinging in a measured "
              "mechanical rhythm",
              "He continues up the lane on foot, gold joints catching the sun with each stride",
              "He carries on along the boulevard on foot, keeping to the near side of the gold "
              "centre lines"],
        jump=["He leaps forward over the lane marking and lands a stride on", "He leaps and lands",
              "He leaps once more"],
        board=["He takes hold of the open side of the {obj} and climbs in, folding his pearl "
               "limbs into the cabin",
               "He climbs in through the open side of the {obj}, settling onto its seat",
               "He climbs in and the door of the {obj} closes on him"],
        ride=["He is seated inside the halted {obj}, framed by its black-glass cabin",
              "He sits seated inside the {obj}, feet clear of the asphalt",
              "He remains seated inside the stopped {obj}, its cyan strip glowing"],
        drive=["He rides inside the {obj} as it pulls away up the boulevard",
               "He rides inside the moving {obj}, the ivory walls sliding past its glass",
               "He rides inside the {obj} while it carries him on toward the residences"]),
    opening=_opening(
        "At this instant he stands still on {place}",
        "At this instant he is walking away along {place} on foot",
        "At this instant he is in the air above {place}, mid-leap",
        board="At this instant he is climbing in through the open side of the court shuttle on {place}",
        ride="At this instant he sits inside the halted court shuttle on {place}",
        drive="At this instant he is being carried along {place} inside the moving court shuttle"),
    actor_phrase={
        A_HALT: "It stands halted on the asphalt beside him, its door open and its cyan strip lit",
        A_NEAR: "It comes up the boulevard toward him, slowing as it closes on his position",
        A_AWAY: "It draws away up the boulevard, receding between the ivory walls",
        A_DESC: "It comes down along the line of the road, growing in the frame",
        A_FALL: "It falls from the sky and bursts on the asphalt",
        A_CARRY: "It carries him up the boulevard at a steady pace with him seated inside",
        A_ALONG: "It draws level with him on the asphalt, slowing as it comes"},
    no_actors=NO_ACTORS,
)


# ------------------------------------------------------------------------ 2. causeway
CAUSEWAY = dict(
    place="the pale temple-dust shoulder beside the basalt causeway",
    identity=(
        "The protagonist is Sasuke Uchiha, a pale-skinned young man with a lean, athletic build. "
        "His thick dark hair rises in irregular backward spikes; his narrow face, dark eyes, "
        "uncovered hands and compact shinobi proportions are stable identity features. He wears "
        "a gray high-collared short-sleeve ninja shirt with a plain back, dark charcoal trousers, "
        "black open-toed shinobi sandals, a thick purple rope belt, dark wrist guards and a "
        "straight sword in a dark sheath angled behind his left hip. Warm sunlight catches the "
        "shirt's broad painterly folds. He is seen from behind, centred, in a forward running "
        "stride on the broad pale shoulder, with the basalt roadway to his right."),
    scene=(
        "The world is a painterly pirate-jungle ruin realm of chunky expressive silhouettes, "
        "softened edges and lively brushwork. A very broad pale shoulder fills the left foreground "
        "beside a continuous pale guide stripe, while cracked charcoal basalt slabs form the open "
        "causeway to the right. Low turquoise-stone parapets follow both outer edges. Dense emerald "
        "plants, coral-tipped leaves, loose rocks and a stepped vine-covered sanctuary stand to the "
        "left beneath a broad tree canopy, and steep jungle mountains rise under a blue sky with "
        "soft tropical clouds."),
    others_appearance=(
        "The jade-shell reliquary wagon is a compact enchanted road conveyance with a squat "
        "turquoise-jade body, dark rootwood ribs, a short square canopy, antique-gold corner "
        "bindings, four thick basalt wheels and a blunt carved prow; it travels down the causeway "
        "toward the camera on his right. Farther up the road a bronze-lotus temple reliquary on a "
        "six-wheeled plinth moves slowly away."),
    actor_names={"wagon": "the jade-shell reliquary wagon", "lotus": "the bronze-lotus temple reliquary"},
    camera_ground=(
        "The trailing camera follows him from behind near waist height, keeping him centred; "
        "as the wagon passes it eases back and widens, holding the causeway, the shoulder and the "
        "mountains in view."),
    phrases=_phrases(
        stand=["He holds his mark on the pale dust, sandals planted, watching the wagon pass",
               "He keeps his mark on the shoulder, sword still at his hip",
               "He stays on his mark beside the guide stripe"],
        walk=["Sasuke runs up the pale shoulder on foot, arms bent and separated from his torso",
              "He drives on up the shoulder on foot, the guide stripe steady at his right",
              "He keeps running along the shoulder on foot as the causeway opens ahead"],
        jump=["He leaps and lands", "He leaps clear", "He leaps once more"]),
    opening=_opening(
        "At this instant he stands still on {place}",
        "At this instant he is running along {place} on foot",
        "At this instant he is in the air above {place}, mid-leap"),
    actor_phrase={
        A_HALT: "It stands halted on the basalt, wheels still",
        A_NEAR: "It comes up the causeway toward him on his right, its prow facing the camera",
        A_AWAY: "It draws away down the causeway behind him, receding toward the road bend",
        A_DESC: "It comes down along the line of the road, growing in the frame",
        A_FALL: "It falls from the sky and bursts on the basalt",
        A_CARRY: "It carries him along the causeway",
        A_ALONG: "It draws level with him on his right, wheels turning beneath its carved body"},
    no_actors=NO_ACTORS,
)


# -------------------------------------------------------------------------- 3. stream
STREAM = dict(
    place="the wet boundary between the shallow river and the coral-silt bank",
    identity=(
        "The protagonist is Sonic the Hedgehog, the blue speedster hedgehog, with a slender, "
        "athletic nonhuman build. His oversized head carries six broad swept-back cobalt quills; "
        "his compact blue torso, peach muzzle and arms, small rounded ears, white four-fingered "
        "gloves, long blue legs, white socks and vivid red shoes with broad white straps are "
        "stable identity features. No shirt or trousers cover his blue body. He occupies the lower "
        "centre, seen from behind in a running stride at the water's edge, his cobalt quills and "
        "red shoes the strongest accents against the muted prairie."),
    scene=(
        "A shallow pewter-blue river fills the foreground and extends along screen-left, while a "
        "pale coral-silt shoreline opens across screen-right and curves toward the centre distance. "
        "Rounded hills rise on both sides in folded bands of coral clay and lavender shale under "
        "wind-combed mint, sage and straw grasses, with squat blue-green thickets on the slopes. "
        "Monumental charcoal-cobalt storm clouds stack across a butter-yellow sky. The whole "
        "landscape is a handmade pastel risograph print with coarse halftone dots, cream paper "
        "grain, overlapping limited-colour inks and imperfect registration."),
    others_appearance="No other person, creature, craft or aircraft is present.",
    actor_names={},
    camera_ground=(
        "The trailing third-person camera follows low behind Sonic, holding him low and slightly "
        "left of centre; when he turns away from the river it swings round behind him so the "
        "bank and the hills wheel across the frame."),
    phrases=_phrases(
        stand=["He holds his mark at the water's edge, red shoes planted in the wet silt",
               "He keeps his mark on the bank, quills still",
               "He stays on his mark beside the river"],
        walk=["Sonic runs along the water's edge on foot, red shoes throwing crisp splashes",
              "He races on along the shoreline on foot, quills streaming behind his head",
              "He keeps running on foot, the ripples and the coral bank sliding past"],
        jump=["He leaps and lands", "He leaps clear", "He leaps once more"]),
    opening=_opening(
        "At this instant he stands still on {place}",
        "At this instant he is running along {place} on foot",
        "At this instant he is in the air above {place}, mid-leap"),
    no_actors=NO_ACTORS,
)


# ----------------------------------------------------------------------- 4. witchroad
WITCHROAD = dict(
    place="the broad pale witch-road of cracked moonstone slabs",
    identity=(
        "The protagonist is Tanjiro Kamado, a slim, athletic teenage swordsman with a compact "
        "human build. He begins large in the lower centre-left, seen from behind. His tousled "
        "burgundy-black hair forms thick angular locks, and a forehead scar, warm fair skin and "
        "long rectangular hanafuda earrings are stable identity features. He wears a "
        "green-and-black checkered haori with broad sleeves over a fitted dark uniform, loose "
        "dark trousers, white calf wraps and dark footwear, with a sheathed Nichirin sword at his "
        "left hip. The haori reads as matte woven cloth; the sword fittings carry small polished "
        "highlights."),
    scene=(
        "The road is cracked moonstone in ivory and cool gray, broken by shallow seams and sparse "
        "clusters of silver grass. A smooth charcoal route runs parallel along screen-left toward "
        "a burgundy-roofed gothic workshop arcade with two narrow alchemical cistern towers. A "
        "compact carved stone reliquary stands ahead near the upper centre. Farther right a "
        "skeletal wrought-iron bell spire rises above a long iron briar fence climbed by burgundy "
        "roses. Windswept ash-gray highlands and sparse cypress lift beneath a luminous overcast "
        "sky with a moonlit opening at upper right. Vivid polished anime cel-shading, crisp "
        "linework and gloomy but inviting cinematic light."),
    others_appearance="No vehicle, mount or other person is present.",
    actor_names={},
    camera_ground=(
        "The trailing camera follows close behind him near shoulder height, holding his large "
        "lower centre-left placement; while he stands at the reliquary it swings slowly round "
        "him in a partial orbit, then settles behind him again as he walks on."),
    phrases=_phrases(
        stand=["Tanjiro holds his mark before the stone reliquary, hands loose at his sides",
               "He keeps his mark on the moonstone, head turning slowly as the view wheels round him",
               "He stays on his mark a moment longer, the haori settling still"],
        walk=["Tanjiro walks away up the witch-road on foot, the checkered haori swinging with his stride",
              "He continues along the moonstone slabs on foot, earrings swaying beside his neck",
              "He walks on past the reliquary on foot toward the highlands"],
        jump=["He leaps and lands", "He leaps clear", "He leaps once more"]),
    opening=_opening(
        "At this instant he stands still on {place}",
        "At this instant he is walking away along {place} on foot",
        "At this instant he is in the air above {place}, mid-leap"),
    no_actors=NO_ACTORS,
)


# ---------------------------------------------------------------------- 5. frostcourt
FROSTCOURT = dict(
    place="the frost-dark basalt apron of the gate court",
    identity=(
        "The protagonist is Santa Claus, a sturdy adult man with a broad, powerfully built torso. "
        "His full white beard, long frost-touched white hair beneath a crimson fur-edged cap, "
        "crimson wool coat trimmed with dense white fur, matching trousers, a broad dark leather "
        "belt stamped with Norse patterns and heavy banded leather boots are stable identity "
        "features. Fine snow clings to the wool and the beard. No sack accompanies him. He is "
        "centred low in the view and seen from behind, torso leaning forward in a running stride "
        "across the frost-edged apron."),
    scene=(
        "A broad frost-dark basalt apron fills the foreground and middle distance with long fitted "
        "seams, a raised curb and thin ember-gold inlays. Frost-heather borders a massive "
        "ice-limestone wall ahead-left. Rune-carved timber gateway columns rise at the right and "
        "near the centre, each carrying a fixed amber lantern. Beyond them silver firs, snowy "
        "slopes and clustered timber-and-stone halls rise into a cold blue-black night lit by "
        "warm windows, lanterns and silver moonlight, with faint drifting snow."),
    others_appearance=(
        "A tiny fur-cloaked lantern steward stands far ahead-right on the outer walkway and does "
        "not move."),
    actor_names={},
    camera_ground=(
        "The trailing camera follows him at a low human height from behind, keeping him centred "
        "low in the frame across the curb toward the gate court; it rises briefly with each "
        "leap and settles again."),
    phrases=_phrases(
        stand=["Santa holds his mark on the basalt, boots planted on the frost",
               "He keeps his mark on the apron, beard stirring in the cold air",
               "He stays on his mark below the gate"],
        walk=["Santa runs across the basalt apron on foot, the crimson coat swinging heavily",
              "He drives on across the frost on foot, boots striking the fitted seams",
              "He keeps running on foot toward the rune-carved gate"],
        jump=["He leaps over the raised curb, both boots leaving the basalt together, and comes "
              "down running a stride further on",
              "He leaps clear of the frost-edged step and lands with his weight forward, the "
              "coat lifting and settling",
              "He leaps once more, a short bound over the ember-gold inlay, and carries on"]),
    opening=_opening(
        "At this instant he stands still on {place}",
        "At this instant he is running across {place} on foot",
        "At this instant he is in the air above {place}, mid-leap"),
    no_actors=NO_ACTORS,
)


# ----------------------------------------------------------------------- 6. monastery
MONASTERY = dict(
    place="the broad cracked charcoal-stone monastery road",
    identity=(
        "The protagonist is Kobe Bryant, an adult human man with a tall athletic build translated "
        "into compact, rounded toy proportions. His deep brown skin, shaved head, smooth simplified "
        "face, broad shoulders, rounded hands and long powerful legs are stable identity features. "
        "He wears a blank purple sleeveless basketball jersey with narrow gold piping, purple "
        "shorts with gold stripes, a purple sleeve on his right arm, white calf socks and "
        "purple-and-gold basketball shoes with thick white soles. He begins in the lower centre, "
        "right of centre, facing away from the camera in a running stride."),
    scene=(
        "A broad cracked charcoal-stone road fills the foreground, crossed by a clean white "
        "boundary line. A tall pale stone pole rises just left of his path, and a carved stone "
        "balustrade separates the road from white-plaster monastery halls with vermilion beams, "
        "jade roof tiles and golden ornaments on the right. Two stacked cedar prayer-wheel drums "
        "rest on squared supports ahead-right beside the balustrade. A trimmed alpine tree rises "
        "beyond the drums, prayer pennants span the distant courtyard, and snowcapped peaks stand "
        "beneath a pale blue sky. Every surface has softly bevelled, polished toy construction, "
        "matte painted wood, smooth stone and warm sunlight."),
    others_appearance=(
        "Four monastery novices, small chibi adults with shaved rounded heads in plum, jade, "
        "saffron-and-vermilion and tan robes, stand grouped beside the prayer drums on the right "
        "and do not move."),
    actor_names={},
    camera_ground=(
        "The trailing camera follows low behind him, holding him right of centre; as he stops at "
        "the drums it pushes in closer over his shoulder, then drops back as he runs on."),
    phrases=_phrases(
        stand=["Kobe holds his mark on the charcoal stone before the prayer drums, shoulders squared",
               "He keeps his mark beside the balustrade, looking over the carved drums",
               "He stays on his mark a moment longer, weight settled on both shoes"],
        walk=["Kobe runs up the monastery road on foot, torso pitched forward, long legs driving",
              "He continues on foot past the pale stone pole toward the drums",
              "He runs on along the road on foot, the balustrade sliding past on his right"],
        jump=["He leaps and lands", "He leaps clear", "He leaps once more"]),
    opening=_opening(
        "At this instant he stands still on {place}",
        "At this instant he is running along {place} on foot",
        "At this instant he is in the air above {place}, mid-leap"),
    no_actors=NO_ACTORS,
)


CONTENT = {"boulevard": BOULEVARD, "causeway": CAUSEWAY, "stream": STREAM,
           "witchroad": WITCHROAD, "frostcourt": FROSTCOURT, "monastery": MONASTERY}

"""Shared demo world content, so the caption steps and the end-to-end build agree.

Structure comes from compiler/caption.py; this is the prose a coding agent would write.
Section lengths follow the released proportions measured in step `caption_schema`.
"""
from __future__ import annotations

from .caption import Caption

COURIER = dict(
    protagonist=(
        "The protagonist is a lean adult courier with an athletic, long-limbed build. She has "
        "warm brown skin, close-cropped dark hair, a narrow face with high cheekbones, and a "
        "small pale scar across her right eyebrow. Her cropped hair, scar, and the wide strap "
        "crossing her chest are stable identity features. She wears a slate-grey canvas work "
        "jacket with the sleeves pushed to the elbow over a pale linen shirt, tapered charcoal "
        "trousers gathered at the ankle, and scuffed brown leather boots with worn heels. A "
        "flat canvas satchel rides against her left hip on a wide webbing strap, its buckle "
        "hanging loose. Fingerless leather gloves cover her palms. The jacket hem, strap end, "
        "and shirt cuffs move freely with her stride. She begins at lower centre-left in "
        "three-quarter view, facing away from the camera and angled slightly toward "
        "screen-right, her weight already committed to her leading foot."),
    held_objects=(
        "The courier carries nothing in her hands. The canvas satchel is the only object that "
        "moves directly with her: it hangs from a wide strap across her chest, rests against "
        "her left hip, and swings backward each time her left foot strikes the paving. Its "
        "flap is buckled shut, its corners are softened with wear, and its surface is a faded "
        "olive that has bleached unevenly along the top seam. The loose buckle end taps "
        "against the bag body throughout her run without ever separating from the strap."),
    other_actors_appearance=(
        "Two market vendors stand further along the street, both adults in loose undyed aprons "
        "over dark work clothes; one is broad-shouldered and bare-headed, the other slighter "
        "and wearing a faded red headscarf. A low wooden handcart with iron-rimmed wheels sits "
        "between them, stacked with crates. A single boxy delivery vehicle in dull cream paint "
        "idles at the kerb beyond the cart, its rear panel open."),
    scene=(
        "The setting is a sunlit stone market street at mid-morning. Warm ochre facades rise "
        "on both sides, three storeys high, their wooden shutters half-closed against the "
        "light. Striped canvas awnings project from the upper floors and throw hard-edged "
        "shadows in bands across worn granite paving. Fine dust hangs in the shafts of light "
        "between them. Shallow steps, iron drain grates, and chalked price boards break the "
        "ground plane at intervals. The palette runs to ochre, chalk white, faded terracotta, "
        "olive, and a deep shadow blue in the recesses. Sunlight is strong and slightly warm, "
        "producing crisp shadow edges and bright bounce from the pale stone. The camera "
        "follows from behind and slightly above her shoulder, holding her left of centre "
        "while the street opens ahead, and adjusts smoothly without overtaking her."),
    actions=[
        "The courier pushes off hard from her leading foot and accelerates along the granite "
        "paving, her torso pitching forward as her trailing leg extends fully behind her; the "
        "satchel swings backward against her left hip, the loose buckle end lifting away from "
        "the bag body, and her jacket hem flares once before settling",
        "She lengthens her stride and both arms begin driving in opposition, elbows bent close "
        "to her ribs; she passes out of the open light and into the hard-edged shadow of an "
        "awning, her shoulders squaring as the shadow band crosses her back and the bright "
        "paving ahead fills the lower frame",
        "She angles slightly toward screen-right to clear the low handcart, her outside "
        "shoulder dipping into the turn and her inside foot crossing over; the satchel swings "
        "outward away from her hip on the turn, and her head stays level and fixed on the "
        "line ahead rather than on the cart",
        "Her trailing arm sweeps wide for balance as she recovers from the turn, her hips "
        "squaring back to the direction of travel; her stride shortens for one step, then "
        "opens again, and the satchel settles back against her left hip with the strap "
        "briefly taut across her chest",
        "She settles into an even run with her head level and her breathing regular, arms "
        "swinging in a smaller, more economical arc; the street opens out ahead of her as the "
        "facades step back, and her full body remains visible against the bright paving with "
        "the awning shadows sliding across her in bands",
        "Her pace eases slightly and she turns her head briefly toward the vendors without "
        "breaking stride, her torso continuing to face down the street; her near arm drops a "
        "little lower through its swing, and the satchel loses some of its backward motion "
        "before her head returns forward",
        "She resumes full speed, her stride opening again as she drives away from the camera "
        "down the street; the satchel swings back to its fullest extent, her jacket hem lifts, "
        "and she ends still running, smaller in frame, with the open street and the parked "
        "delivery vehicle beyond her",
    ],
    other_actors=(
        "0.00\u20132.21s, the two vendors: the broad-shouldered vendor lifts a crate from the "
        "handcart while the headscarfed vendor steadies the stack with both hands; neither "
        "leaves the cart. 2.21\u20135.17s, the delivery vehicle: it remains parked at the kerb "
        "with its rear panel open and no motion of its own throughout."),
)


def courier_caption(continuation: bool = False) -> Caption:
    """One window of the demo take. Continuation windows must not restart the scene."""
    data = dict(COURIER)
    if continuation:
        p = data["protagonist"]
        data["protagonist"] = "Continuing without a cut, " + p[0].lower() + p[1:]
    return Caption(**data)

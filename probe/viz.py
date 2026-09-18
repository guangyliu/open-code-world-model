"""Step 1 deliverable: render proxy frames the way the model sees them."""
from __future__ import annotations

import numpy as np
from PIL import Image

from .proxy import CLASSES, VALID_EPS_M, depth_to_log_codes, to_duv

# Distinct hues so class boundaries are readable at 336x192.
CLASS_RGB = np.array([
    (40, 40, 40),      # 0 void_unknown
    (120, 180, 255),   # 1 sky
    (0, 90, 200),      # 2 water
    (150, 110, 60),    # 3 terrain
    (170, 170, 170),   # 4 road_paved
    (40, 160, 60),     # 5 vegetation
    (230, 120, 40),    # 6 building_structure
    (200, 60, 200),    # 7 infrastructure
    (255, 40, 40),     # 8 human
    (255, 230, 60),    # 9 animal
    (60, 230, 230),    # 10 vehicle
    (255, 150, 180),   # 11 prop
], dtype=np.uint8)


def depth_image(metric: np.ndarray) -> Image.Image:
    """Fixed-log depth exactly as encoded for the model: near = bright, sky = black."""
    return Image.fromarray((depth_to_log_codes(metric) >> 8).astype(np.uint8), "L").convert("RGB")


def semantic_image(semantic: np.ndarray) -> Image.Image:
    return Image.fromarray(CLASS_RGB[semantic], "RGB")


def duv_image(metric: np.ndarray, semantic: np.ndarray) -> Image.Image:
    return Image.fromarray(to_duv(metric, semantic), "RGB")


def contact_sheet(panels: list[tuple[str, Image.Image]], *, scale: int = 2, cols: int = 3) -> Image.Image:
    """Tile labelled panels into one sheet."""
    from PIL import ImageDraw

    pw, ph = panels[0][1].size
    pw, ph = pw * scale, ph * scale
    pad, bar = 6, 14
    rows = -(-len(panels) // cols)
    sheet = Image.new("RGB", (cols * (pw + pad) + pad, rows * (ph + bar + pad) + pad), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)
    for k, (label, img) in enumerate(panels):
        r, c = divmod(k, cols)
        x, y = pad + c * (pw + pad), pad + r * (ph + bar + pad)
        sheet.paste(img.resize((pw, ph), Image.NEAREST), (x, y + bar))
        draw.text((x + 2, y + 2), label, fill=(220, 220, 220))
    return sheet

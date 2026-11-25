"""Pull the new hair's color toward the person's existing hair.

Edit models sometimes return hair a shade lighter, warmer or greyer than the real hair, which
is the giveaway in a before/after. We shift the edited hair pixels' mean color in Lab space
toward the original hair's mean, only inside the edit mask.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .segment import HAIR

MIN_HAIR_PIXELS = 400
STRENGTH = 0.8  # how far to move toward the original mean (1 = all the way)
MAX_SHIFT = 40.0  # Lab units; caps the correction if segmentation picked the wrong pixels


def match_hair_tone(original: Image.Image, edited: Image.Image, orig_cats: np.ndarray | None,
                    edited_cats: np.ndarray | None, mask: Image.Image) -> tuple[Image.Image, float]:
    """Return (adjusted edited image, size of the Lab shift applied)."""
    if orig_cats is None or edited_cats is None:
        return edited, 0.0
    m = np.asarray(mask, dtype=np.float32) / 255.0
    orig_hair = orig_cats == HAIR
    new_hair = (edited_cats == HAIR) & (m > 0.05)
    if orig_hair.sum() < MIN_HAIR_PIXELS or new_hair.sum() < MIN_HAIR_PIXELS:
        return edited, 0.0

    orig_lab = cv2.cvtColor(np.asarray(original.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)
    edit_lab = cv2.cvtColor(np.asarray(edited.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)
    shift = orig_lab[orig_hair].mean(axis=0) - edit_lab[new_hair].mean(axis=0)
    size = float(np.linalg.norm(shift))
    if size > MAX_SHIFT:
        shift *= MAX_SHIFT / size
    weight = cv2.GaussianBlur(new_hair.astype(np.float32) * m * STRENGTH, (0, 0), 2)[..., None]
    out = np.clip(edit_lab + weight * shift, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_LAB2RGB)), min(size, MAX_SHIFT)

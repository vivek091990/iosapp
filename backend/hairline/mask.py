"""Build the soft mask of where new hair is allowed to appear."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .face import Face
from .segment import FACE_SKIN, HAIR


def _head_region(size: tuple[int, int], face: Face, preset: str) -> tuple[np.ndarray, int]:
    """Dome over the head whose flat bottom sits a little above the eyebrows."""
    w, h = size
    lo_x, hi_x = face.oval_x
    fw = face.width
    brow = face.brow_top_y
    gap = max(brow - face.forehead_y, 1.0)
    reach = {"hairline": 0.30, "part": 0.45, "crown": 0.45}.get(preset, 0.35)
    bottom = brow - reach * gap
    side = 0.22 * fw
    left, right = lo_x - side, hi_x + side
    region_top = face.forehead_y - 1.0 * face.height

    dome = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(dome)
    draw.ellipse([left, region_top, right, bottom + (bottom - region_top)], fill=255)
    draw.rectangle([0, bottom, w, h], fill=0)
    return np.asarray(dome) > 0, int(bottom)


def hair_region_mask(
    size: tuple[int, int], face: Face, preset: str, categories: np.ndarray | None = None
) -> Image.Image:
    """White = may change, black = must stay original. Feathered edges.

    Inside the head dome, only current hair, forehead/scalp skin and a thin margin around
    them may change, so the background and everything from the brows down stay original.
    """
    w, h = size
    fw = face.width
    allowed, bottom = _head_region(size, face, preset)

    if categories is not None:
        head = np.isin(categories, (HAIR, FACE_SKIN)).astype(np.uint8)
        grow = max(3, int(0.06 * fw))  # room for a slightly fuller silhouette
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        head = cv2.dilate(head, kernel) > 0
        allowed &= head

    mask = Image.fromarray((allowed * 255).astype(np.uint8), mode="L")
    feather = max(2, int(0.03 * fw))
    mask = mask.filter(ImageFilter.GaussianBlur(feather))
    # Re-apply the hard protection after blurring so no blur bleeds onto the brows.
    arr = np.asarray(mask).copy()
    arr[int(min(h, max(0, bottom + feather))):, :] = 0
    return Image.fromarray(arr, mode="L")


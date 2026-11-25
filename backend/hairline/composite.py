"""Align the model's output to the original photo and paste back only the masked region."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .face import STABLE, Face


@dataclass
class Alignment:
    image: Image.Image
    residual: float  # mean stable-landmark error after alignment, as a fraction of face width


def align_to_original(edited: Image.Image, original_face: Face, edited_face: Face, size: tuple[int, int]) -> Alignment:
    """Warp `edited` so its face lines up with the original (similarity transform)."""
    src = edited_face.points.astype(np.float32)
    dst = original_face.points.astype(np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if matrix is None:
        return Alignment(edited.resize(size, Image.LANCZOS), float("inf"))
    warped = cv2.warpAffine(
        np.asarray(edited.convert("RGB")), matrix, size, flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT
    )
    moved = cv2.transform(src[None, STABLE, :], matrix)[0]
    err = np.linalg.norm(moved - dst[STABLE], axis=1).mean() / max(original_face.width, 1.0)
    return Alignment(Image.fromarray(warped), float(err))


def composite(original: Image.Image, edited: Image.Image, mask: Image.Image) -> Image.Image:
    if edited.size != original.size:
        edited = edited.resize(original.size, Image.LANCZOS)
    return Image.composite(edited.convert("RGB"), original.convert("RGB"), mask)

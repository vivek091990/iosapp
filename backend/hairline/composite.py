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


def lighting_correction(original: Image.Image, edited: Image.Image, mask: np.ndarray, sigma: float) -> Image.Image:
    """Pull the edited image's low-frequency colour and light back to the original's.

    Measures blur(original) - blur(edited) in Lab on a ring just outside `mask`, fills that
    correction smoothly across the masked area (a harmonic fill from the ring), and adds it to
    the edited pixels. Strand texture stays; only the lighting/colour shift goes.
    """
    m = mask > 0.5
    if not m.any():
        return edited
    orig = cv2.cvtColor(np.asarray(original.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)
    edit = cv2.cvtColor(np.asarray(edited.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)
    s = max(1.0, sigma)
    diff = cv2.GaussianBlur(orig, (0, 0), s) - cv2.GaussianBlur(edit, (0, 0), s)
    r = max(3, int(round(s)))
    ring = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    ring &= ~m
    # Solve on a small grid: the field is smooth by construction.
    h, w = m.shape
    scale = min(1.0, 160.0 / max(h, w))
    sw, sh = max(8, int(w * scale)), max(8, int(h * scale))
    small_ring = cv2.resize(ring.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
    small_mask = cv2.resize(m.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
    small_diff = cv2.resize(diff, (sw, sh), interpolation=cv2.INTER_AREA)
    field = np.where(small_ring[..., None], small_diff, 0.0).astype(np.float32)
    if small_ring.any():
        field[small_mask] = small_diff[small_ring].mean(axis=0)
    for _ in range(300):
        field = np.where(small_mask[..., None], cv2.blur(field, (3, 3)), field)
        field = np.where(small_ring[..., None], small_diff, field)
    full = cv2.resize(field, (w, h), interpolation=cv2.INTER_LINEAR)
    out = np.where(m[..., None] | ring[..., None], edit + full, edit)
    out = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(out)


def pyramid_blend(original: Image.Image, edited: Image.Image, alpha: np.ndarray, levels: int = 5) -> Image.Image:
    """Laplacian-pyramid blend: low frequencies mix over a wide band, fine strands over a narrow
    one, so the seam disappears without ghosting. Pixels where alpha == 0 stay bit-identical."""
    a = np.clip(alpha.astype(np.float32), 0, 1)
    o = np.asarray(original.convert("RGB"), dtype=np.float32)
    e = np.asarray(edited.convert("RGB").resize(original.size, Image.LANCZOS), dtype=np.float32)
    go, ge, ga = [o], [e], [a]
    for _ in range(levels):
        go.append(cv2.pyrDown(go[-1]))
        ge.append(cv2.pyrDown(ge[-1]))
        ga.append(cv2.pyrDown(ga[-1]))
    blended = ge[-1] * ga[-1][..., None] + go[-1] * (1 - ga[-1][..., None])
    for i in range(levels - 1, -1, -1):
        size = (go[i].shape[1], go[i].shape[0])
        lo_ = go[i] - cv2.pyrUp(go[i + 1], dstsize=size)
        le = ge[i] - cv2.pyrUp(ge[i + 1], dstsize=size)
        blended = cv2.pyrUp(blended, dstsize=size) + le * ga[i][..., None] + lo_ * (1 - ga[i][..., None])
    out = np.clip(blended, 0, 255)
    out = np.where(a[..., None] > 0, out, o)  # guarantee: untouched outside the mask
    return Image.fromarray(out.astype(np.uint8))

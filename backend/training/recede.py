"""Synthesise a receded hairline on a photo of someone with a full hairline.

Method: find the current hairline column by column, then move it up in an M shape (deeper
at the temples, shallow in the middle). The removed band is filled with forehead skin
reflected upward from just below the hairline, so skin texture and lighting continue. The
new edge is thinned gradually with strand-like noise rather than cut, the way real hair
thins, and the recession never takes more than part of the hair's thickness, so hair
always remains behind the new hairline. Purely classical image processing: no generative
model is involved.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from hairline.face import Face
from hairline.segment import HAIR

MAX_THICKNESS_SHARE = 0.45  # never remove more than this share of the hair above a column


@dataclass
class Recession:
    image: Image.Image  # the synthetic "before"
    mask: Image.Image  # where hair was removed or thinned (white), soft
    severity: float  # depth at the temples as a fraction of face height
    columns: int  # how many columns had a usable hairline


class Unsuitable(ValueError):
    """The photo can't make a good training pair; the message says why."""


def hairline_profile(cats: np.ndarray, face: Face) -> tuple[np.ndarray, np.ndarray]:
    """Per column inside the face width: y of the hairline (lowest hair pixel above the
    brows) and the hair's thickness above it. NaN where there is no hair."""
    h, w = cats.shape
    lo, hi = face.oval_x
    gap = max(face.brow_top_y - face.forehead_y, 1.0)
    limit = int(face.brow_top_y - 0.25 * gap)  # hair below this is brows or sideburns
    top = int(max(0, face.forehead_y - 1.2 * face.height))
    line = np.full(w, np.nan, dtype=np.float32)
    thick = np.full(w, np.nan, dtype=np.float32)
    hair = cats == HAIR
    for x in range(int(max(0, lo)), int(min(w, hi))):
        ys = np.flatnonzero(hair[top:limit, x])
        if ys.size:
            line[x] = top + ys.max()
            thick[x] = ys.max() - ys.min()
    return line, thick


def _depth_curve(w: int, face: Face, severity: float, rng: np.random.Generator) -> np.ndarray:
    lo, hi = face.oval_x
    cx, half = (lo + hi) / 2, max((hi - lo) / 2, 1.0)
    t = (np.arange(w) - cx) / half
    temple_at = rng.uniform(0.55, 0.75)
    centre = rng.uniform(0.2, 0.45)  # how much the middle recedes relative to the temples
    shape = centre + (1 - centre) * np.exp(-(((np.abs(t) - temple_at) / 0.25) ** 2))
    # Irregular edge: low-frequency wobble so the new hairline isn't a clean curve.
    wobble = cv2.GaussianBlur(rng.normal(0, 1, (1, w)).astype(np.float32), (0, 0), sigmaX=max(3.0, 0.03 * (hi - lo)))[0]
    wobble /= max(np.abs(wobble).max(), 1e-6)
    return (severity * face.height * shape * (1 + 0.15 * wobble)).astype(np.float32)


def recede(img: Image.Image, face: Face, cats: np.ndarray, severity: float, seed: int = 0) -> Recession:
    """Return a receded copy of `img`; raises Unsuitable when the photo can't be used."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    h, w = cats.shape
    line, thick = hairline_profile(cats, face)
    gap = max(face.brow_top_y - face.forehead_y, 1.0)
    has_hair = ~np.isnan(line)
    fringe = np.zeros_like(has_hair)
    fringe[has_hair] = line[has_hair] > face.forehead_y + 0.45 * gap  # hair hanging over the forehead
    if has_hair.any() and fringe[has_hair].mean() > 0.3:
        raise Unsuitable("hair (bangs or a fringe) covers the forehead")
    line[fringe] = np.nan
    thick[fringe] = np.nan
    valid = ~np.isnan(line)
    if valid.sum() < 0.6 * face.width:
        raise Unsuitable("hairline not visible across the forehead")

    xs = np.arange(w)
    line_s = np.interp(xs, xs[valid], line[valid]).astype(np.float32)
    line_s = cv2.GaussianBlur(line_s[None, :], (0, 0), sigmaX=max(2.0, 0.02 * face.width))[0]
    thick_s = np.interp(xs, xs[valid], thick[valid]).astype(np.float32)
    depth = np.minimum(_depth_curve(w, face, severity, rng), MAX_THICKNESS_SHARE * thick_s)

    hair = cats == HAIR
    alpha = np.zeros((h, w), dtype=np.float32)
    fill = arr.copy()
    # Strand-like noise: stretched vertically so thinning reads as separate hairs.
    strands = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), sigmaX=0.6, sigmaY=4.0)
    strands = (strands - strands.min()) / max(np.ptp(strands), 1e-6)
    # The old hairline's soft edge (shadow, fine hairs) is mostly not segmented as hair;
    # cover a margin below it too, or a ghost of the old hairline stays visible.
    margin = max(3.0, 0.03 * face.height)
    for x in np.flatnonzero(valid):
        d = depth[x]
        if d < 2:
            continue
        y_old, y_new = line_s[x], line_s[x] - d
        base = y_old + margin
        y0, y1 = int(max(0, y_new - 0.35 * d)), int(min(h - 1, base))
        rows = np.arange(y0, y1 + 1)
        rows = rows[hair[rows, x] | (rows > y_old - 2)]
        if rows.size == 0:
            continue
        # Removed below the new line, thinning over a zone above it, fading out into real
        # skin over the margin at the bottom.
        a = np.clip((rows - (y_new - 0.35 * d)) / max(0.35 * d, 1.0), 0, 1)
        a = np.where(a < 1, a * (strands[rows, x] > 0.45), a)  # keep some strands in the zone
        a = np.minimum(a, np.clip((base - rows) / (0.5 * margin), 0, 1))
        alpha[rows, x] = a
        src = np.clip((2 * base - rows).astype(int), 0, h - 1)  # reflect skin from below the margin
        fill[rows, x] = arr[src, x]

    if (alpha > 0.5).sum() < 50:
        raise Unsuitable("too little hair to remove")
    # Close gaps where strands were segmented as non-hair, then soften slightly.
    k = max(3, int(0.01 * face.width)) | 1
    closed = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((k, 1), np.uint8))
    gaps = (closed > alpha + 0.3)
    fill = np.where(gaps[..., None], cv2.GaussianBlur(fill, (0, 0), max(2.0, 0.01 * face.width)), fill)
    alpha = cv2.GaussianBlur(closed, (0, 0), 1.0)[..., None]
    blended = arr * (1 - alpha) + fill * alpha
    grain = rng.normal(0, 1.8, blended.shape).astype(np.float32) * alpha
    result = np.clip(blended + grain, 0, 255).astype(np.uint8)
    return Recession(
        image=Image.fromarray(result),
        mask=Image.fromarray((alpha[..., 0] * 255).astype(np.uint8), mode="L"),
        severity=severity,
        columns=int(valid.sum()),
    )

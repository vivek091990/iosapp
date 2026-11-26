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



# Landmarks used to place a natural target hairline (MediaPipe 478-point mesh).
GLABELLA = 168
SUBNASALE = 2
IRIS_A, IRIS_B = 468, 473          # iris centres
CANTHUS_A, CANTHUS_B = 33, 263     # outer eye corners

# Hairline-design rules from hair-restoration surgery, scaled by inter-pupil distance (IPD,
# about 6.3 cm in adult men): mid-frontal point 1.05-1.30 IPD above the glabella (the surgical
# 8 +- 1 cm measured over the curved forehead); temple corners (frontotemporal angles) on the
# vertical through the outer eye corners and higher than the middle; a flat arc between the
# pupils; an irregular edge. See the research doc. The coverage slider moves the line: Light
# keeps it conservative (surgical, age-appropriate), Strong goes to the rule-of-thirds line.
MFP_MIN_IPD, MFP_MAX_IPD = 0.95, 1.30
COVERAGE_RAISE_IPD = (0.20, 0.0)   # extra height at coverage 0 and at coverage 100
TEMPLE_RISE_IPD = 0.12
BEYOND_CORNER_SLOPE = 0.35         # past the corners the line runs back toward the ear
TRANSITION_IPD = 0.12              # depth over which density fades in behind the hairline


def _ipd(face: Face) -> float:
    p = face.points
    if len(p) > max(IRIS_A, IRIS_B):
        d = float(np.linalg.norm(p[IRIS_A] - p[IRIS_B]))
        if d > 1:
            return d
    return 0.45 * face.width  # face mesh without iris points


def target_hairline_y(face: Face, width: int, coverage: int = 50, irregular: bool = True, seed: int = 7) -> np.ndarray:
    """Per-column y (pixels) of a natural target hairline, from surgical hairline-design rules."""
    p = face.points
    ipd = _ipd(face)
    glabella_y = float(p[GLABELLA, 1])
    thirds = float(p[SUBNASALE, 1]) - glabella_y  # rule of thirds: hairline-glabella ~ glabella-nose base
    c = float(np.clip(coverage, 0, 100)) / 100.0
    raise_ipd = COVERAGE_RAISE_IPD[0] + (COVERAGE_RAISE_IPD[1] - COVERAGE_RAISE_IPD[0]) * c
    rise = float(np.clip(thirds, MFP_MIN_IPD * ipd, MFP_MAX_IPD * ipd)) + raise_ipd * ipd
    mfp_y = glabella_y - rise

    pupils = sorted((float(p[IRIS_A, 0]), float(p[IRIS_B, 0]))) if len(p) > IRIS_B else None
    corners = sorted((float(p[CANTHUS_A, 0]), float(p[CANTHUS_B, 0])))
    cx = (corners[0] + corners[1]) / 2
    half_pupil = (pupils[1] - pupils[0]) / 2 if pupils else 0.5 * ipd
    half_corner = max((corners[1] - corners[0]) / 2, half_pupil + 1.0)

    x = np.arange(width, dtype=np.float64)
    dx = np.abs(x - cx)
    y = np.full(width, mfp_y)
    inner = dx <= half_pupil
    y[inner] -= 0.03 * ipd * (dx[inner] / half_pupil) ** 2          # gentle oval between the pupils
    mid = (dx > half_pupil) & (dx <= half_corner)
    t = (dx[mid] - half_pupil) / (half_corner - half_pupil)
    y[mid] -= 0.03 * ipd + (TEMPLE_RISE_IPD - 0.03) * ipd * (3 * t**2 - 2 * t**3)  # smooth rise to the corners
    outer = dx > half_corner
    y[outer] -= TEMPLE_RISE_IPD * ipd + BEYOND_CORNER_SLOPE * (dx[outer] - half_corner)

    if irregular:
        rng = np.random.default_rng(seed)
        macro = sum(np.sin(2 * np.pi * x / (0.5 * ipd * k) + rng.uniform(0, 2 * np.pi)) for k in (1.0, 1.6))
        step = max(2.0, 0.025 * ipd)
        knots = np.arange(0, width + step, step)
        micro = np.interp(x, knots, rng.uniform(-1, 1, len(knots)))
        y += 0.012 * ipd * macro + 0.006 * ipd * micro
    return y


def hairline_density(face: Face, shape: tuple[int, int], line: np.ndarray | None = None) -> np.ndarray:
    """Target hair density (0..1) per pixel: 0 below the target line, fading in over the
    transition zone behind it (single fine hairs at the front), 1 further back."""
    h, w = shape
    line = target_hairline_y(face, w) if line is None else line
    depth = (line[None, :] - np.arange(h)[:, None]) / max(TRANSITION_IPD * _ipd(face), 1.0)
    t = np.clip(depth, 0.0, 1.0)
    dens = 0.15 + 0.85 * (t * t * (3 - 2 * t))
    return np.where(depth > 0, dens, 0.0).astype(np.float32)


def fill_zone_mask(
    face: Face, categories: np.ndarray, hair_prob: np.ndarray | None = None, coverage: int = 50
) -> np.ndarray:
    """Boolean mask of bare forehead/temple skin above the target hairline: where new hair goes.

    Existing hair is not part of it. With `hair_prob` (the 512 px hair segmenter), see-through
    thinning hair that the coarse segmenter calls "hair" counts as bare when its probability is low.
    """
    h, w = categories.shape
    lo, hi = face.oval_x
    fw = hi - lo
    line = target_hairline_y(face, w, coverage)
    if hair_prob is None:
        bare = categories == FACE_SKIN
    else:
        bare = np.isin(categories, (FACE_SKIN, HAIR)) & (hair_prob < 0.5)
    cols = np.arange(w)[None, :]
    bare &= (cols > lo - 0.05 * fw) & (cols < hi + 0.05 * fw)
    # Only bare skin connected to the forehead counts; gaps inside the hair mass or at the top
    # of the head are not a hairline.
    n, labels = cv2.connectedComponents(bare.astype(np.uint8))
    p = face.points
    seed_y = int(np.clip(face.brow_top_y - 0.15 * _ipd(face), 0, h - 1))
    seeds = {int(labels[seed_y, int(np.clip(x, 0, w - 1))]) for x in np.linspace(p[CANTHUS_A, 0], p[CANTHUS_B, 0], 9)}
    seeds.discard(0)
    forehead = np.isin(labels, list(seeds)) if seeds else np.zeros_like(bare)
    zone = forehead & (np.arange(h)[:, None] < line[None, :])
    n, labels, stats, _ = cv2.connectedComponentsWithStats(zone.astype(np.uint8))
    keep = np.zeros_like(zone)
    for i in range(1, n):  # drop specks from segmentation noise
        if stats[i, cv2.CC_STAT_AREA] > 0.002 * fw * fw:
            keep |= labels == i
    return keep


def inpaint_mask(face: Face, zone: np.ndarray, categories: np.ndarray, overlap_ipd: float = 0.12) -> np.ndarray:
    """Soft (0..1) generation mask for masked inpainting: the fill zone plus a band of the
    existing hair just behind it, so new and old hair are generated together and the seam
    lands inside hair texture rather than on the skin/hair edge. Never reaches the brows."""
    h, w = zone.shape
    ipd = _ipd(face)
    r = max(2, int(round(overlap_ipd * ipd)))
    grown = cv2.dilate(zone.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    grown &= np.isin(categories, (HAIR, FACE_SKIN)) | zone
    grown[int(max(0, face.brow_top_y - 0.1 * ipd)) :, :] = False
    soft = cv2.GaussianBlur(grown.astype(np.float32), (0, 0), max(1.0, 0.03 * ipd))
    soft[zone] = 1.0
    soft[int(max(0, face.brow_top_y - 0.05 * ipd)) :, :] = 0.0
    return np.clip(soft, 0, 1)

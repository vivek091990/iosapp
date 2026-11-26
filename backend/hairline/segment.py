"""Hair / skin segmentation via MediaPipe's selfie multiclass segmenter (Apache 2.0).

Classes: 0 background, 1 hair, 2 body skin, 3 face skin, 4 clothes, 5 others.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from .face import download_model

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)
MODEL_PATH = Path(os.environ.get("SEGMENTER_PATH", Path(__file__).resolve().parent.parent / "models" / "selfie_multiclass_256x256.tflite"))

HAIR, FACE_SKIN = 1, 3

_lock = threading.Lock()
_segmenter = None


def _get_segmenter():
    global _segmenter
    if _segmenter is None:
        from mediapipe.tasks.python import BaseOptions, vision

        opts = vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(download_model(MODEL_URL, MODEL_PATH))),
            output_category_mask=True,
        )
        _segmenter = vision.ImageSegmenter.create_from_options(opts)
    return _segmenter


def category_mask(img: Image.Image) -> np.ndarray:
    """Per-pixel class ids, same size as `img`."""
    import mediapipe as mp

    rgb = np.ascontiguousarray(np.asarray(img.convert("RGB")))
    with _lock:
        result = _get_segmenter().segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        # numpy_view() points into MediaPipe-owned memory; copy before the result is freed,
        # or concurrent requests read freed memory and crash the process.
        cats = np.array(result.category_mask.numpy_view(), copy=True).squeeze()
        del result
    if cats.shape != (img.height, img.width):
        cats = np.asarray(Image.fromarray(cats.astype(np.uint8)).resize(img.size, Image.NEAREST))
    return cats


HAIR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "hair_segmenter/float32/latest/hair_segmenter.tflite"
)
HAIR_MODEL_PATH = Path(os.environ.get("HAIR_SEGMENTER_PATH", Path(__file__).resolve().parent.parent / "models" / "hair_segmenter.tflite"))
_hair_segmenter = None


def _get_hair_segmenter():
    global _hair_segmenter
    if _hair_segmenter is None:
        from mediapipe.tasks.python import BaseOptions, vision

        opts = vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(download_model(HAIR_MODEL_URL, HAIR_MODEL_PATH))),
            output_confidence_masks=True,
            output_category_mask=False,
        )
        _hair_segmenter = vision.ImageSegmenter.create_from_options(opts)
    return _hair_segmenter


def hair_probability(img: Image.Image, box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Soft hair probability (0..1), same size as `img`, from MediaPipe's 512x512 hair segmenter.

    With `box` (left, top, right, bottom), only that crop is segmented and the rest is 0, so a
    head crop gets the model's full resolution instead of a fraction of it.
    """
    import mediapipe as mp

    w, h = img.size
    left, top, right, bottom = box or (0, 0, w, h)
    left, top = max(0, int(left)), max(0, int(top))
    right, bottom = min(w, int(right)), min(h, int(bottom))
    crop = img.convert("RGB").crop((left, top, right, bottom))
    rgb = np.ascontiguousarray(np.asarray(crop))
    with _lock:
        result = _get_hair_segmenter().segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        masks = result.confidence_masks
        prob = np.array(masks[-1].numpy_view(), copy=True).squeeze().astype(np.float32)
        del result, masks
    if prob.shape != (crop.height, crop.width):
        prob = np.asarray(Image.fromarray(prob).resize(crop.size, Image.BILINEAR), dtype=np.float32)
    out = np.zeros((h, w), dtype=np.float32)
    out[top:bottom, left:right] = np.clip(prob, 0, 1)
    return out

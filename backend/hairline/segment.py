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
    cats = np.asarray(result.category_mask.numpy_view()).squeeze()
    if cats.shape != (img.height, img.width):
        cats = np.asarray(Image.fromarray(cats.astype(np.uint8)).resize(img.size, Image.NEAREST))
    return cats

"""Face landmarks via MediaPipe Face Landmarker (Apache 2.0)."""
from __future__ import annotations

import os
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = Path(os.environ.get("FACE_LANDMARKER_PATH", Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"))

# Landmark indices in the 478-point MediaPipe face mesh.
FOREHEAD_TOP = 10
CHIN = 152
NOSE_TIP = 1
BROWS = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
             400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
             54, 103, 67, 109]
# Points that should not move when only hair changes (eyes, nose, mouth corners, jaw).
STABLE = [33, 133, 362, 263, 1, 4, 61, 291, 199, 152, 234, 454, 168, 6]


@dataclass
class Face:
    points: np.ndarray  # (478, 2) pixel coordinates

    @property
    def brow_top_y(self) -> float:
        return float(self.points[BROWS, 1].min())

    @property
    def forehead_y(self) -> float:
        return float(self.points[FOREHEAD_TOP, 1])

    @property
    def chin_y(self) -> float:
        return float(self.points[CHIN, 1])

    @property
    def oval_x(self) -> tuple[float, float]:
        xs = self.points[FACE_OVAL, 0]
        return float(xs.min()), float(xs.max())

    @property
    def yaw(self) -> float:
        """Nose offset from the face centre as a fraction of face width (0 = facing camera)."""
        lo, hi = self.oval_x
        return (float(self.points[NOSE_TIP, 0]) - (lo + hi) / 2) / max(hi - lo, 1.0)

    @property
    def width(self) -> float:
        lo, hi = self.oval_x
        return hi - lo

    @property
    def height(self) -> float:
        return self.chin_y - self.forehead_y


_lock = threading.Lock()
_detector = None


def download_model(url: str, path: Path) -> Path:
    """Fetch a MediaPipe model file on first use (models/ is git-ignored)."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


def _get_detector():
    global _detector
    if _detector is None:
        from mediapipe.tasks.python import BaseOptions, vision

        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(download_model(MODEL_URL, MODEL_PATH))),
            num_faces=2,
        )
        _detector = vision.FaceLandmarker.create_from_options(opts)
    return _detector


def detect_faces(img: Image.Image) -> list[Face]:
    import mediapipe as mp

    rgb = np.ascontiguousarray(np.asarray(img.convert("RGB")))
    with _lock:
        result = _get_detector().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    w, h = img.size
    faces = []
    for lms in result.face_landmarks:
        pts = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)
        faces.append(Face(pts))
    return faces

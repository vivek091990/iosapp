"""End-to-end check with real face detection and segmentation, and a fake edit model that
reframes and tints the whole image. Needs network on first run (MediaPipe models and a
public sample portrait from MediaPipe's test assets); skipped when offline."""
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SAMPLE_URL = "https://storage.googleapis.com/mediapipe-assets/portrait.jpg"
CACHE = Path(__file__).resolve().parent.parent / "models" / "sample_portrait.jpg"


class ReframingTintProvider:
    name = "fake"

    def edit(self, img, prompt):
        w, h = img.size
        big = img.resize((int(w * 1.05), int(h * 1.05))).crop((12, 20, 12 + w, 20 + h))
        a = np.asarray(big.resize((1024, int(1024 * h / w)))).astype(float) * 0.4 + np.array([255, 0, 255]) * 0.6
        return Image.fromarray(a.astype(np.uint8))


@pytest.fixture(scope="module")
def photo():
    try:
        if not CACHE.exists():
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(SAMPLE_URL, CACHE)
        from hairline.face import detect_faces

        img = Image.open(CACHE).convert("RGB")
        detect_faces(img)
    except Exception as exc:  # offline or missing system libs (libEGL)
        pytest.skip(f"real-model test unavailable: {exc}")
    return img


def test_face_below_brows_is_untouched(photo):
    from hairline.face import detect_faces
    from hairline.pipeline import run_preview

    face = detect_faces(photo)[0]
    result = run_preview(photo, "Hairline", 50, ReframingTintProvider())
    assert result.masked and result.face_drift < 0.035
    out, orig = np.asarray(result.image).astype(int), np.asarray(photo).astype(int)
    brow = int(face.brow_top_y)
    assert np.abs(out[brow:] - orig[brow:]).max() == 0
    assert np.abs(out[:brow] - orig[:brow]).max() > 50  # hair region was replaced


def test_fill_zone_is_bare_skin_above_target_hairline(photo):
    from hairline.face import detect_faces
    from hairline.mask import fill_zone_mask, target_hairline_y
    from hairline.segment import FACE_SKIN, HAIR, category_mask

    face = detect_faces(photo)[0]
    cats = category_mask(photo)
    zone = fill_zone_mask(face, cats)
    line = target_hairline_y(face, photo.width)
    ys, xs = np.nonzero(zone)
    assert not (zone & (cats == HAIR)).any()
    assert (cats[zone] == FACE_SKIN).all()
    if len(ys):
        assert (ys < line[xs]).all()
    # target hairline sits between the top of the face mesh and the brows
    cx = int(sum(face.oval_x) / 2)
    assert face.forehead_y - 0.2 * face.height < line[cx] < face.brow_top_y

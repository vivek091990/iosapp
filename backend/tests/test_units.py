import io

import numpy as np
import pytest
from PIL import Image

from hairline import composite as comp
from hairline.face import Face
from hairline.mask import hair_region_mask
from hairline.prepare import MAX_PIXELS, load_photo
from hairline.prompts import build_prompt, coverage_phrase
from hairline.providers import PROVIDERS, ProviderError, get_provider
from tests.helpers import fake_face as _fake_face


def test_prompt_mentions_preset_and_strength():
    p = build_prompt("Hairline", 80)
    assert "receding hairline" in p and "full head of hair" in p and "Keep everything else" in p
    assert coverage_phrase(0) != coverage_phrase(50) != coverage_phrase(100)
    with pytest.raises(ValueError):
        build_prompt("mohawk", 50)


def test_load_photo_downscales_and_converts():
    big = Image.new("RGBA", (2000, 1500), (10, 20, 30, 255))
    buf = io.BytesIO()
    big.save(buf, format="PNG")
    img = load_photo(buf.getvalue())
    assert img.mode == "RGB"
    assert img.width * img.height <= MAX_PIXELS * 1.01
    assert abs(img.width / img.height - 2000 / 1500) < 0.01


def test_mask_never_touches_face_below_brows():
    face = _fake_face()
    mask = np.asarray(hair_region_mask((400, 500), face, "hairline"))
    assert mask[: int(face.brow_top_y) - 40].max() == 255  # scalp area editable
    assert mask[int(face.brow_top_y):].max() == 0  # brows and below protected


def test_mask_respects_segmentation():
    face = _fake_face()
    cats = np.zeros((500, 400), dtype=np.uint8)  # all background -> nothing editable
    assert np.asarray(hair_region_mask((400, 500), face, "hairline", cats)).max() == 0


def test_composite_keeps_unmasked_pixels():
    orig = Image.new("RGB", (64, 64), (0, 0, 255))
    edited = Image.new("RGB", (128, 128), (255, 0, 0))
    mask = Image.new("L", (64, 64), 0)
    mask.paste(255, (0, 0, 64, 32))
    out = np.asarray(comp.composite(orig, edited, mask))
    assert tuple(out[10, 10]) == (255, 0, 0)
    assert tuple(out[50, 10]) == (0, 0, 255)


def test_alignment_recovers_shift():
    face = _fake_face()
    rng = np.random.default_rng(0)
    face.points[:] += rng.normal(0, 30, face.points.shape).astype(np.float32)  # spread points out
    shifted = Face(face.points * 1.05 + np.array([12, -8], dtype=np.float32))
    img = Image.new("RGB", (400, 500))
    aligned = comp.align_to_original(img, face, shifted, (400, 500))
    assert aligned.residual < 0.01


def test_provider_registry():
    assert {"nano-banana-2", "nano-banana-pro", "gpt-image", "flux-kontext", "flux-2", "qwen-edit"} <= set(PROVIDERS)
    assert get_provider("nano-banana-2").name == "nano-banana-2"
    with pytest.raises(ProviderError):
        get_provider("nope")

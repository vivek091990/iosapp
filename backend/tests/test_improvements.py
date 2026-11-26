import numpy as np
from PIL import Image

import hairline.pipeline as pl
from hairline.quality import photo_problem
from hairline.segment import HAIR
from hairline.tone import match_hair_tone
from tests.helpers import fake_face


class Recorder:
    name = "rec"

    def __init__(self):
        self.prompts = []

    def edit(self, img, prompt):
        self.prompts.append(prompt)
        return img


def test_auto_without_face_uses_crown(monkeypatch):
    monkeypatch.setattr(pl, "detect_faces", lambda img: [])
    rec = Recorder()
    result = pl.run_preview(Image.new("RGB", (200, 200), (90, 90, 90)), "Auto", 50, rec)
    assert result.preset == "crown" and "crown" in rec.prompts[0] and not result.masked


def test_auto_with_face_uses_hairline(monkeypatch):
    face = fake_face(400, 500)
    monkeypatch.setattr(pl, "detect_faces", lambda img: [face])
    monkeypatch.setattr(pl, "category_mask", lambda img: None)
    rec = Recorder()
    result = pl.run_preview(Image.new("RGB", (400, 500), (150, 120, 100)), "auto", 50, rec)
    assert result.preset == "hairline" and "receding hairline" in rec.prompts[0]


def test_quality_checks():
    photo = Image.new("RGB", (400, 500), (150, 120, 100))
    assert photo_problem(photo, fake_face(400, 500)) is None
    assert "too small" in photo_problem(Image.new("RGB", (4000, 500), (150, 120, 100)), fake_face(400, 500))
    turned = fake_face(400, 500)
    turned.points[1, 0] += 60  # nose far off centre
    assert "Face the camera" in photo_problem(photo, turned)
    assert "too dark" in photo_problem(Image.new("RGB", (400, 500), (10, 10, 10)), fake_face(400, 500))


def test_tone_moves_new_hair_toward_original():
    h, w = 100, 100
    original = Image.new("RGB", (w, h), (40, 30, 25))  # dark brown hair everywhere
    edited = Image.new("RGB", (w, h), (90, 70, 55))  # model returned lighter hair
    cats = np.full((h, w), HAIR, dtype=np.uint8)
    mask = Image.new("L", (w, h), 255)
    toned, shift = match_hair_tone(original, edited, cats, cats, mask)
    assert shift > 0
    before = np.abs(np.asarray(edited, int) - np.asarray(original, int)).mean()
    after = np.abs(np.asarray(toned, int) - np.asarray(original, int)).mean()
    assert after < before * 0.6


def test_tone_skips_without_hair():
    img = Image.new("RGB", (50, 50))
    cats = np.zeros((50, 50), dtype=np.uint8)
    out, shift = match_hair_tone(img, img, cats, cats, Image.new("L", (50, 50), 255))
    assert shift == 0 and out is img

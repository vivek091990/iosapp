import numpy as np
import pytest
from PIL import Image

from hairline.segment import HAIR
from tests.helpers import fake_face
from training import build_dataset
from training.recede import Unsuitable, recede

SKIN, DARK = (200, 160, 130), (30, 25, 20)


def _head(hairline_y=148, w=400, h=500):
    arr = np.full((h, w, 3), SKIN, dtype=np.uint8)
    arr[:hairline_y] = DARK
    cats = np.zeros((h, w), dtype=np.uint8)
    cats[:hairline_y] = HAIR
    return Image.fromarray(arr), cats


def test_recede_moves_hairline_up_and_keeps_face():
    img, cats = _head()
    face = fake_face(400, 500)
    rec = recede(img, face, cats, severity=0.12, seed=1)
    before, after = np.asarray(img).astype(int), np.asarray(rec.image).astype(int)
    cols = slice(140, 260)
    # Skin now shows where hair was just above the old hairline.
    assert np.abs(after[130:146, cols] - np.array(SKIN)).mean() < np.abs(before[130:146, cols] - np.array(SKIN)).mean() - 50
    # Hair remains higher up (recession is partial), and the face below the brows is untouched.
    assert np.abs(after[:40, cols] - np.array(DARK)).mean() < 10
    assert np.abs(after[220:] - before[220:]).max() == 0
    assert np.asarray(rec.mask).max() > 200


def test_recede_temples_deeper_than_middle():
    img, cats = _head()
    face = fake_face(400, 500)
    after = np.asarray(recede(img, face, cats, severity=0.12, seed=3).image).astype(int)
    skin_rows = lambda x: int((np.abs(after[:150, x] - np.array(SKIN)).sum(axis=1) < 60).sum())
    lo, hi = face.oval_x
    cx, half = (lo + hi) / 2, (hi - lo) / 2
    temple = max(skin_rows(int(cx - 0.65 * half)), skin_rows(int(cx + 0.65 * half)))
    assert temple > skin_rows(int(cx))


def test_bangs_are_rejected():
    img, cats = _head(hairline_y=195)  # hair down to just above the brows
    with pytest.raises(Unsuitable):
        recede(img, fake_face(400, 500), cats, severity=0.1)


def test_split_is_by_person_and_stable():
    assert build_dataset.split_for("person_001", 0.1) == build_dataset.split_for("person_001", 0.1)
    splits = {build_dataset.split_for(f"person_{i:03d}", 0.2) for i in range(50)}
    assert splits == {"train", "test"}


def test_people_grouping(tmp_path):
    (tmp_path / "alice").mkdir()
    for name in ("alice/a.jpg", "alice/b.HEIC", "solo.png", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    groups = build_dataset.people(tmp_path)
    assert sorted(groups) == ["alice", "solo"] and len(groups["alice"]) == 2

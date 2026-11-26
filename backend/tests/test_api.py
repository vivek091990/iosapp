import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import main
from hairline import PipelineResult, PreviewError
from tests.helpers import fake_face


def _png(color=(200, 150, 120)):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _clear_cache():
    main._cache.clear()


def _post(client, data=None, **form):
    form = {"preset": "Hairline", "coverage": "50", **form}
    return client.post("/v1/preview", files={"file": ("p.png", data or _png(), "image/png")}, data=form)


def test_preview_returns_png_and_caches(monkeypatch):
    calls = []

    def fake_run(photo, preset, coverage, provider):
        calls.append(preset)
        return PipelineResult(photo, provider.name, 0.1, 1, 0.01, True)

    monkeypatch.setattr(main, "run_preview", fake_run)
    client = TestClient(main.app)
    r1 = _post(client)
    assert r1.status_code == 200 and r1.headers["content-type"] == "image/png"
    assert r1.headers["X-Hairline-Provider"] == "nano-banana-pro"
    Image.open(io.BytesIO(r1.content)).verify()
    r2 = _post(client)
    assert r2.status_code == 200 and len(calls) == 1


def test_photo_problem_is_422(monkeypatch):
    def fake_run(*a, **k):
        raise PreviewError("We couldn't find a face.")

    monkeypatch.setattr(main, "run_preview", fake_run)
    r = _post(TestClient(main.app))
    assert r.status_code == 422 and "face" in r.json()["detail"]


def test_missing_key_is_503(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    import hairline.pipeline as pl

    monkeypatch.setattr(pl, "detect_faces", lambda img: [fake_face(64, 64)])
    monkeypatch.setattr(pl, "category_mask", lambda img: None)
    r = _post(TestClient(main.app))
    assert r.status_code == 503 and "GEMINI_API_KEY" in r.json()["detail"]


def test_bad_file_and_unknown_model():
    client = TestClient(main.app)
    assert _post(client, data=b"not an image").status_code == 422
    assert _post(client, model="nope").status_code == 400

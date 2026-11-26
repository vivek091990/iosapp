import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("bakeoff", Path(__file__).resolve().parent.parent / "scripts" / "bakeoff.py")
bakeoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bakeoff)


def test_report_has_ratings_and_summary(tmp_path):
    rows = [
        {"photo": "a.jpg", "preset": "auto", "coverage": 50, "provider": "nano-banana-2", "file": "a__auto__nb.jpg",
         "seconds": 2.1, "attempts": 1, "face_drift": 0.01, "notes": []},
        {"photo": "a.jpg", "preset": "auto", "coverage": 50, "provider": "flux-kontext", "error": "boom"},
    ]
    path = bakeoff.write_html(tmp_path, rows, ["nano-banana-2", "flux-kontext"], blind=True)
    page = path.read_text()
    assert "class='rate'" in page and "Export ratings" in page and "boom" in page
    assert "nano-banana-2" not in page  # blind mode hides names
    data = json.loads((tmp_path / "results.json").read_text())
    assert set(data["labels"]) == {"nano-banana-2", "flux-kontext"}
    nb = next(s for s in data["summary"] if s["provider"] == "nano-banana-2")
    assert nb["ok"] == 1 and nb["median_seconds"] == 2.1

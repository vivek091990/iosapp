"""Race the edit models on a folder of photos and write a side-by-side HTML report.

Usage (from backend/):
    python scripts/bakeoff.py PHOTOS_DIR [--providers nano-banana-2,flux-kontext] \
        [--presets hairline,crown,part] [--coverage 50] [--blind]

Only providers whose API key is set are run unless --providers names them.
Each cell shows the composited result, seconds and face drift; --blind hides model names
(the key is saved in results.json) so people can rate realism without bias.
"""
from __future__ import annotations

import argparse
import html
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hairline import PreviewError, run_preview  # noqa: E402
from hairline.prepare import load_photo  # noqa: E402
from hairline.providers import PROVIDERS, configured_providers, get_provider  # noqa: E402

PHOTO_EXT = {".jpg", ".jpeg", ".png", ".heic", ".webp"}


def run_one(photo_path: Path, preset: str, coverage: int, provider_name: str, out_dir: Path) -> dict:
    row = {"photo": photo_path.name, "preset": preset, "coverage": coverage, "provider": provider_name}
    try:
        photo = load_photo(photo_path.read_bytes())
        result = run_preview(photo, preset, coverage, get_provider(provider_name))
    except PreviewError as exc:
        return {**row, "error": str(exc)}
    except Exception as exc:  # keep the race going when one model fails
        return {**row, "error": f"{type(exc).__name__}: {exc}"}
    name = f"{photo_path.stem}__{preset}__{provider_name}.jpg"
    result.image.save(out_dir / name, quality=92)
    return {
        **row,
        "file": name,
        "seconds": round(result.seconds, 2),
        "attempts": result.attempts,
        "face_drift": None if result.face_drift is None else round(result.face_drift, 4),
        "notes": result.notes,
    }


def write_html(out_dir: Path, rows: list[dict], providers: list[str], blind: bool) -> Path:
    labels = {p: p for p in providers}
    if blind:
        shuffled = providers[:]
        random.shuffle(shuffled)
        labels = {p: f"Model {chr(65 + i)}" for i, p in enumerate(shuffled)}
    cols = sorted(providers, key=lambda p: labels[p])
    by_key: dict[tuple[str, str], dict[str, dict]] = {}
    for r in rows:
        by_key.setdefault((r["photo"], r["preset"]), {})[r["provider"]] = r

    parts = [
        "<!doctype html><meta charset='utf-8'><title>Hairline bake-off</title>",
        "<style>body{font:14px system-ui;margin:16px}table{border-collapse:collapse}td,th{padding:6px;vertical-align:top;border:1px solid #ddd}"
        "img{width:260px;display:block}small{color:#666}.err{color:#b00;width:260px}</style>",
        "<h1>Hairline bake-off</h1><table><tr><th>Photo / preset</th><th>Original</th>",
    ]
    parts += [f"<th>{html.escape(labels[p])}</th>" for p in cols]
    parts.append("</tr>")
    for (photo, preset), cells in sorted(by_key.items()):
        parts.append(f"<tr><td>{html.escape(photo)}<br><small>{html.escape(preset)}</small></td>")
        parts.append(f"<td><img src='originals/{html.escape(photo)}'></td>")
        for p in cols:
            r = cells.get(p)
            if r is None:
                parts.append("<td></td>")
            elif "error" in r:
                parts.append(f"<td><div class='err'>{html.escape(r['error'])}</div></td>")
            else:
                meta = f"{r['seconds']} s" + (f" · drift {r['face_drift']}" if r["face_drift"] is not None else "")
                parts.append(f"<td><img src='{html.escape(r['file'])}'><small>{html.escape(meta)}</small></td>")
        parts.append("</tr>")
    parts.append("</table>")
    path = out_dir / "index.html"
    path.write_text("".join(parts))
    (out_dir / "results.json").write_text(json.dumps({"labels": labels, "rows": rows}, indent=2))
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("photos", type=Path)
    ap.add_argument("--providers", default="")
    ap.add_argument("--presets", default="hairline")
    ap.add_argument("--coverage", type=int, default=50)
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("bakeoff-out"))
    args = ap.parse_args()

    providers = [p for p in args.providers.split(",") if p] or configured_providers()
    unknown = [p for p in providers if p not in PROVIDERS]
    if unknown or not providers:
        sys.exit(f"No usable providers (unknown: {unknown}). Set API keys or pass --providers from: {', '.join(PROVIDERS)}")
    photos = sorted(p for p in args.photos.iterdir() if p.suffix.lower() in PHOTO_EXT)
    if not photos:
        sys.exit(f"No photos in {args.photos}")

    out_dir = args.out / time.strftime("%Y%m%d-%H%M%S")
    (out_dir / "originals").mkdir(parents=True)
    for p in photos:
        load_photo(p.read_bytes()).save(out_dir / "originals" / p.name, quality=92)

    jobs = [(ph, pr, pv) for ph in photos for pr in args.presets.split(",") for pv in providers]
    print(f"{len(jobs)} runs: {len(photos)} photos x {args.presets} x {', '.join(providers)}")
    with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as pool:
        rows = list(pool.map(lambda j: run_one(j[0], j[1], args.coverage, j[2], out_dir), jobs))
    for r in rows:
        status = r.get("error") or f"{r['seconds']} s, drift {r['face_drift']}"
        print(f"  {r['photo']:<24} {r['preset']:<9} {r['provider']:<16} {status}")
    print(f"Report: {write_html(out_dir, rows, providers, args.blind)}")


if __name__ == "__main__":
    main()

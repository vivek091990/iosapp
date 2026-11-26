"""Turn a folder of consented full-hairline photos into training pairs.

Usage (from backend/):
    python -m training.build_dataset ~/Documents/hairline-training out/dataset \
        [--size 1024] [--variants 2] [--test-frac 0.1] [--limit 0]

Input: one subfolder per person (person_001/, person_002/, ...), or a flat folder where
each photo counts as its own person. Splitting is by person, so nobody appears in both
train and test.

Output (the control/target/caption layout that LoRA trainers for edit models read):
    out/train/control/<id>.jpg   synthetic receded "before"
    out/train/target/<id>.jpg    the real photo (the "after")
    out/train/target/<id>.txt    the instruction, same wording the app sends
    out/train/mask/<id>.png      where hair was removed
    out/test/...                 same layout
    out/manifest.jsonl, out/skipped.csv, out/review.html
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hairline.face import detect_faces  # noqa: E402
from hairline.prepare import load_photo  # noqa: E402
from hairline.prompts import build_prompt  # noqa: E402
from hairline.quality import photo_problem  # noqa: E402
from hairline.segment import HAIR, category_mask  # noqa: E402
from training.recede import Unsuitable, recede  # noqa: E402

PHOTO_EXT = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
SEVERITIES = (0.05, 0.08, 0.11, 0.14)  # temple depth as a fraction of face height
MIN_HAIR_FRACTION = 0.25  # hair pixels above the forehead vs the head dome: needs a real hairline


def people(src: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for p in sorted(src.rglob("*")):
        if p.suffix.lower() in PHOTO_EXT and p.is_file():
            person = p.parent.name if p.parent != src else p.stem
            groups.setdefault(person, []).append(p)
    return groups


def split_for(person: str, test_frac: float) -> str:
    bucket = int(hashlib.sha256(person.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if bucket < test_frac else "train"


def head_crop_box(img: Image.Image, face) -> tuple[int, int, int, int]:
    """Square box around the head with room above for hair, centred on the face."""
    cx = (face.oval_x[0] + face.oval_x[1]) / 2
    top = face.forehead_y - 0.9 * face.height
    bottom = face.chin_y + 0.25 * face.height
    side = max(bottom - top, 2.0 * face.width)
    cy = (top + bottom) / 2
    return (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))


def crop_square(img: Image.Image, box: tuple[int, int, int, int], size: int) -> Image.Image:
    """Crop, padding with edge pixels where the box leaves the photo, then resize."""
    l, t, r, b = box
    pad = max(0, -l, -t, r - img.width, b - img.height)
    if pad:
        arr = np.pad(np.asarray(img), ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        img = Image.fromarray(arr)
        l, t, r, b = l + pad, t + pad, r + pad, b + pad
    return img.crop((l, t, r, b)).resize((size, size), Image.LANCZOS)


def hair_coverage(cats: np.ndarray, face) -> float:
    lo, hi = face.oval_x
    top = int(max(0, face.forehead_y - 0.6 * face.height))
    region = cats[top:int(face.forehead_y), int(max(0, lo)):int(hi)]
    return float((region == HAIR).mean()) if region.size else 0.0


def process(path: Path, person: str, split: str, out: Path, size: int, variants: int) -> tuple[list[dict], str | None]:
    img = load_photo(path.read_bytes(), max_pixels=6_000_000)
    faces = detect_faces(img)
    if len(faces) != 1:
        return [], f"{len(faces)} faces"
    crop = crop_square(img, head_crop_box(img, faces[0]), size)
    faces = detect_faces(crop)
    if len(faces) != 1:
        return [], "face lost after crop"
    face = faces[0]
    problem = photo_problem(crop, face)
    if problem:
        return [], problem
    cats = category_mask(crop)
    cover = hair_coverage(cats, face)
    if cover < MIN_HAIR_FRACTION:
        return [], f"not enough hair above the forehead ({cover:.2f}); can't be a 'full hairline' target"

    rows = []
    base = hashlib.sha256(f"{person}/{path.name}".encode()).hexdigest()[:12]
    prompt = build_prompt("hairline", 50)
    for v in range(variants):
        severity = SEVERITIES[(int(base, 16) + v) % len(SEVERITIES)]
        try:
            rec = recede(crop, face, cats, severity, seed=int(base, 16) + v)
        except Unsuitable as exc:
            return rows, str(exc)
        item = f"{base}_{v}"
        d = out / split
        for sub in ("control", "target", "mask"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        rec.image.save(d / "control" / f"{item}.jpg", quality=95)
        crop.save(d / "target" / f"{item}.jpg", quality=95)
        (d / "target" / f"{item}.txt").write_text(prompt)
        rec.mask.save(d / "mask" / f"{item}.png")
        rows.append({"id": item, "person": person, "source": str(path.name), "split": split,
                     "severity": severity, "hair_coverage": round(cover, 3)})
    return rows, None


def write_review(out: Path, rows: list[dict], limit: int = 80) -> Path:
    parts = ["<!doctype html><meta charset='utf-8'><title>Training pairs</title>",
             "<style>body{font:14px system-ui;margin:16px}figure{display:inline-block;margin:6px}"
             "img{width:220px;display:block}figcaption{color:#666;font-size:12px}</style>",
             f"<h1>Training pairs</h1><p>{len(rows)} pairs. Left: synthetic before. Right: real after.</p>"]
    for r in rows[:limit]:
        i, s = html.escape(r["id"]), r["split"]
        parts.append(f"<div><figure><img src='{s}/control/{i}.jpg'><figcaption>{i} before (severity {r['severity']})</figcaption></figure>"
                     f"<figure><img src='{s}/target/{i}.jpg'><figcaption>{html.escape(r['person'])} after ({s})</figcaption></figure></div>")
    path = out / "review.html"
    path.write_text("".join(parts))
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--variants", type=int, default=2, help="receded versions per photo")
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many photos (0 = all)")
    args = ap.parse_args()

    groups = people(args.src)
    if not groups:
        sys.exit(f"No photos in {args.src}")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest, skipped, done = [], [], 0
    for person, paths in groups.items():
        split = split_for(person, args.test_frac)
        for p in paths:
            if args.limit and done >= args.limit:
                break
            done += 1
            try:
                rows, reason = process(p, person, split, args.out, args.size, args.variants)
            except Exception as exc:  # one bad file shouldn't stop the run
                rows, reason = [], f"{type(exc).__name__}: {exc}"
            manifest += rows
            if reason:
                skipped.append({"person": person, "file": p.name, "reason": reason})
            print(f"{person}/{p.name}: {reason or f'{len(rows)} pairs'}")

    with open(args.out / "manifest.jsonl", "w") as f:
        for r in manifest:
            f.write(json.dumps(r) + "\n")
    with open(args.out / "skipped.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["person", "file", "reason"])
        w.writeheader()
        w.writerows(skipped)
    n_train = sum(r["split"] == "train" for r in manifest)
    print(f"\n{len(manifest)} pairs ({n_train} train, {len(manifest) - n_train} test) from {done} photos; "
          f"{len(skipped)} skipped. Review: {write_review(args.out, manifest)}")


if __name__ == "__main__":
    main()

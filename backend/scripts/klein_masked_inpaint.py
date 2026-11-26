"""Local hairline fill with FLUX.2 klein 4B on a Mac: masked inpainting, seed sweep, auto-pick.

Three phases, run one after another (never in parallel: the sampler needs the memory):

    cd backend
    python scripts/klein_masked_inpaint.py prep   PHOTO.jpg [PHOTO.jpg ...] --out OUT [--coverage 50]
    python scripts/klein_masked_inpaint.py sample OUT [--seeds 42,7,123,2024]
    python scripts/klein_masked_inpaint.py score  OUT

prep   (backend venv)  face landmarks, 512 px hair probability, the shared fill zone and inpaint
                       mask, a head crop enlarged to ~1024 px, a hair-coloured prefill of the zone
                       with an irregular front edge and a few stray strands, and the token mask.
sample (no ML imports) runs klein_masked_sampler.py with mflux's Python for every photo x seed,
                       one at a time with --low-ram, and records wall time and peak memory.
score  (backend venv)  pastes each result back (lighting correction + pyramid blend), rejects any
                       result that changes a pixel outside the mask, scores the rest and picks the
                       best per photo; writes OUT/sheet.jpg and OUT/scores.json.

Scores: coverage = % of the fill zone with hair probability >= 0.5 on the result; colour = Lab
delta-E between new hair in the zone and the existing hair just above it; seam = mean gradient
magnitude across the target line divided by that inside the existing hair; detail = fine strand
detail (mean |Laplacian|) inside the zone divided by that of his existing hair, which catches a
flat hair-coloured patch (klein keeping the prefill) that coverage alone scores as perfect.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MFLUX_PYTHON = Path(os.environ.get("MFLUX_PYTHON", Path.home() / ".local/share/uv/tools/mflux/bin/python"))
MODEL = os.environ.get("MFLUX_MODEL", "Runpod/FLUX.2-klein-4B-mflux-4bit")
PROMPT = "photo of a man with dense dark curly hair and a full natural hairline, realistic hair strands, soft indoor light"
SAMPLER = Path(__file__).resolve().parent / "klein_masked_sampler.py"
PRIOR = 0.5
CROP_LONG_SIDE = 1024


# ---------------------------------------------------------------- prep
def prep(photos: list[Path], out: Path, coverage: int) -> None:
    import cv2
    import numpy as np
    from PIL import Image

    from hairline.face import detect_faces
    from hairline.mask import _ipd, fill_zone_mask, inpaint_mask, target_hairline_y
    from hairline.prepare import load_photo
    from hairline.segment import HAIR, category_mask, hair_probability

    out.mkdir(parents=True, exist_ok=True)
    for path in photos:
        name = path.stem
        d = out / name
        d.mkdir(exist_ok=True)
        img = load_photo(path.read_bytes()) if path.suffix.lower() == ".heic" else Image.open(path).convert("RGB")
        img.save(d / "original.png")
        w, h = img.size
        face = detect_faces(img)[0]
        cats = category_mask(img)
        lo, hi = face.oval_x
        fw, fh = face.width, face.height
        box = (int(max(0, lo - 0.45 * fw)), int(max(0, face.forehead_y - 0.9 * fh)),
               int(min(w, hi + 0.45 * fw)), int(min(h, face.brow_top_y)))
        hp = hair_probability(img, box)
        zone = fill_zone_mask(face, cats, hp, coverage=coverage)
        alpha = inpaint_mask(face, zone, cats).astype(np.float32)
        ipd = _ipd(face)
        line = target_hairline_y(face, w, coverage=coverage)
        rng = np.random.default_rng(7)

        # Irregular front edge: per-column offsets (~0.015 IPD) of the zone's lower edge.
        cols = np.where(zone.any(0))[0]
        prefill = zone.copy()
        if len(cols):
            off = cv2.GaussianBlur(rng.standard_normal((1, w)).astype(np.float32), (0, 0), 2.0)[0]
            off *= 0.015 * ipd / (np.abs(off[cols]).max() + 1e-6)
            yy = np.arange(h)[:, None]
            edge = line[None, :] + off[None, :]
            bottom = np.where(zone.any(0), h - 1 - np.argmax(zone[::-1], 0), -1)
            near = (yy > bottom[None, :] - 0.04 * ipd) & (zone.any(0)[None, :])
            prefill = (zone & ~(near & (yy > edge))) | (near & (yy <= edge) & (cats != HAIR) & (yy < bottom[None, :] + 0.03 * ipd))

        # Hair colour of his existing hair, with a little texture.
        arr = np.asarray(img).astype(np.float32)
        hair_px = arr[(hp > 0.7)]
        colour = np.median(hair_px, 0) if len(hair_px) > 50 else np.array([40, 28, 22], np.float32)
        tex = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 1.5)[..., None] * 18
        a = cv2.GaussianBlur(prefill.astype(np.float32), (0, 0), 2.0)[..., None]
        filled = arr * (1 - a) + (colour + tex) * a
        # A few sparse single strands just below the edge.
        strands = np.zeros((h, w), np.float32)
        if len(cols):
            for x in rng.choice(cols, size=min(len(cols) // 10, 25), replace=False):
                y0 = int(bottom[x])
                length = rng.uniform(0.03, 0.06) * ipd
                ang = np.deg2rad(rng.uniform(70, 110))
                x1, y1 = int(x + length * np.cos(ang)), int(y0 + length * np.sin(ang))
                cv2.line(strands, (int(x), y0), (x1, y1), float(rng.uniform(0.5, 0.8)), 1, cv2.LINE_AA)
        filled = filled * (1 - strands[..., None]) + colour * strands[..., None]
        strand_area = cv2.dilate((strands > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        gen = np.maximum(alpha, (prefill | strand_area).astype(np.float32))

        # Head crop around the generation mask, ~4:3, long side ~1024, multiple of 16.
        ys, xs = np.where(gen > 0.02)
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        side = max((xs.max() - xs.min()) * 1.6, (ys.max() - ys.min()) * 2.2 * 4 / 3)
        x0, x1 = int(max(0, cx - side / 2)), int(min(w, cx + side / 2))
        y0, y1 = int(max(0, cy - side * 0.75 * 0.55)), int(min(h, cy + side * 0.75 * 0.45))
        s = CROP_LONG_SIDE / max(x1 - x0, y1 - y0)
        ow, oh = int(round((x1 - x0) * s / 16)) * 16, int(round((y1 - y0) * s / 16)) * 16
        crop = Image.fromarray(np.clip(filled, 0, 255).astype(np.uint8)).crop((x0, y0, x1, y1))
        crop.resize((ow, oh), Image.LANCZOS).save(d / "prefilled_crop.png")
        g = cv2.resize(gen[y0:y1, x0:x1], (ow, oh), interpolation=cv2.INTER_LINEAR)
        tok = (cv2.resize(g, (ow // 16, oh // 16), interpolation=cv2.INTER_AREA) > 0.05).astype(np.float32)
        np.save(d / "tokmask.npy", tok)
        np.save(d / "zone.npy", zone)
        np.save(d / "alpha.npy", np.maximum(alpha, cv2.GaussianBlur(strand_area.astype(np.float32), (0, 0), 1.0)))
        np.save(d / "line.npy", line.astype(np.float32))
        json.dump({"box": [x0, y0, x1, y1], "size": [ow, oh], "ipd": ipd, "head_box": box, "coverage": coverage},
                  open(d / "meta.json", "w"))
        print(f"{name}: zone {int(zone.sum())} px, crop {x1 - x0}x{y1 - y0} -> {ow}x{oh}, "
              f"{int(tok.sum())}/{tok.size} tokens to generate")


# ---------------------------------------------------------------- sample
def sample(out: Path, seeds: list[int]) -> None:
    for d in sorted(p for p in out.iterdir() if (p / "meta.json").exists()):
        meta = json.load(open(d / "meta.json"))
        ow, oh = meta["size"]
        for seed in seeds:
            dst = d / f"gen_{seed}.png"
            cmd = ["/usr/bin/time", "-l", str(MFLUX_PYTHON), str(SAMPLER), str(d / "tokmask.npy"), "--prior", str(PRIOR),
                   "--", "--low-ram", "--model", MODEL, "--base-model", "flux2-klein-4b",
                   "--image-paths", str(d / "prefilled_crop.png"), "--prompt", PROMPT, "--steps", "4",
                   "--seed", str(seed), "--width", str(ow), "--height", str(oh), "--output", str(dst)]
            t0 = time.time()
            run = subprocess.run(cmd, capture_output=True, text=True)
            secs = time.time() - t0
            peak = re.search(r"(\d+)\s+peak memory footprint", run.stderr + run.stdout)
            peak_gb = int(peak.group(1)) / 1e9 if peak else None
            ok = run.returncode == 0 and dst.exists()
            json.dump({"seconds": round(secs, 1), "peak_gb": round(peak_gb, 2) if peak_gb else None, "ok": ok},
                      open(d / f"gen_{seed}.json", "w"))
            print(f"{d.name} seed {seed}: {'ok' if ok else 'FAILED'} {secs:.0f}s peak {peak_gb or 0:.1f} GB")
            if not ok:
                print((run.stderr or run.stdout)[-2000:])
            if peak_gb and peak_gb > 8.0:
                sys.exit(f"peak memory {peak_gb:.1f} GB is over the 8 GB limit; stopping")


# ---------------------------------------------------------------- score
def score(out: Path) -> None:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    from hairline.composite import lighting_correction, pyramid_blend
    from hairline.face import detect_faces
    from hairline.segment import hair_probability

    def lab(x):
        return cv2.cvtColor(np.asarray(x).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)

    rows, report = [], {}
    for d in sorted(p for p in out.iterdir() if (p / "meta.json").exists()):
        meta = json.load(open(d / "meta.json"))
        x0, y0, x1, y1 = meta["box"]
        ipd = meta["ipd"]
        orig = Image.open(d / "original.png").convert("RGB")
        w, h = orig.size
        zone, alpha, line = np.load(d / "zone.npy"), np.load(d / "alpha.npy"), np.load(d / "line.npy")
        face = detect_faces(orig)[0]
        hp0 = hair_probability(orig, meta["head_box"])
        # existing hair band just above the zone (anchor at the top row: the zone grows upward)
        up = int(0.15 * ipd)
        k = np.zeros((up + 1, 1), np.uint8)
        k[:] = 1
        above = (cv2.dilate(zone.astype(np.uint8), k, anchor=(0, 0)) > 0) & ~zone & (hp0 >= 0.5)
        cols = zone.any(0)
        yy = np.arange(h)[:, None]
        seam_band = (np.abs(yy - line[None, :]) <= 3) & cols[None, :]
        inner = cv2.erode(zone.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        cands = []
        for g in sorted(d.glob("gen_*.png")):
            seed = int(g.stem.split("_")[1])
            info = json.load(open(d / f"gen_{seed}.json"))
            full = orig.copy()
            full.paste(Image.open(g).convert("RGB").resize((x1 - x0, y1 - y0), Image.LANCZOS), (x0, y0))
            full = lighting_correction(orig, full, alpha > 0.02, sigma=0.05 * face.width)
            res = pyramid_blend(orig, full, alpha)
            o, r = np.asarray(orig).astype(int), np.asarray(res).astype(int)
            outside = float(np.abs(r[alpha == 0] - o[alpha == 0]).max()) if (alpha == 0).any() else 0.0
            hp = hair_probability(res, meta["head_box"])
            new_hair = zone & (hp >= 0.5)
            coverage = 100.0 * new_hair.sum() / max(zone.sum(), 1)
            lr = lab(res)
            de = float(np.linalg.norm(lr[new_hair].mean(0) - lab(orig)[above].mean(0))) if new_hair.sum() > 30 and above.sum() > 30 else 99.0
            gray = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY).astype(np.float32)
            gm = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
            seam = float(gm[seam_band].mean() / max(gm[above].mean(), 1e-3)) if above.any() else 9.0
            # Strand detail: a flat hair-coloured patch (klein kept the prefill) has coverage but no
            # strands, so compare fine detail inside the zone with his existing hair.
            lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
            detail = float(lap[inner].mean() / max(lap[above].mean(), 1e-3)) if inner.any() and above.any() else 0.0
            total = coverage / 100 - de / 50 - 0.6 * abs(np.log(max(detail, 1e-3))) - 0.1 * abs(np.log(max(seam, 1e-3)))
            c = {"seed": seed, "coverage": round(coverage, 1), "delta_e": round(de, 1), "seam": round(seam, 2),
                 "detail": round(detail, 2),
                 "outside_max_change": outside, "rejected": outside != 0, "score": round(float(total), 3), **info}
            res.save(d / f"final_{seed}.png")
            cands.append(c)
            print(f"{d.name} seed {seed}: coverage {coverage:.0f}%  dE {de:.1f}  detail {detail:.2f}  seam {seam:.2f}  "
                  f"outside {outside:.0f}  score {total:.3f}{'  REJECTED' if outside else ''}")
        ok = [c for c in cands if not c["rejected"]]
        best = max(ok, key=lambda c: c["score"]) if ok else None
        report[d.name] = {"best": best, "candidates": cands}
        if best:
            Image.open(d / f"final_{best['seed']}.png").save(d / "best.png")
            rows.append((d.name, orig, Image.open(d / "best.png"), best, meta))
    json.dump(report, open(out / "scores.json", "w"), indent=2)
    # sheet: per photo, head close-up original | best, with scores
    tiles = []
    for name, orig, best_img, b, meta in rows:
        hb = meta["head_box"]
        pad = int(0.1 * (hb[2] - hb[0]))
        box = (max(0, hb[0] - pad), max(0, hb[1] - pad), min(orig.width, hb[2] + pad), min(orig.height, hb[3] + 2 * pad))
        a, b2 = orig.crop(box), best_img.crop(box)
        sc = 480 / a.height
        a, b2 = a.resize((int(a.width * sc), 480)), b2.resize((int(b2.width * sc), 480))
        t = Image.new("RGB", (a.width * 2 + 10, 480 + 44), "white")
        t.paste(a, (0, 44))
        t.paste(b2, (a.width + 10, 44))
        dr = ImageDraw.Draw(t)
        dr.text((6, 4), f"{name}  original", fill="black")
        dr.text((a.width + 16, 4), f"best: seed {b['seed']}  coverage {b['coverage']}%  dE {b['delta_e']}  detail {b['detail']}  seam {b['seam']}", fill="black")
        dr.text((a.width + 16, 22), f"{b['seconds']} s, peak {b['peak_gb']} GB, outside change {b['outside_max_change']:.0f}", fill="black")
        tiles.append(t)
    if tiles:
        sheet = Image.new("RGB", (max(t.width for t in tiles), sum(t.height + 10 for t in tiles)), "white")
        y = 0
        for t in tiles:
            sheet.paste(t, (0, y))
            y += t.height + 10
        sheet.save(out / "sheet.jpg", quality=92)
        print(f"wrote {out / 'sheet.jpg'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("photos", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--coverage", type=int, default=50)
    s = sub.add_parser("sample")
    s.add_argument("out", type=Path)
    s.add_argument("--seeds", default="42,7,123,2024")
    c = sub.add_parser("score")
    c.add_argument("out", type=Path)
    args = ap.parse_args()
    if args.cmd == "prep":
        prep(args.photos, args.out, args.coverage)
    elif args.cmd == "sample":
        sample(args.out, [int(x) for x in args.seeds.split(",")])
    else:
        score(args.out)


if __name__ == "__main__":
    main()

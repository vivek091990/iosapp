from __future__ import annotations

import hashlib
import io
from typing import Dict

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, ImageDraw, ImageFont

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache: Dict[str, bytes] = {}


def _coverage_bucket(value: int) -> int:
    return max(0, min(100, value)) // 10


def _cache_key(image_bytes: bytes, preset: str, coverage: int) -> str:
    coverage_bucket = _coverage_bucket(coverage)
    digest = hashlib.sha256()
    digest.update(image_bytes)
    digest.update(preset.encode("utf-8"))
    digest.update(str(coverage_bucket).encode("utf-8"))
    return digest.hexdigest()


def _add_watermark(image_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(image_bytes)).convert("RGBA") as base:
        width, height = base.size
        overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        text = "HairStyle Preview"
        font = ImageFont.load_default()
        text_width, text_height = draw.textsize(text, font=font)
        spacing = max(text_width, text_height) * 2
        for offset in range(-height, width, spacing):
            draw.text((offset, height - offset), text, font=font, fill=(255, 255, 255, 120))

        rotated = overlay.rotate(-30, expand=1)
        combined = Image.alpha_composite(base, rotated.crop((0, 0, width, height)))
        output = io.BytesIO()
        combined.convert("RGBA").save(output, format="PNG")
        return output.getvalue()


@app.post("/v1/preview")
async def preview(file: UploadFile = File(...), preset: str = Form(...), coverage: int = Form(...)) -> Response:
    image_bytes = await file.read()
    key = _cache_key(image_bytes, preset, coverage)
    if key in _cache:
        return Response(content=_cache[key], media_type="image/png")

    watermarked = _add_watermark(image_bytes)
    _cache[key] = watermarked
    return Response(content=watermarked, media_type="image/png")

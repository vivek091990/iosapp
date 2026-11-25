from __future__ import annotations

import hashlib
import io
import os
import time
from collections import OrderedDict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from hairline import PreviewError, run_preview
from hairline.prepare import load_photo, to_png_bytes
from hairline.providers import ProviderError, ProviderNotConfigured, configured_providers, get_provider

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE_TTL_S = 15 * 60
CACHE_MAX = 64
WATERMARK = os.environ.get("HAIRLINE_WATERMARK", "1") != "0"

# key -> (expires_at, png bytes). Photos are kept in memory only, briefly, never on disk.
_cache: "OrderedDict[str, tuple[float, bytes]]" = OrderedDict()


def _coverage_bucket(value: int) -> int:
    return max(0, min(100, value)) // 10


def _cache_key(image_bytes: bytes, preset: str, coverage: int, provider: str = "") -> str:
    digest = hashlib.sha256()
    digest.update(image_bytes)
    digest.update(preset.encode("utf-8"))
    digest.update(str(_coverage_bucket(coverage)).encode("utf-8"))
    digest.update(provider.encode("utf-8"))
    return digest.hexdigest()


def _cache_get(key: str) -> bytes | None:
    hit = _cache.get(key)
    if hit is None:
        return None
    expires, data = hit
    if expires < time.monotonic():
        _cache.pop(key, None)
        return None
    _cache.move_to_end(key)
    return data


def _cache_put(key: str, data: bytes) -> None:
    _cache[key] = (time.monotonic() + CACHE_TTL_S, data)
    _cache.move_to_end(key)
    while len(_cache) > CACHE_MAX:
        _cache.popitem(last=False)


def _add_watermark(img: Image.Image) -> Image.Image:
    base = img.convert("RGBA")
    width, height = base.size
    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    text = "HairStyle Preview"
    font = ImageFont.load_default(size=max(12, width // 40))
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text((width - (right - left) - 16, height - (bottom - top) - 16), text, font=font, fill=(255, 255, 255, 160))
    return Image.alpha_composite(base, overlay).convert("RGB")


@app.get("/v1/health")
def health() -> dict:
    return {"ok": True, "providers": configured_providers()}


@app.post("/v1/preview")
async def preview(
    file: UploadFile = File(...),
    preset: str = Form(...),
    coverage: int = Form(...),
    model: str | None = Form(None),
) -> Response:
    image_bytes = await file.read()
    try:
        provider = get_provider(model)
    except ProviderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    key = _cache_key(image_bytes, preset, coverage, provider.name)
    cached = _cache_get(key)
    if cached is not None:
        return Response(content=cached, media_type="image/png")

    try:
        photo = load_photo(image_bytes)
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=422, detail="That file isn't a photo we can read.") from exc

    try:
        result = run_preview(photo, preset, coverage, provider)
    except ProviderNotConfigured as exc:
        raise HTTPException(status_code=503, detail=f"Preview model not configured: {exc}") from exc
    except PreviewError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc

    out = _add_watermark(result.image) if WATERMARK else result.image
    png = to_png_bytes(out)
    _cache_put(key, png)
    return Response(
        content=png,
        media_type="image/png",
        headers={
            "X-Hairline-Provider": result.provider,
            "X-Hairline-Seconds": f"{result.seconds:.2f}",
            "X-Hairline-Attempts": str(result.attempts),
        },
    )

from __future__ import annotations

import io
import math

from PIL import Image, ImageOps

MAX_PIXELS = 1_500_000


def load_photo(image_bytes: bytes, max_pixels: int = MAX_PIXELS) -> Image.Image:
    """Decode, apply EXIF orientation, convert to RGB and cap the size."""
    with Image.open(io.BytesIO(image_bytes)) as raw:
        img = ImageOps.exif_transpose(raw).convert("RGB")
    w, h = img.size
    if w * h > max_pixels:
        scale = math.sqrt(max_pixels / (w * h))
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return img


def to_png_bytes(img: Image.Image) -> bytes:
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def to_jpeg_bytes(img: Image.Image, quality: int = 92) -> bytes:
    out = io.BytesIO()
    img.convert("RGB").save(out, format="JPEG", quality=quality)
    return out.getvalue()

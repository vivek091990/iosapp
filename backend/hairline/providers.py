"""Image-edit model providers. Each takes (image, prompt) and returns an edited image.

Every provider reads its key from the environment, so the same code can race them in the
bake-off and the endpoint can switch with HAIRLINE_PROVIDER.
"""
from __future__ import annotations

import base64
import io
import os
import time
from dataclasses import dataclass
from typing import Callable, Protocol

import httpx
from PIL import Image

from .prepare import to_jpeg_bytes


class ProviderError(RuntimeError):
    pass


class ProviderNotConfigured(ProviderError):
    pass


class Provider(Protocol):
    name: str

    def edit(self, image: Image.Image, prompt: str) -> Image.Image: ...


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ProviderNotConfigured(f"{name} is not set")
    return value


def _data_uri(img: Image.Image) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(to_jpeg_bytes(img)).decode()


def _download(url: str) -> Image.Image:
    resp = httpx.get(url, timeout=60)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


@dataclass
class GeminiProvider:
    """Google Nano Banana 2 (gemini-3.1-flash-image) or Nano Banana Pro (gemini-3-pro-image)."""

    name: str
    model: str

    def edit(self, image: Image.Image, prompt: str) -> Image.Image:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=_env("GEMINI_API_KEY"))
        resp = client.models.generate_content(
            model=self.model,
            contents=[prompt, image],
            config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
        )
        for part in resp.parts or []:
            if part.inline_data is not None and part.inline_data.data:
                return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
        raise ProviderError(f"{self.model} returned no image")


@dataclass
class OpenAIProvider:
    """OpenAI GPT Image (images.edit)."""

    name: str
    model: str

    def edit(self, image: Image.Image, prompt: str) -> Image.Image:
        from openai import OpenAI

        client = OpenAI(api_key=_env("OPENAI_API_KEY"))
        buf = io.BytesIO(to_jpeg_bytes(image))
        buf.name = "photo.jpg"
        try:
            resp = client.images.edit(model=self.model, image=buf, prompt=prompt, input_fidelity="high")
        except Exception as exc:  # some GPT Image versions reject input_fidelity
            if "input_fidelity" not in str(exc):
                raise
            buf.seek(0)
            resp = client.images.edit(model=self.model, image=buf, prompt=prompt)
        item = resp.data[0]
        if getattr(item, "b64_json", None):
            return Image.open(io.BytesIO(base64.b64decode(item.b64_json))).convert("RGB")
        if getattr(item, "url", None):
            return _download(item.url)
        raise ProviderError(f"{self.model} returned no image")


@dataclass
class BFLProvider:
    """Black Forest Labs API: FLUX.1 Kontext and FLUX.2 (async submit + poll)."""

    name: str
    endpoint: str  # e.g. "flux-kontext-pro", "flux-2-pro"
    timeout_s: float = 90

    def edit(self, image: Image.Image, prompt: str) -> Image.Image:
        key = _env("BFL_API_KEY")
        headers = {"x-key": key, "accept": "application/json"}
        body = {"prompt": prompt, "input_image": base64.b64encode(to_jpeg_bytes(image)).decode(), "output_format": "png"}
        with httpx.Client(timeout=30) as http:
            sub = http.post(f"https://api.bfl.ai/v1/{self.endpoint}", headers=headers, json=body)
            sub.raise_for_status()
            job = sub.json()
            poll_url = job.get("polling_url") or f"https://api.bfl.ai/v1/get_result?id={job['id']}"
            deadline = time.monotonic() + self.timeout_s
            while time.monotonic() < deadline:
                time.sleep(0.75)
                res = http.get(poll_url, headers=headers).json()
                status = res.get("status")
                if status == "Ready":
                    return _download(res["result"]["sample"])
                if status not in ("Pending", "Processing", "Queued", None):
                    raise ProviderError(f"{self.endpoint}: {status}")
        raise ProviderError(f"{self.endpoint}: timed out")


@dataclass
class FalProvider:
    """fal.ai synchronous endpoint, used here for open-weight Qwen-Image-Edit."""

    name: str
    endpoint: str  # e.g. "fal-ai/qwen-image-edit-2511"

    def edit(self, image: Image.Image, prompt: str) -> Image.Image:
        key = _env("FAL_KEY")
        resp = httpx.post(
            f"https://fal.run/{self.endpoint}",
            headers={"Authorization": f"Key {key}"},
            json={"prompt": prompt, "image_urls": [_data_uri(image)], "num_images": 1},
            timeout=120,
        )
        resp.raise_for_status()
        images = resp.json().get("images") or []
        if not images:
            raise ProviderError(f"{self.endpoint} returned no image")
        url = images[0]["url"]
        if url.startswith("data:"):
            return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
        return _download(url)


# Model ids can be overridden without a code change, since vendors rename preview models.
PROVIDERS: dict[str, Callable[[], Provider]] = {
    "nano-banana-2": lambda: GeminiProvider("nano-banana-2", os.environ.get("GEMINI_FLASH_IMAGE_MODEL", "gemini-3.1-flash-image")),
    "nano-banana-2-lite": lambda: GeminiProvider("nano-banana-2-lite", os.environ.get("GEMINI_LITE_IMAGE_MODEL", "gemini-3.1-flash-lite-image")),
    "nano-banana-pro": lambda: GeminiProvider("nano-banana-pro", os.environ.get("GEMINI_PRO_IMAGE_MODEL", "gemini-3-pro-image")),
    "gpt-image": lambda: OpenAIProvider("gpt-image", os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-2")),
    "flux-kontext": lambda: BFLProvider("flux-kontext", os.environ.get("BFL_KONTEXT_ENDPOINT", "flux-kontext-pro")),
    "flux-2": lambda: BFLProvider("flux-2", os.environ.get("BFL_FLUX2_ENDPOINT", "flux-2-pro")),
    "qwen-edit": lambda: FalProvider("qwen-edit", os.environ.get("FAL_QWEN_ENDPOINT", "fal-ai/qwen-image-edit-2511")),
}

DEFAULT_PROVIDER = "nano-banana-pro"  # picked by Vivek after the first real run: keeps existing hair, fills gaps


def get_provider(name: str | None = None) -> Provider:
    key = (name or os.environ.get("HAIRLINE_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if key not in PROVIDERS:
        raise ProviderError(f"unknown provider {key!r}; choose from {', '.join(PROVIDERS)}")
    return PROVIDERS[key]()


def configured_providers() -> list[str]:
    needs = {
        "nano-banana-2": "GEMINI_API_KEY",
        "nano-banana-2-lite": "GEMINI_API_KEY",
        "nano-banana-pro": "GEMINI_API_KEY",
        "gpt-image": "OPENAI_API_KEY",
        "flux-kontext": "BFL_API_KEY",
        "flux-2": "BFL_API_KEY",
        "qwen-edit": "FAL_KEY",
    }
    return [name for name, env in needs.items() if os.environ.get(env)]

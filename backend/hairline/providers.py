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


@dataclass
class MfluxProvider:
    """Open-weight FLUX.2 klein running locally on Apple Silicon through mflux (MLX).

    No API key and no per-image cost; needs `mflux` installed on the machine running the
    backend (`uv tool install mflux`). Set MFLUX_MODEL to a local or Hub model path, e.g. the
    4-bit `Runpod/FLUX.2-klein-4B-mflux-4bit`, which fits a 16 GB Mac.
    """

    name: str
    model: str
    steps: int = 4
    timeout_s: float = 600

    def edit(self, image: Image.Image, prompt: str) -> Image.Image:
        import shutil
        import subprocess
        import tempfile

        exe = os.environ.get("MFLUX_EDIT_BIN") or shutil.which("mflux-generate-flux2-edit")
        if not exe:
            raise ProviderNotConfigured("mflux is not installed (mflux-generate-flux2-edit not found)")
        # FLUX works in 16 px blocks; keep the aspect ratio with the long side at 1024.
        scale = 1024 / max(image.size)
        w, h = (max(16, int(round(d * scale / 16)) * 16) for d in image.size)
        with tempfile.TemporaryDirectory() as tmp:
            src, out = os.path.join(tmp, "in.jpg"), os.path.join(tmp, "out.png")
            image.convert("RGB").resize((w, h), Image.LANCZOS).save(src, quality=95)
            cmd = [exe, "--model", self.model, "--image-paths", src, "--prompt", prompt,
                   "--steps", str(self.steps), "--guidance", "1.0", "--seed", "42",
                   "--width", str(w), "--height", str(h), "--output", out]
            if os.environ.get("MFLUX_LOW_RAM", "1") != "0":
                cmd.append("--low-ram")
            done = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_s)
            if done.returncode != 0 or not os.path.exists(out):
                raise ProviderError(f"mflux failed: {(done.stderr or done.stdout)[-400:]}")
            return Image.open(out).convert("RGB")


# Model ids can be overridden without a code change, since vendors rename preview models.
PROVIDERS: dict[str, Callable[[], Provider]] = {
    "nano-banana-2": lambda: GeminiProvider("nano-banana-2", os.environ.get("GEMINI_FLASH_IMAGE_MODEL", "gemini-3.1-flash-image")),
    "nano-banana-2-lite": lambda: GeminiProvider("nano-banana-2-lite", os.environ.get("GEMINI_LITE_IMAGE_MODEL", "gemini-3.1-flash-lite-image")),
    "nano-banana-pro": lambda: GeminiProvider("nano-banana-pro", os.environ.get("GEMINI_PRO_IMAGE_MODEL", "gemini-3-pro-image")),
    "gpt-image": lambda: OpenAIProvider("gpt-image", os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-2")),
    "flux-kontext": lambda: BFLProvider("flux-kontext", os.environ.get("BFL_KONTEXT_ENDPOINT", "flux-kontext-pro")),
    "flux-2": lambda: BFLProvider("flux-2", os.environ.get("BFL_FLUX2_ENDPOINT", "flux-2-pro")),
    "qwen-edit": lambda: FalProvider("qwen-edit", os.environ.get("FAL_QWEN_ENDPOINT", "fal-ai/qwen-image-edit-2511")),
    "local-klein": lambda: MfluxProvider("local-klein", os.environ.get("MFLUX_MODEL", "Runpod/FLUX.2-klein-4B-mflux-4bit")),
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
    found = [name for name, env in needs.items() if os.environ.get(env)]
    import shutil

    if os.environ.get("MFLUX_EDIT_BIN") or shutil.which("mflux-generate-flux2-edit"):
        found.append("local-klein")
    return found

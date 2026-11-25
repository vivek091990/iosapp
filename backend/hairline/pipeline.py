from __future__ import annotations

import time
from dataclasses import dataclass, field

from PIL import Image

from . import composite as comp
from .face import detect_faces
from .mask import hair_region_mask
from .prompts import build_prompt
from .providers import Provider, ProviderNotConfigured
from .quality import photo_problem
from .segment import category_mask
from .tone import match_hair_tone

# Stable face points may drift by at most this fraction of face width after alignment.
# Beyond it the model changed the face itself (pose, shape), so we retry or fail.
MAX_FACE_DRIFT = 0.035

PRESETS = {"hairline", "crown", "part", "auto"}


class PreviewError(Exception):
    """A problem with the user's photo; message is safe to show in the app."""

    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


@dataclass
class PipelineResult:
    image: Image.Image
    provider: str
    seconds: float
    attempts: int
    face_drift: float | None
    masked: bool
    preset: str = ""
    tone_shift: float = 0.0
    notes: list[str] = field(default_factory=list)


def _call(provider: Provider, photo: Image.Image, prompt: str) -> Image.Image:
    try:
        return provider.edit(photo, prompt)
    except ProviderNotConfigured:
        raise
    except Exception as exc:
        raise PreviewError(f"The preview service failed: {type(exc).__name__}: {exc}", status=502) from exc


def run_preview(photo: Image.Image, preset: str, coverage: int, provider: Provider, max_attempts: int = 2) -> PipelineResult:
    started = time.monotonic()
    preset_key = preset.strip().lower()
    if preset_key not in PRESETS:
        raise PreviewError(f"Unknown preset {preset!r}.")

    faces = detect_faces(photo)
    notes: list[str] = []
    if len(faces) > 1:
        raise PreviewError("Please use a photo with just one face.")

    if preset_key == "auto":
        # A front-facing photo means hairline; no face usually means a top-down crown shot.
        preset_key = "hairline" if faces else "crown"
        notes.append(f"auto preset -> {preset_key}")
    prompt = build_prompt(preset_key, coverage)

    if not faces:
        if preset_key != "crown":
            raise PreviewError("We couldn't find a face. Use a clear, front-facing photo with your hairline visible.")
        # Top-down crown shots often show no face: edit the whole photo, nothing to protect.
        notes.append("no face found; crown edit applied to whole photo")
        edited = _call(provider, photo, prompt).resize(photo.size, Image.LANCZOS)
        return PipelineResult(edited, provider.name, time.monotonic() - started, 1, None, False, preset_key, 0.0, notes)

    face = faces[0]
    problem = photo_problem(photo, face)
    if problem:
        raise PreviewError(problem)

    orig_cats = category_mask(photo)
    mask = hair_region_mask(photo.size, face, preset_key, orig_cats)

    def finish(aligned_img: Image.Image, attempt: int, drift: float) -> PipelineResult:
        toned, shift = match_hair_tone(photo, aligned_img, orig_cats, category_mask(aligned_img), mask)
        out = comp.composite(photo, toned, mask)
        return PipelineResult(out, provider.name, time.monotonic() - started, attempt, drift, True, preset_key, shift, notes)

    last_error: str | None = None
    best: tuple[float, Image.Image] | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            edited = _call(provider, photo, prompt)
        except PreviewError as exc:  # network, safety refusal, bad response: try again
            last_error = str(exc)
            notes.append(f"attempt {attempt}: {last_error}")
            continue
        scaled = edited.resize(photo.size, Image.LANCZOS) if edited.size != photo.size else edited
        edited_faces = detect_faces(scaled)
        if len(edited_faces) != 1:
            last_error = "model output lost the face"
            notes.append(f"attempt {attempt}: {last_error}")
            continue
        aligned = comp.align_to_original(scaled, face, edited_faces[0], photo.size)
        if best is None or aligned.residual < best[0]:
            best = (aligned.residual, aligned.image)
        if aligned.residual <= MAX_FACE_DRIFT:
            return finish(aligned.image, attempt, aligned.residual)
        notes.append(f"attempt {attempt}: face drift {aligned.residual:.3f} over limit")

    if best is None:
        raise PreviewError(last_error or "The preview service failed.", status=502)
    # Every attempt moved the face too much; the composite still keeps the face original,
    # so return the closest one and flag it.
    notes.append("returned best attempt above drift limit")
    return finish(best[1], max_attempts, best[0])

from __future__ import annotations

import time
from dataclasses import dataclass, field

from PIL import Image

from . import composite as comp
from .face import detect_faces
from .mask import hair_region_mask
from .prompts import build_prompt
from .segment import category_mask
from .providers import Provider, ProviderNotConfigured

# Stable face points may drift by at most this fraction of face width after alignment.
# Beyond it the model changed the face itself (pose, shape), so we retry or fail.
MAX_FACE_DRIFT = 0.035


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
    notes: list[str] = field(default_factory=list)


def run_preview(photo: Image.Image, preset: str, coverage: int, provider: Provider, max_attempts: int = 2) -> PipelineResult:
    started = time.monotonic()
    preset_key = preset.strip().lower()
    try:
        prompt = build_prompt(preset_key, coverage)
    except ValueError as exc:
        raise PreviewError(str(exc)) from exc

    faces = detect_faces(photo)
    notes: list[str] = []
    if len(faces) > 1:
        raise PreviewError("Please use a photo with just one face.")
    if not faces:
        if preset_key != "crown":
            raise PreviewError("We couldn't find a face. Use a clear, front-facing photo with your hairline visible.")
        # Top-down crown shots often show no face: edit the whole photo, nothing to protect.
        notes.append("no face found; crown edit applied to whole photo")
        try:
            edited = provider.edit(photo, prompt).resize(photo.size, Image.LANCZOS)
        except ProviderNotConfigured:
            raise
        except Exception as exc:
            raise PreviewError(f"The preview service failed: {exc}", status=502) from exc
        return PipelineResult(edited, provider.name, time.monotonic() - started, 1, None, False, notes)

    face = faces[0]
    mask = hair_region_mask(photo.size, face, preset_key, category_mask(photo))

    last_error: str | None = None
    best: tuple[float, Image.Image] | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            edited = provider.edit(photo, prompt)
        except ProviderNotConfigured:
            raise
        except Exception as exc:  # network, safety refusal, bad response
            last_error = f"{type(exc).__name__}: {exc}"
            notes.append(f"attempt {attempt}: {last_error}")
            continue
        edited_faces = detect_faces(edited)
        if len(edited_faces) != 1:
            last_error = "model output lost the face"
            notes.append(f"attempt {attempt}: {last_error}")
            continue
        scaled = edited.resize(photo.size, Image.LANCZOS)
        scaled_face = detect_faces(scaled)[0] if edited.size != photo.size else edited_faces[0]
        aligned = comp.align_to_original(scaled, face, scaled_face, photo.size)
        if best is None or aligned.residual < best[0]:
            best = (aligned.residual, aligned.image)
        if aligned.residual <= MAX_FACE_DRIFT:
            out = comp.composite(photo, aligned.image, mask)
            return PipelineResult(out, provider.name, time.monotonic() - started, attempt, aligned.residual, True, notes)
        notes.append(f"attempt {attempt}: face drift {aligned.residual:.3f} over limit")

    if best is None:
        raise PreviewError(f"The preview service failed: {last_error}", status=502)
    # Every attempt moved the face too much; the composite still keeps the face original,
    # so return the closest one and flag it.
    notes.append("returned best attempt above drift limit")
    out = comp.composite(photo, best[1], mask)
    return PipelineResult(out, provider.name, time.monotonic() - started, max_attempts, best[0], True, notes)

from __future__ import annotations

_KEEP = (
    "Keep everything else exactly the same: the same person, face, eyes, eyebrows, skin, "
    "expression, head pose, framing, lighting, clothing and background. "
    "Match the existing hair color, texture, curl and length so the new hair looks like it "
    "has always been there. Photorealistic, no retouching of the face."
)

_PRESET = {
    "hairline": "Fill in the receding hairline and temples",
    "crown": "Fill in the thinning crown so the scalp at the top of the head is covered",
    "part": "Make the hair along the part denser so less scalp shows through",
}

_STRENGTH = [
    (33, "slightly, so the hair is only a little denser than now"),
    (66, "to a natural, age-appropriate hairline and density"),
    (101, "fully, to a strong, full head of hair with a lower hairline"),
]


def coverage_phrase(coverage: int) -> str:
    c = max(0, min(100, int(coverage)))
    for limit, phrase in _STRENGTH:
        if c < limit:
            return phrase
    return _STRENGTH[-1][1]


def build_prompt(preset: str, coverage: int) -> str:
    key = preset.strip().lower()
    if key not in _PRESET:
        raise ValueError(f"unknown preset {preset!r}")
    return f"{_PRESET[key]} {coverage_phrase(coverage)}. {_KEEP}"

"""Masked (RePaint-style) sampling for FLUX.2 klein in mflux.

mflux's klein edit command has no mask input. This wrapper adds one without changing the
installed package: after every denoising step, latents OUTSIDE the token mask are reset to
the reference image's latents noised to the current sigma, so only the masked area is
generated. Optionally (PRIOR > 0) the masked area is pulled once, on the first step, toward
the reference's pixels there (a hair-coloured prefill), which is what makes klein put hair
on bare skin instead of redrawing skin.

Run it with mflux's own Python (it is not part of the backend venv):

    ~/.local/share/uv/tools/mflux/bin/python scripts/klein_masked_sampler.py TOKMASK.npy [--prior 0.5] -- \
        --low-ram --model Runpod/FLUX.2-klein-4B-mflux-4bit --base-model flux2-klein-4b \
        --image-paths PREFILLED_CROP.png --prompt "..." --steps 4 --seed 42 \
        --width W --height H --output OUT.png

The first --image-paths entry must be the crop itself at exactly --width x --height: its
latents double as the "known" image, so no extra VAE encode is needed (an extra encode, and
letting a float32 sigma promote the latents, each pushed peak memory past 8 GB on a 16 GB Mac).
TOKMASK.npy is a (H/16, W/16) float array, 1 = generate.
klein_masked_inpaint.py builds all inputs and calls this script.
"""
from __future__ import annotations

import sys

import numpy as np

if "--" not in sys.argv:
    sys.exit(__doc__)
_own, _mflux = sys.argv[1 : sys.argv.index("--")], sys.argv[sys.argv.index("--") + 1 :]
TOKMASK = _own[0]
PRIOR = float(_own[_own.index("--prior") + 1]) if "--prior" in _own else 0.0
sys.argv = [sys.argv[0], *_mflux]

import mlx.core as mx  # noqa: E402
from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import (  # noqa: E402
    FlowMatchEulerDiscreteScheduler as Scheduler,
)
from mflux.models.flux2.variants.edit import flux2_klein_edit as klein_edit  # noqa: E402
from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers as Helpers  # noqa: E402

_state: dict = {}

_orig_reference = Helpers.prepare_reference_image_conditioning


def _reference(**kwargs):
    latents, ids = _orig_reference(**kwargs)
    _state["x0"] = latents  # first reference = the crop: our "known" latents
    return latents, ids


Helpers.prepare_reference_image_conditioning = staticmethod(_reference)

_orig_generate = klein_edit.Flux2KleinEdit.generate_image


def _generate(self, seed, prompt, num_inference_steps=4, height=1024, width=1024, **kwargs):
    eps, _, lat_h, lat_w = Helpers.prepare_generation_latents(seed=seed, height=height, width=width)
    m = np.load(TOKMASK).astype(np.float32)
    if m.shape != (lat_h, lat_w):
        raise SystemExit(f"token mask is {m.shape}, expected {(lat_h, lat_w)} for {width}x{height}")
    _state.update(eps=eps, m=mx.array(m.reshape(1, -1, 1)), n=lat_h * lat_w)
    print(f"masked sampling: {int(m.sum())}/{m.size} tokens generated, prior {PRIOR}")
    return _orig_generate(
        self, seed=seed, prompt=prompt, num_inference_steps=num_inference_steps, height=height, width=width, **kwargs
    )


klein_edit.Flux2KleinEdit.generate_image = _generate

_orig_step = Scheduler.step


def _step(self, noise, timestep, latents, sigmas, **kwargs):
    out = _orig_step(self, noise=noise, timestep=timestep, latents=latents, sigmas=sigmas, **kwargs)
    if "eps" not in _state:
        return out
    dt = out.dtype  # keep the model dtype; float32 here silently doubles memory
    x0 = _state["x0"][:, : _state["n"]].astype(dt)
    eps = _state["eps"].astype(dt)
    m = _state["m"].astype(dt)
    s = sigmas[timestep + 1].astype(dt)
    known = (1 - s) * x0 + s * eps
    if PRIOR > 0 and timestep == 0:
        out = out * (1 - PRIOR * m) + known * (PRIOR * m)
    return (m * out + (1 - m) * known).astype(dt)


Scheduler.step = _step

from mflux.models.flux2.cli.flux2_edit_generate import main  # noqa: E402

main()

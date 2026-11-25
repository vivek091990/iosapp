# HairStyle

HairStyle is a starter project with a SwiftUI iOS app and a FastAPI backend. The app sends a selected photo plus styling parameters to the backend and displays a watermarked preview.

## Project Structure

- `ios/` — SwiftUI iOS app (Xcode project).
- `backend/` — FastAPI backend.
- `README.md` — setup instructions.

## Backend (FastAPI)

The backend turns a photo into a one-tap hairline preview:

1. Fix orientation and cap the photo at about 1.5 MP.
2. Find the face (MediaPipe Face Landmarker) and the hair/forehead area (MediaPipe selfie segmenter).
3. Check the photo (face big enough, facing the camera, head not cut off, not too dark) and return a message the app can show if not.
4. Ask an image-edit model to fill in the hairline, crown or part (prompt per preset, strength from `coverage`). `Auto` picks hairline for a front-facing photo and crown when no face is visible.
5. Align the model's output to the original face and paste back only the hair region, so everything from the eyebrows down is the original photo, pixel for pixel.
6. Shift the new hair's color toward the person's existing hair, so the fill doesn't come out a shade off.
7. If the model moved the face too much, retry once; return the PNG and cache it for 15 minutes in memory.

### Requirements

- Python 3.10 or newer (google-genai does not install on 3.9; macOS ships 3.9, so use Homebrew or python.org Python)
- On Linux servers, MediaPipe needs `libegl1` and `libgles2` (`apt-get install libegl1 libgles2`).
- At least one model API key (see below). Gemini image models have no free-tier quota: the Google AI Studio project needs billing turned on, or every call returns 429. MediaPipe model files download to `backend/models/` on first use.

### Install

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

### Models and keys

Pick the default with `HAIRLINE_PROVIDER` (default `nano-banana-pro`), or per request with an optional `model` form field.

| `model` | Model | Key |
| --- | --- | --- |
| `nano-banana-2` | Google Gemini 3.1 Flash Image (faster, cheaper) | `GEMINI_API_KEY` |
| `nano-banana-2-lite` | Google Gemini 3.1 Flash Lite Image (cheaper) | `GEMINI_API_KEY` |
| `nano-banana-pro` | Google Gemini 3 Pro Image (default: best at keeping existing hair) | `GEMINI_API_KEY` |
| `gpt-image` | OpenAI GPT Image | `OPENAI_API_KEY` |
| `flux-kontext` | Black Forest Labs FLUX.1 Kontext pro | `BFL_API_KEY` |
| `flux-2` | Black Forest Labs FLUX.2 pro | `BFL_API_KEY` |
| `qwen-edit` | Qwen-Image-Edit (open weights) via fal.ai | `FAL_KEY` |
| `local-klein` | FLUX.2 klein 4B (open weights, Apache 2.0) running on this Mac via mflux | none; `uv tool install mflux` (Apple Silicon) |

Model ids can be overridden with `GEMINI_FLASH_IMAGE_MODEL`, `GEMINI_LITE_IMAGE_MODEL`, `GEMINI_PRO_IMAGE_MODEL`, `OPENAI_IMAGE_MODEL`, `BFL_KONTEXT_ENDPOINT`, `BFL_FLUX2_ENDPOINT` and `FAL_QWEN_ENDPOINT`; `MFLUX_MODEL` picks the local model (default `Runpod/FLUX.2-klein-4B-mflux-4bit`, 4-bit, fits a 16 GB Mac). Set `HAIRLINE_WATERMARK=0` to turn off the preview watermark.

### Run

```bash
export GEMINI_API_KEY=...
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

`POST /v1/preview` takes multipart form data (`file`, `preset` = Auto/Hairline/Crown/Part (default Auto), `coverage` 0-100 (default 50), optional `model`) and returns a PNG. `GET /v1/health` lists which models have keys set. Photo problems (no face, several faces) return 422 with a message the app can show.

### Bake-off: make the models compete

Put test photos in a folder and run every model that has a key:

```bash
python scripts/bakeoff.py ~/hairline-photos --blind
```

It writes `bakeoff-out/<time>/index.html` with the original and each model's result side by side, a summary per model (success rate, median seconds, face drift) and 1 to 5 star ratings you can export. `--presets` defaults to `auto`. `--blind` hides model names (the key is in `results.json`) so people can rate realism without bias.

### Training data for a self-hosted model

`backend/training/` turns consented photos of people with full hairlines into training pairs for fine-tuning an open edit model (Qwen-Image-Edit or FLUX.2 klein). Pairs are made in reverse: each real photo is the "after", and a synthetic receded copy is the "before", so the model always learns toward real hair. The receding is classical image processing (M-shaped, deeper at the temples, thinned edge, forehead skin continued upward); no Gemini or other generative output is used.

```bash
python -m training.build_dataset ~/Documents/hairline-training out/dataset --variants 2
```

Put one subfolder per person (`person_001/`, ...) so the train/test split never puts the same person in both. The output uses the control / target / caption layout LoRA trainers read, plus `manifest.jsonl`, `skipped.csv` (photos that can't be used and why: no visible hairline, bangs over the forehead, too little hair) and `review.html` to eyeball pairs before training. It works best on short-to-medium hair; long hair swept over the forehead gives weaker pairs, so drop those after review.

### Tests

```bash
pytest -q
```

## iOS App (SwiftUI)

### Open in Xcode

1. Open `ios/HairStyle/HairStyle.xcodeproj` in Xcode.
2. Select the `HairStyle` scheme.
3. Build and run.

### Configure the Backend URL

Update the base URL in `ios/HairStyle/HairStyle/Config.swift` if your backend is running on a different host or port.

### Run on Simulator

1. Select an iPhone simulator from the device list.
2. Click **Run** to launch.
3. Use the Photos picker to choose an image (you may need to add images to the simulator).

### Run on a Physical iPhone

1. Connect your device via USB.
2. In Xcode, select your device as the run target.
3. Ensure you have a valid signing team configured in the project settings.
4. Build and run, then allow photo permissions when prompted.

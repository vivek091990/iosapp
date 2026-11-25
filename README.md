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
3. Ask an image-edit model to fill in the hairline, crown or part (prompt per preset, strength from `coverage`).
4. Align the model's output to the original face and paste back only the hair region, so everything from the eyebrows down is the original photo, pixel for pixel.
5. If the model moved the face too much, retry once; return the PNG and cache it for 15 minutes in memory.

### Requirements

- Python 3.10+
- On Linux servers, MediaPipe needs `libegl1` and `libgles2` (`apt-get install libegl1 libgles2`).
- At least one model API key (see below). MediaPipe model files download to `backend/models/` on first use.

### Install

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

### Models and keys

Pick the default with `HAIRLINE_PROVIDER` (default `nano-banana-2`), or per request with an optional `model` form field.

| `model` | Model | Key |
| --- | --- | --- |
| `nano-banana-2` | Google Gemini 3.1 Flash Image | `GEMINI_API_KEY` |
| `nano-banana-pro` | Google Gemini 3 Pro Image | `GEMINI_API_KEY` |
| `gpt-image` | OpenAI GPT Image | `OPENAI_API_KEY` |
| `flux-kontext` | Black Forest Labs FLUX.1 Kontext pro | `BFL_API_KEY` |
| `flux-2` | Black Forest Labs FLUX.2 pro | `BFL_API_KEY` |
| `qwen-edit` | Qwen-Image-Edit (open weights) via fal.ai | `FAL_KEY` |

Model ids can be overridden with `GEMINI_FLASH_IMAGE_MODEL`, `GEMINI_PRO_IMAGE_MODEL`, `OPENAI_IMAGE_MODEL`, `BFL_KONTEXT_ENDPOINT`, `BFL_FLUX2_ENDPOINT` and `FAL_QWEN_ENDPOINT`. Set `HAIRLINE_WATERMARK=0` to turn off the preview watermark.

### Run

```bash
export GEMINI_API_KEY=...
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

`POST /v1/preview` takes multipart form data (`file`, `preset` = Hairline/Crown/Part, `coverage` 0-100, optional `model`) and returns a PNG. `GET /v1/health` lists which models have keys set. Photo problems (no face, several faces) return 422 with a message the app can show.

### Bake-off: make the models compete

Put test photos in a folder and run every model that has a key:

```bash
python scripts/bakeoff.py ~/hairline-photos --presets hairline,crown,part --blind
```

It writes `bakeoff-out/<time>/index.html` with the original and each model's result side by side, plus seconds and face drift per result. `--blind` hides model names (the key is in `results.json`) so people can rate realism without bias.

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

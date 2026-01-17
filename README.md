# HairStyle

HairStyle is a starter project with a SwiftUI iOS app and a FastAPI backend. The app sends a selected photo plus styling parameters to the backend and displays a watermarked preview.

## Project Structure

- `ios/` — SwiftUI iOS app (Xcode project).
- `backend/` — FastAPI backend.
- `README.md` — setup instructions.

## Backend (FastAPI)

### Requirements

- Python 3.10+

### Install

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The preview endpoint is available at `http://localhost:8000/v1/preview` and expects multipart form data.

## iOS App (SwiftUI)

### Open in Xcode

1. Open `ios/HairStyle.xcodeproj` in Xcode.
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

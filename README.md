# Automatic Camera Controller

A Windows desktop prototype for an automatic presenter camera.

The project currently supports:

- A wide camera source from an Android IP Webcam stream, laptop webcam, or video file
- A virtual camera crop controlled with keyboard/buttons
- Single-click object selection
- Local object tracking with OpenCV
- Optional local human tracking with OpenCV face detection
- Manual control as a permanent fallback
- A native-style Tkinter/ttk Windows interface
- A compact Windows console layout with adaptive, high-resolution video panels
- A camera/focus app icon for the window and packaged executable
- Smooth pan/tilt motion and zoom
- Independent Gemini and Groq analysis with staggered requests
- Gemini image analysis through the official `google-genai` Python SDK
- Per-model status, model name, confidence, and colored detection boxes
- A live diagnostics terminal in the app and in PowerShell/CMD

## Project files

| File | Purpose |
| --- | --- |
| `windows_camera_app.py` | Main Windows desktop interface |
| `target_tracking.py` | Session-only references, appearance matching, and local face signals |
| `vision_providers.py` | Gemini and Groq vision adapters |
| `virtual_camera_tracking.py` | Command-line webcam/stream tracker |
| `virtual_camera.py` | Original keyboard-controlled simulator |
| `AI_PROVIDER_DESIGN.md` | Gemini/Groq architecture for automatic selection |
| `requirements.txt` | Python dependencies |

## Windows setup

Open the project folder in VS Code and activate the project environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

To build a standalone Windows executable, run this from PowerShell in the
project folder:

```powershell
.\setup_windows_app.ps1
```

The script creates or reuses `.venv`, installs the project dependencies and
PyInstaller, and writes `dist\AutomaticCamera.exe`. The executable can be
copied to another Windows machine; API keys still need to be configured there
as environment variables. The first build requires an internet connection to
install Python packages.

The script also regenerates the multi-size icon in `assets\automatic-camera.ico`
before packaging. The icon is intentionally simple so the camera and tracking
reticle remain readable in the Windows taskbar and title bar.

## Run the desktop application

```powershell
python windows_camera_app.py
```

Choose a source in the application:

- **Android stream:** `http://PHONE_IP:8080/video`
- **Webcam:** camera index `0`, `1`, or another available index
- **Video file:** select a local video with Browse

In the Source Camera panel, click an object once. The local tracker box is
yellow and updates from the tracker on each decoded frame; the green crop is
the virtual camera output and intentionally follows with smoothing. Tracking
uses CSRT on an aspect-preserved image reduced to a 640-pixel long edge, then
maps boxes back to source resolution. If tracking is lost, local template
matching searches for the saved target every 0.2 seconds. In Automatic AI
mode, Gemini detections are light blue and Groq detections are light brown.
When no target has been selected manually, the Tracking panel's AI target
priority list is evaluated from first to last on every request. A manual
selection always overrides that list.
The model marked `[CAMERA]` currently controls the crop.

To export the virtual camera over the network, enter a port in the **Network
export** section and click **Start network export**. Open the displayed URL in
a browser to see the feed-only page and use its fullscreen button. The
`/video.mjpg` endpoint provides the MJPEG stream directly, while
`/coordinates` returns a current JSON snapshot containing the virtual-camera
crop, normalized center coordinates, and motor-friendly `pan` and `tilt` values
from `-1` to `1`. For a continuously pushed feed, use `/coordinates/stream`,
which sends one JSON object per updated frame. The server listens on all network interfaces, so another device can use
the computer's LAN IP instead of `localhost`.

Enable **Track human (local face detection)** in the Tracking panel when the
selected target is a person. Bundled OpenCV Haar cascades detect faces, eyes,
and mouth/smile regions locally; these cues support appearance matching and
are shown on the source image. They are not face recognition: no identity is
inferred or stored. A temporarily hidden face or missing feature does not
automatically clear or drop the selected person's tracker.

Each live AI request sends one aspect-preserved JPEG still plus a text prompt,
not a video stream. When a target reference exists, the session-only reference
crop accompanies that still. Without a selected target, providers retain the
original presenter-detection behavior. Structured responses are validated
locally before they can move the camera, and their label is shown with the
analyzed frame number.

At launch, background diagnostics probe webcam indexes 0-4, the configured
Android stream, and the local OpenCV tracker and face detector. Camera probes
read locally and release each feed. When API keys are configured, Gemini and
Groq each receive a text check and a synthetic calibration image for a short
description; these checks do not send camera footage, but they do make API
requests and may use provider quota. With Automatic AI enabled, each model's
visual label is shown alongside its detection box.

## Current operating modes

- **Manual control:** keyboard, arrow buttons, zoom, and reset
- **Assisted tracking:** click an object and let the local tracker follow it
- **Automatic AI:** Gemini and Groq can run in parallel. Each model analyzes
  one aspect-preserved JPEG image every five seconds, with the two requests
  staggered by 2.5 seconds. If local tracking loses its target, Automatic AI
  activates immediately and checks every two seconds for up to ten seconds
  before returning to the normal cadence. Provider cooldowns still apply to
  provider errors, and the local tracker handles frames between corrections.

Manual mode remains available if providers are unavailable. A failure from
one cloud provider no longer disables the other provider or turns Automatic
AI off.

## Safety and privacy

Do not commit API keys, `.env` files, camera footage, or virtual environments.
Face detection runs locally and uses only the current in-memory frame. No face
images, biometric identities, or references are persisted. Startup AI checks
use a generated calibration image; live camera frames are sent to cloud AI
only after Automatic AI is enabled. Local tracking and manual control remain
available without cloud providers. Live camera frames are sent only while
Automatic AI is enabled, including its automatic tracking-loss fallback.

## Automatic AI configuration

Automatic AI reads keys from local Windows environment variables. Never put them in this repository:

```powershell
$env:GEMINI_API_KEY = "your Gemini key"
$env:GROQ_API_KEY = "your Groq key"
```

Optional model overrides are supported through `GEMINI_MODEL` and `GROQ_MODEL`. The Tracking panel shows each configured model as OFF, WAITING, ANALYZING, TRACKING, NO TARGET, or ERROR.

Use the **Enable AI** button in the Tracking panel to start cloud detection.
The LED and per-provider indicators show whether the workers are active. Every
captured video second, AI request, result, error, bounding box, confidence,
arbitration score, and camera target is written to the live diagnostics terminal
and stdout. The model with the best confidence after temporal/disagreement
error is calculated gets the `[CAMERA]` label and moves the virtual camera.
Disabling AI returns the app to local Assisted tracking and stops new cloud
requests.

Between cloud detections, the local OpenCV tracker keeps the camera movement smooth. Disabling AI returns the app to local Assisted tracking and stops new cloud requests.

Run the automated checks from the project folder with:

```powershell
python -m pytest -q --ignore=github
```

Before using a real camera, perform a Windows smoke test with a video file,
webcam, and Android stream where available. Confirm that selecting and clearing
a target works, the **Track human** status changes as faces appear/disappear,
and a temporarily hidden face does not force Manual mode.

The complete acceptance checklist is in
[`WINDOWS_SMOKE_TEST.md`](WINDOWS_SMOKE_TEST.md).

## Test the AI models from VS Code

Use the standalone test script to check the providers without starting the
Tkinter camera application. It sends one image to Gemini and Groq independently
and never prints API keys:

```powershell
python test_ai_models.py --image .\test-image.jpg
```

To test only one provider:

```powershell
python test_ai_models.py --image .\test-image.jpg --provider Gemini
python test_ai_models.py --image .\test-image.jpg --provider Groq
```

To capture one frame from a webcam instead:

```powershell
python test_ai_models.py --camera 0
```

The script prints the configured model, response time, detection result,
confidence, normalized bounding box, or the provider error. Set
`GEMINI_API_KEY` and `GROQ_API_KEY` in the same VS Code terminal before
running the test.
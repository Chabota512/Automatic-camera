"""
Keyboard-controlled virtual camera.

The source image/video represents a wide camera pointed at a stage.
The smaller rectangle is the virtual camera view. Arrow keys move it.

Run with a generated stage:
    python virtual_camera.py

Run with a video file:
    python virtual_camera.py path/to/wide_stage_video.mp4

Controls:
    Arrow keys  - move the virtual camera
    W/A/S/D     - alternative movement keys
    + or =      - zoom in
    -           - zoom out
    R           - reset camera position and zoom
    Q or Esc    - quit
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


# ========================= USER SETTINGS =========================
# How quickly the virtual camera approaches its target position.
# Smaller values are smoother/slower; larger values are faster.
# Recommended range: 0.05 to 0.40.
SMOOTHING_SPEED = 0.18


# OpenCV returns different arrow-key values depending on the operating system
# and window backend, so support the common values plus W/A/S/D.
KEY_LEFT = {81, 2424832}
KEY_UP = {82, 2490368}
KEY_RIGHT = {83, 2555904}
KEY_DOWN = {84, 2621440}


def make_demo_stage(width: int = 1280, height: int = 720) -> np.ndarray:
    """Create a simple stage image when no input video is supplied."""
    stage = np.zeros((height, width, 3), dtype=np.uint8)

    # Blue/purple stage lighting gradient.
    for y in range(height):
        brightness = int(28 + 42 * (1 - y / height))
        stage[y, :, :] = (brightness + 20, brightness // 2, brightness)

    # Back wall and floor.
    cv2.rectangle(stage, (0, 0), (width, int(height * 0.72)), (70, 35, 75), -1)
    cv2.rectangle(stage, (0, int(height * 0.72)), (width, height), (35, 35, 38), -1)

    # Stage lights.
    for x in range(100, width, 180):
        cv2.circle(stage, (x, 80), 22, (180, 180, 255), -1)
        cv2.circle(stage, (x, 80), 55, (80, 55, 100), 3)

    # Podium.
    podium_x = width // 2
    cv2.rectangle(
        stage,
        (podium_x - 95, int(height * 0.57)),
        (podium_x + 95, int(height * 0.86)),
        (65, 55, 50),
        -1,
    )
    cv2.rectangle(
        stage,
        (podium_x - 125, int(height * 0.54)),
        (podium_x + 125, int(height * 0.59)),
        (100, 85, 75),
        -1,
    )

    # A simple presenter so camera movement is easy to see.
    presenter_x = width // 2
    head_y = int(height * 0.29)
    cv2.circle(stage, (presenter_x, head_y), 38, (180, 155, 125), -1)
    cv2.ellipse(
        stage,
        (presenter_x, int(height * 0.51)),
        (80, 145),
        0,
        0,
        360,
        (55, 95, 170),
        -1,
    )
    cv2.line(
        stage,
        (presenter_x - 20, int(height * 0.74)),
        (presenter_x - 62, int(height * 0.88)),
        (180, 155, 125),
        14,
    )
    cv2.line(
        stage,
        (presenter_x + 20, int(height * 0.74)),
        (presenter_x + 62, int(height * 0.88)),
        (180, 155, 125),
        14,
    )
    cv2.putText(
        stage,
        "DEMO STAGE",
        (width // 2 - 130, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (220, 220, 235),
        2,
        cv2.LINE_AA,
    )
    return stage


def load_first_frame(video_path: str | None) -> tuple[np.ndarray, cv2.VideoCapture | None]:
    """Return the first source frame and an optional video capture object."""
    if video_path is None:
        return make_demo_stage(), None

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    capture = cv2.VideoCapture(str(path))
    ok, frame = capture.read()
    if not ok or frame is None:
        capture.release()
        raise RuntimeError(f"Could not read a frame from: {path}")
    return frame, capture


def draw_wide_view(
    frame: np.ndarray,
    camera_x: int,
    camera_y: int,
    crop_width: int,
    crop_height: int,
) -> np.ndarray:
    """Draw the virtual camera rectangle on a copy of the wide frame."""
    wide_view = frame.copy()
    height, width = wide_view.shape[:2]
    left = camera_x - crop_width // 2
    top = camera_y - crop_height // 2
    right = left + crop_width
    bottom = top + crop_height

    cv2.rectangle(wide_view, (left, top), (right, bottom), (0, 255, 0), 4)
    cv2.putText(
        wide_view,
        "WIDE CAMERA - green rectangle is virtual view",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return wide_view


def add_status(view: np.ndarray, camera_x: int, camera_y: int, zoom: float) -> np.ndarray:
    """Add controls and current camera state to the virtual-camera view."""
    output = view.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 76), (20, 20, 20), -1)
    cv2.putText(
        output,
        f"Virtual camera   x={camera_x}  y={camera_y}  zoom={zoom:.2f}x",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "Arrow keys/WASD: move   +/-: zoom   R: reset   Q/Esc: quit",
        (18, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return output


def run(video_path: str | None = None) -> None:
    first_frame, capture = load_first_frame(video_path)
    source_height, source_width = first_frame.shape[:2]

    # Keep a 16:9 virtual view while leaving room to pan around the source.
    aspect_ratio = 16 / 9
    base_crop_width = min(int(source_width * 0.60), source_width)
    base_crop_height = min(int(base_crop_width / aspect_ratio), source_height)
    base_crop_width = int(base_crop_height * aspect_ratio)

    center_x = source_width // 2
    center_y = source_height // 2
    camera_x = float(center_x)
    camera_y = float(center_y)
    target_x = float(center_x)
    target_y = float(center_y)
    zoom = 1.0
    movement_step = max(12, source_width // 25)

    cv2.namedWindow("Virtual Camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Wide Camera", cv2.WINDOW_NORMAL)

    while True:
        if capture is None:
            frame = first_frame.copy()
        else:
            ok, frame = capture.read()
            if not ok:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = capture.read()
            if not ok:
                break

        # A video can have dimensions different from its first frame.
        height, width = frame.shape[:2]
        crop_width = max(80, min(int(base_crop_width / zoom), width))
        crop_height = max(45, min(int(base_crop_height / zoom), height))
        crop_width = min(crop_width, int(crop_height * aspect_ratio))
        crop_height = min(crop_height, int(crop_width / aspect_ratio))

        # Keep the target fully inside the wide source frame, then move the
        # current camera position gradually toward it. This creates smooth
        # pan/tilt motion instead of an instant jump.
        target_x = max(crop_width // 2, min(target_x, width - crop_width // 2))
        target_y = max(crop_height // 2, min(target_y, height - crop_height // 2))
        camera_x += (target_x - camera_x) * SMOOTHING_SPEED
        camera_y += (target_y - camera_y) * SMOOTHING_SPEED
        camera_x_int = int(round(camera_x))
        camera_y_int = int(round(camera_y))

        left = int(camera_x_int - crop_width // 2)
        top = int(camera_y_int - crop_height // 2)
        virtual_view = frame[top : top + crop_height, left : left + crop_width]
        virtual_view = cv2.resize(virtual_view, (960, 540), interpolation=cv2.INTER_LINEAR)
        virtual_view = add_status(virtual_view, camera_x_int, camera_y_int, zoom)

        wide_view = draw_wide_view(
            frame,
            camera_x_int,
            camera_y_int,
            crop_width,
            crop_height,
        )
        wide_view = cv2.resize(wide_view, (960, 540), interpolation=cv2.INTER_AREA)

        cv2.imshow("Virtual Camera", virtual_view)
        cv2.imshow("Wide Camera", wide_view)

        key = cv2.waitKeyEx(30)
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in KEY_LEFT or key in (ord("a"), ord("A")):
            target_x -= movement_step
        elif key in KEY_RIGHT or key in (ord("d"), ord("D")):
            target_x += movement_step
        elif key in KEY_UP or key in (ord("w"), ord("W")):
            target_y -= movement_step
        elif key in KEY_DOWN or key in (ord("s"), ord("S")):
            target_y += movement_step
        elif key in (ord("+"), ord("=")):
            zoom = min(3.0, zoom + 0.1)
        elif key in (ord("-"), ord("_")):
            zoom = max(0.6, zoom - 0.1)
        elif key in (ord("r"), ord("R")):
            target_x = width // 2
            target_y = height // 2
            zoom = 1.0

    if capture is not None:
        capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    input_video = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        run(input_video)
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}")
        raise SystemExit(1)
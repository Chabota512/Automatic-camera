"""
Virtual camera with webcam/video input and click-to-select object tracking.

Run with the default webcam:
    python virtual_camera_tracking.py --camera

Run with a video file:
    python virtual_camera_tracking.py --video stage_video.mp4

Run with an Android IP Webcam stream:
    python virtual_camera_tracking.py --stream http://10.228.105.251:8080/video

Run with the generated demo stage:
    python virtual_camera_tracking.py

Controls:
    Arrow keys / W A S D - move the virtual camera manually
    + or =               - zoom in
    -                   - zoom out
    R                   - reset camera position and zoom
    T                   - clear the current target so a new object can be selected
    F                   - toggle automatic following of the selected object
    C                   - clear tracking and return to manual control
    Q or Esc            - quit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


# ========================= USER SETTINGS =========================
# How quickly the virtual camera approaches its target position.
# Smaller values are smoother/slower; larger values are faster.
# Recommended range: 0.05 to 0.40.
SMOOTHING_SPEED = 0.18

# When selecting with a single click, this temporary tracker box is centered
# on the cursor. Increase these values if the object is large in the frame.
CLICK_TRACK_BOX_WIDTH_RATIO = 0.22
CLICK_TRACK_BOX_HEIGHT_RATIO = 0.42
CLICK_MOVEMENT_THRESHOLD = 10

# =========================== KEY CODES ===========================
KEY_LEFT = {81, 2424832}
KEY_UP = {82, 2490368}
KEY_RIGHT = {83, 2555904}
KEY_DOWN = {84, 2621440}


def make_demo_stage(width: int = 1280, height: int = 720) -> np.ndarray:
    """Create a simple stage image when no source is supplied."""
    stage = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        brightness = int(28 + 42 * (1 - y / height))
        stage[y, :, :] = (brightness + 20, brightness // 2, brightness)

    cv2.rectangle(stage, (0, 0), (width, int(height * 0.72)), (70, 35, 75), -1)
    cv2.rectangle(stage, (0, int(height * 0.72)), (width, height), (35, 35, 38), -1)

    for x in range(100, width, 180):
        cv2.circle(stage, (x, 80), 22, (180, 180, 255), -1)
        cv2.circle(stage, (x, 80), 55, (80, 55, 100), 3)

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


def open_source(
    video_path: str | None,
    camera_index: int | None,
    stream_url: str | None,
) -> tuple[np.ndarray, cv2.VideoCapture | None]:
    """Open a webcam/video or return a generated demo frame."""
    if camera_index is not None:
        return open_camera(camera_index)

    if stream_url is not None:
        capture = cv2.VideoCapture(stream_url)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Could not open network stream: {stream_url}. "
                "Check that the phone and laptop are on the same Wi-Fi network "
                "and use the stream endpoint, usually /video."
            )
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise RuntimeError(
                "The network stream opened, but no video frame could be read. "
                "Try the HTTP URL instead of HTTPS."
            )
        return frame, capture

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


def open_camera(camera_index: int) -> tuple[np.ndarray, cv2.VideoCapture]:
    """
    Open a Windows webcam using several OpenCV backends.

    DirectShow is often the most reliable backend for built-in USB webcams.
    Media Foundation and the default backend are useful fallbacks.
    """
    backends = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("Media Foundation", cv2.CAP_MSMF),
        ("Default", cv2.CAP_ANY),
    ]

    for _backend_name, backend in backends:
        capture = cv2.VideoCapture(camera_index, backend)
        if not capture.isOpened():
            capture.release()
            continue

        ok, frame = capture.read()
        if ok and frame is not None:
            return frame, capture

        capture.release()

    backend_names = ", ".join(name for name, _backend in backends)
    raise RuntimeError(
        f"Could not read camera {camera_index} using {backend_names}. "
        "Check Windows camera permissions, close other camera apps, "
        "or try --camera 1."
    )


def create_object_tracker():
    """Create the MIL tracker available in OpenCV 5 or older contrib builds."""
    if hasattr(cv2, "TrackerMIL_create"):
        return cv2.TrackerMIL_create()

    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMIL_create"):
        return cv2.legacy.TrackerMIL_create()

    raise RuntimeError(
        "No object tracker is available. Install opencv-contrib-python "
        "in the active virtual environment."
    )


def select_object(frame: np.ndarray):
    """
    Let the user drag a rectangle around an object and initialize its tracker.

    Returns:
        tracker, (x, y, width, height), or (None, None) if cancelled.
    """
    cv2.namedWindow("Select Object", cv2.WINDOW_NORMAL)
    box = cv2.selectROI(
        "Select Object",
        frame,
        fromCenter=False,
        showCrosshair=True,
    )
    cv2.destroyWindow("Select Object")

    x, y, width, height = [int(value) for value in box]
    if width <= 0 or height <= 0:
        return None, None

    tracker = create_object_tracker()
    tracker.init(frame, (x, y, width, height))
    return tracker, (x, y, width, height)


def draw_wide_view(
    frame: np.ndarray,
    camera_x: int,
    camera_y: int,
    crop_width: int,
    crop_height: int,
    tracking_box: tuple[int, int, int, int] | None,
    tracking_lost: bool,
) -> np.ndarray:
    """Draw the virtual-camera rectangle and tracked-object box."""
    wide_view = frame.copy()
    left = camera_x - crop_width // 2
    top = camera_y - crop_height // 2
    right = left + crop_width
    bottom = top + crop_height

    cv2.rectangle(wide_view, (left, top), (right, bottom), (0, 255, 0), 4)

    if tracking_box is not None:
        object_x, object_y, object_width, object_height = tracking_box
        box_color = (0, 0, 255) if tracking_lost else (0, 255, 255)
        label = "TRACKING LOST" if tracking_lost else "TRACKED OBJECT"
        cv2.rectangle(
            wide_view,
            (object_x, object_y),
            (object_x + object_width, object_y + object_height),
            box_color,
            3,
        )
        cv2.putText(
            wide_view,
            label,
            (object_x, max(28, object_y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            box_color,
            2,
            cv2.LINE_AA,
        )

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
    cv2.putText(
        wide_view,
        "Click an object to select it, or drag a box for precise selection",
        (24, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return wide_view


def add_status(
    view: np.ndarray,
    camera_x: int,
    camera_y: int,
    zoom: float,
    tracking_active: bool,
    follow_enabled: bool,
    tracking_lost: bool,
) -> np.ndarray:
    """Add camera, tracker, and control information to the output."""
    output = view.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 106), (20, 20, 20), -1)

    cv2.putText(
        output,
        f"Virtual camera   x={camera_x}  y={camera_y}  zoom={zoom:.2f}x",
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if tracking_lost:
        tracking_text = "Tracking: LOST"
        tracking_color = (0, 0, 255)
    else:
        tracking_text = f"Tracking: {'ON' if tracking_active else 'OFF'}"
        tracking_color = (0, 255, 255) if tracking_active else (180, 180, 180)

    cv2.putText(
        output,
        f"{tracking_text}   Auto-follow: {'ON' if follow_enabled else 'OFF'}",
        (18, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        tracking_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "Arrows/WASD: move   T: select   F: follow   C: clear   Q/Esc: quit",
        (18, 87),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return output


def run(
    video_path: str | None = None,
    camera_index: int | None = None,
    stream_url: str | None = None,
) -> None:
    first_frame, capture = open_source(video_path, camera_index, stream_url)
    source_height, source_width = first_frame.shape[:2]

    aspect_ratio = 16 / 9
    base_crop_width = min(int(source_width * 0.60), source_width)
    base_crop_height = min(int(base_crop_width / aspect_ratio), source_height)
    base_crop_width = int(base_crop_height * aspect_ratio)

    camera_x = float(source_width // 2)
    camera_y = float(source_height // 2)
    target_x = camera_x
    target_y = camera_y
    zoom = 1.0
    movement_step = max(12, source_width // 25)

    tracker = None
    tracking_box = None
    tracking_active = False
    follow_enabled = False
    tracking_lost = False

    # Selection happens directly on the live Wide Camera window.
    selection = {
        "dragging": False,
        "start": None,
        "current": None,
        "requested": None,
    }

    cv2.namedWindow("Virtual Camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Wide Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Virtual Camera", 960, 540)
    cv2.resizeWindow("Wide Camera", 960, 540)

    def handle_mouse(event, mouse_x, mouse_y, _flags, _parameter):
        if event == cv2.EVENT_LBUTTONDOWN:
            selection["dragging"] = True
            selection["start"] = (mouse_x, mouse_y)
            selection["current"] = (mouse_x, mouse_y)
        elif event == cv2.EVENT_MOUSEMOVE and selection["dragging"]:
            selection["current"] = (mouse_x, mouse_y)
        elif event == cv2.EVENT_LBUTTONUP and selection["dragging"]:
            selection["dragging"] = False
            selection["current"] = (mouse_x, mouse_y)
            selection["requested"] = (
                selection["start"],
                selection["current"],
            )

    cv2.setMouseCallback("Wide Camera", handle_mouse)

    while True:
        if capture is None:
            frame = first_frame.copy()
        else:
            ok, frame = capture.read()
            if not ok:
                if camera_index is not None or stream_url is not None:
                    break
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = capture.read()
            if not ok:
                break

        height, width = frame.shape[:2]
        crop_width = max(80, min(int(base_crop_width / zoom), width))
        crop_height = max(45, min(int(base_crop_height / zoom), height))
        crop_width = min(crop_width, int(crop_height * aspect_ratio))
        crop_height = min(crop_height, int(crop_width / aspect_ratio))

        # Convert the selection drawn on the 960x540 display back into
        # coordinates in the original camera frame.
        if selection["requested"] is not None:
            (display_x1, display_y1), (display_x2, display_y2) = selection[
                "requested"
            ]
            display_width = abs(display_x2 - display_x1)
            display_height = abs(display_y2 - display_y1)

            if (
                display_width <= CLICK_MOVEMENT_THRESHOLD
                and display_height <= CLICK_MOVEMENT_THRESHOLD
            ):
                # A click creates a default tracking box centered on the
                # cursor. This is the simple manual-selection mode.
                center_x = ((display_x1 + display_x2) / 2) * width / 960
                center_y = ((display_y1 + display_y2) / 2) * height / 540
                object_width = int(width * CLICK_TRACK_BOX_WIDTH_RATIO)
                object_height = int(height * CLICK_TRACK_BOX_HEIGHT_RATIO)
                object_x = int(center_x - object_width / 2)
                object_y = int(center_y - object_height / 2)
            else:
                # A drag creates a more accurate tracking box.
                display_left = max(0, min(display_x1, display_x2))
                display_top = max(0, min(display_y1, display_y2))
                display_right = min(960, max(display_x1, display_x2))
                display_bottom = min(540, max(display_y1, display_y2))

                object_x = int(display_left * width / 960)
                object_y = int(display_top * height / 540)
                object_width = int((display_right - display_left) * width / 960)
                object_height = int((display_bottom - display_top) * height / 540)

            object_width = max(10, min(object_width, width))
            object_height = max(10, min(object_height, height))
            object_x = max(0, min(object_x, width - object_width))
            object_y = max(0, min(object_y, height - object_height))

            if object_width >= 10 and object_height >= 10:
                tracker = create_object_tracker()
                tracker.init(
                    frame,
                    (object_x, object_y, object_width, object_height),
                )
                tracking_box = (
                    object_x,
                    object_y,
                    object_width,
                    object_height,
                )
                tracking_active = True
                follow_enabled = True
                tracking_lost = False

            selection["requested"] = None
            selection["start"] = None
            selection["current"] = None

        if tracking_active and tracker is not None:
            ok, updated_box = tracker.update(frame)
            if ok:
                object_x, object_y, object_width, object_height = [
                    int(value) for value in updated_box
                ]
                tracking_box = (
                    object_x,
                    object_y,
                    object_width,
                    object_height,
                )
                tracking_lost = False
                if follow_enabled:
                    target_x = object_x + object_width / 2
                    target_y = object_y + object_height / 2
            else:
                tracking_active = False
                follow_enabled = False
                tracking_lost = True

        target_x = max(crop_width // 2, min(target_x, width - crop_width // 2))
        target_y = max(crop_height // 2, min(target_y, height - crop_height // 2))
        camera_x += (target_x - camera_x) * SMOOTHING_SPEED
        camera_y += (target_y - camera_y) * SMOOTHING_SPEED
        camera_x_int = int(round(camera_x))
        camera_y_int = int(round(camera_y))

        left = camera_x_int - crop_width // 2
        top = camera_y_int - crop_height // 2
        virtual_view = frame[top : top + crop_height, left : left + crop_width]
        virtual_view = cv2.resize(
            virtual_view,
            (960, 540),
            interpolation=cv2.INTER_LINEAR,
        )
        virtual_view = add_status(
            virtual_view,
            camera_x_int,
            camera_y_int,
            zoom,
            tracking_active,
            follow_enabled,
            tracking_lost,
        )

        wide_view = draw_wide_view(
            frame,
            camera_x_int,
            camera_y_int,
            crop_width,
            crop_height,
            tracking_box,
            tracking_lost,
        )
        wide_view = cv2.resize(wide_view, (960, 540), interpolation=cv2.INTER_AREA)

        if selection["dragging"] and selection["start"] and selection["current"]:
            cv2.rectangle(
                wide_view,
                selection["start"],
                selection["current"],
                (255, 0, 255),
                3,
            )
            cv2.putText(
                wide_view,
                "Release mouse to select",
                (24, 128),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )

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
        elif key in (ord("t"), ord("T")):
            tracker = None
            tracking_box = None
            tracking_active = False
            follow_enabled = False
            tracking_lost = False
        elif key in (ord("f"), ord("F")):
            follow_enabled = tracking_active and not follow_enabled
        elif key in (ord("c"), ord("C")):
            tracker = None
            tracking_box = None
            tracking_active = False
            follow_enabled = False
            tracking_lost = False

    if capture is not None:
        capture.release()
    cv2.destroyAllWindows()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Virtual camera with click-to-select object tracking."
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--camera",
        nargs="?",
        const=0,
        type=int,
        metavar="INDEX",
        help="Use a webcam. Defaults to camera 0.",
    )
    source_group.add_argument(
        "--video",
        type=str,
        help="Use a video file as the wide-camera source.",
    )
    source_group.add_argument(
        "--stream",
        type=str,
        help="Use a network video stream, for example an Android IP Webcam URL.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    try:
        run(arguments.video, arguments.camera, arguments.stream)
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}")
        raise SystemExit(1)
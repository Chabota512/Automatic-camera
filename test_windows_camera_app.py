from __future__ import annotations

import time

import numpy as np

from target_tracking import create_target_reference
from windows_camera_app import PresenterCameraApp, ProviderRuntime


class Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def test_ai_frame_resize_preserves_portrait_aspect_ratio() -> None:
    frame = np.zeros((1280, 720, 3), dtype=np.uint8)

    prepared = PresenterCameraApp._prepare_ai_frame(frame)

    assert prepared.shape[:2] == (640, 360)
    assert prepared.shape[0] / prepared.shape[1] == frame.shape[0] / frame.shape[1]


def test_tracker_box_scaling_round_trips_source_coordinates() -> None:
    source_shape = (1280, 720, 3)
    tracking_shape = (640, 360, 3)
    source_box = (240, 300, 180, 400)

    reduced_box = PresenterCameraApp._scale_box_between_frames(
        source_box,
        source_shape,
        tracking_shape,
    )
    restored_box = PresenterCameraApp._scale_box_between_frames(
        reduced_box,
        tracking_shape,
        source_shape,
    )

    assert reduced_box == (120, 150, 90, 200)
    assert restored_box == source_box


def test_tracking_box_accepts_large_fast_center_jump() -> None:
    assert PresenterCameraApp.valid_tracking_box(
        (390, 120, 50, 60),
        640,
        360,
        (40, 120, 50, 60),
    )


def test_ai_head_box_is_refined_to_include_local_face() -> None:
    app = PresenterCameraApp.__new__(PresenterCameraApp)
    app.frame = np.zeros((360, 640, 3), dtype=np.uint8)
    app.face_boxes = ((250, 150, 70, 80),)
    app._log = lambda _message: None
    candidate = (240, 100, 90, 55)

    refined = app._refine_ai_box_with_face(candidate)

    assert refined[1] < candidate[1]
    assert refined[3] > candidate[3]
    assert refined[1] + refined[3] >= 230


def test_lost_tracker_reacquires_saved_target_at_new_location() -> None:
    rng = np.random.default_rng(21)
    initial_frame = np.full((240, 400, 3), 90, dtype=np.uint8)
    target_crop = rng.integers(20, 235, (80, 50, 3), dtype=np.uint8)
    initial_box = (30, 50, 50, 80)
    initial_frame[50:130, 30:80] = target_crop
    reference = create_target_reference(initial_frame, initial_box)

    later_frame = np.full_like(initial_frame, 90)
    later_frame[120:200, 280:330] = target_crop

    app = PresenterCameraApp.__new__(PresenterCameraApp)
    app.frame = later_frame
    app.target_reference = reference
    app.tracking_box = initial_box
    app.tracker = None
    app.tracking_lost = True
    app.target_mismatch_frames = 3
    app.target_match_score = 0.0
    app.next_target_reacquire = 0.0
    app.track_human = Value(False)
    app.face_boxes = ()
    app.eye_boxes = ()
    app.smile_boxes = ()
    app.human_match_score = None
    app.status = Value("")
    app.target_status = Value("")
    def initialize_tracker(box):
        app.tracker = object()
        app.tracking_box = box

    app._initialize_local_tracker = initialize_tracker
    app._update_target_status = lambda _score=None: None
    app._update_human_match = lambda: None
    app._log = lambda _message: None

    reacquired = app._try_reacquire_target(later_frame)

    assert reacquired
    assert app.tracker is not None
    assert not app.tracking_lost
    assert app.target_reference is reference
    assert app.tracking_box == (280, 120, 50, 80)


def test_ai_fallback_after_loss_makes_provider_requests_due_immediately() -> None:
    cooldown_deadline = time.monotonic() + 30
    app = PresenterCameraApp.__new__(PresenterCameraApp)
    app.ai_runtime = {
        "Gemini": ProviderRuntime(
            model="gemini-test",
            next_check=cooldown_deadline,
            consecutive_failures=1,
        ),
        "Groq": ProviderRuntime(
            model="groq-test",
            next_check=cooldown_deadline,
            consecutive_failures=1,
        ),
    }
    app.ai_provider = Value("Compare both")
    app.ai_enabled = Value(False)
    app.mode = Value("Manual control")
    app.status = Value("")
    app.ai_generation = 4
    app.tracking_lost = True
    app.ai_recovery_deadline = 0.0
    app.mode_changed = lambda update_status=False: None
    app.update_ai_indicator = lambda *args, **kwargs: None
    app._log = lambda _message: None
    app._refresh_provider_statuses = lambda: None

    app._start_ai_after_tracking_loss()

    assert app.ai_enabled.get()
    assert app.mode.get() == "Automatic AI"
    assert app.ai_generation == 5
    assert all(runtime.next_check < cooldown_deadline for runtime in app.ai_runtime.values())

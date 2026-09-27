"""Unit tests for session-only target references."""

from __future__ import annotations

import cv2
import numpy as np

from target_tracking import (
    box_iou,
    create_target_reference,
    detect_faces,
    detect_face_features,
    find_template_match,
    human_match_score,
)


def test_reference_matches_the_selected_target() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.rectangle(frame, (40, 25), (90, 95), (30, 120, 220), thickness=-1)

    reference = create_target_reference(frame, (40, 25, 51, 71))

    assert reference.appearance_score(frame, (40, 25, 51, 71)) > 0.98
    assert reference.reference_crop is not None
    assert reference.reference_crop.shape == (71, 51, 3)


def test_reference_rejects_a_different_appearance() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.rectangle(frame, (40, 25), (90, 95), (30, 120, 220), thickness=-1)
    cv2.rectangle(frame, (100, 25), (150, 95), (220, 30, 30), thickness=-1)

    reference = create_target_reference(frame, (40, 25, 51, 71))

    assert reference.appearance_score(frame, (100, 25, 51, 71)) < 0.5


def test_box_iou_handles_overlap_and_separation() -> None:
    assert box_iou((0, 0, 10, 10), (5, 5, 10, 10)) == 25 / 175
    assert box_iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0


def test_human_match_score_distinguishes_target_and_distant_face() -> None:
    target = (40, 20, 80, 140)
    assert human_match_score(target, [(70, 35, 30, 30)]) == 1.0
    assert human_match_score(target, [(300, 35, 30, 30)]) < 0.5


def test_human_match_score_returns_none_when_face_is_temporarily_hidden() -> None:
    assert human_match_score((40, 20, 80, 140), []) is None


def test_detect_faces_normalizes_opencv_boxes() -> None:
    class FakeClassifier:
        def detectMultiScale(self, _image, **_kwargs):
            return np.array([[3, 4, 20, 25], [30, 40, 50, 60]])

    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    assert detect_faces(frame, FakeClassifier()) == (
        (3, 4, 20, 25),
        (30, 40, 50, 60),
    )


def test_template_match_reacquires_a_moved_target() -> None:
    rng = np.random.default_rng(17)
    template = rng.integers(30, 220, (50, 32, 3), dtype=np.uint8)
    later_frame = np.full((220, 340, 3), 110, dtype=np.uint8)
    slightly_changed = np.clip(
        template.astype(np.int16) + 5,
        0,
        255,
    ).astype(np.uint8)
    later_frame[115:165, 180:212] = slightly_changed

    match = find_template_match(later_frame, template, scale_factors=(1.0,))

    assert match is not None
    box, score = match
    assert box == (180, 115, 32, 50)
    assert score > 0.75


def test_face_feature_detection_maps_eye_and_mouth_boxes() -> None:
    class FakeClassifier:
        def __init__(self, detections):
            self.detections = np.array(detections)

        def detectMultiScale(self, _image, **_kwargs):
            return self.detections

    frame = np.zeros((120, 120), dtype=np.uint8)
    eyes, smiles = detect_face_features(
        frame,
        [(10, 10, 80, 80)],
        FakeClassifier([[15, 20, 20, 10]]),
        FakeClassifier([[12, 5, 30, 12]]),
    )

    assert eyes == ((25, 30, 20, 10),)
    assert smiles == ((22, 51, 30, 12),)


def test_eye_and_mouth_cues_raise_human_target_match() -> None:
    target = (0, 0, 50, 50)
    face = [(60, 0, 20, 20)]

    face_only = human_match_score(target, face)
    with_features = human_match_score(
        target,
        face,
        eye_boxes=[(10, 10, 8, 5), (25, 10, 8, 5)],
        smile_boxes=[(15, 25, 18, 8)],
    )

    assert face_only is not None
    assert with_features is not None
    assert with_features > face_only
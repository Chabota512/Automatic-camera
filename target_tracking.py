"""Session-only target references and appearance matching.

This module deliberately contains no Tkinter or provider code. It gives the
desktop application a small, testable contract for comparing a candidate box
with the target selected by the operator.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

Box = tuple[int, int, int, int]
REFERENCE_SIZE = (128, 128)
FACE_DETECT_SCALE = 1.1
FACE_DETECT_MIN_NEIGHBORS = 5
FACE_DETECT_MIN_SIZE = (24, 24)
FEATURE_MATCH_WEIGHT = 0.35


def crop_box(frame: np.ndarray, box: Box) -> np.ndarray | None:
    """Return a copy of a box, clipped to the frame, or None when empty."""
    if frame is None or frame.ndim < 2:
        return None

    frame_height, frame_width = frame.shape[:2]
    x, y, width, height = [int(value) for value in box]
    left = max(0, x)
    top = max(0, y)
    right = min(frame_width, x + width)
    bottom = min(frame_height, y + height)
    if right <= left or bottom <= top:
        return None
    return frame[top:bottom, left:right].copy()


def appearance_histogram(crop: np.ndarray) -> np.ndarray:
    """Build a normalized HSV histogram that is tolerant of small motion."""
    if crop is None or crop.size == 0:
        raise ValueError("Cannot build a target reference from an empty crop.")

    resized = cv2.resize(crop, REFERENCE_SIZE, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [24, 16],
        [0, 180, 0, 256],
    )
    normalized = cv2.normalize(histogram, histogram, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return normalized.astype(np.float32)


def box_iou(first: Box, second: Box) -> float:
    """Return intersection-over-union for two pixel-space boxes."""
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    left = max(first_x, second_x)
    top = max(first_y, second_y)
    right = min(first_x + first_width, second_x + second_width)
    bottom = min(first_y + first_height, second_y + second_height)
    intersection_width = max(0, right - left)
    intersection_height = max(0, bottom - top)
    intersection = intersection_width * intersection_height
    first_area = max(0, first_width) * max(0, first_height)
    second_area = max(0, second_width) * max(0, second_height)
    union = first_area + second_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def detect_faces(
    frame: np.ndarray,
    classifier: cv2.CascadeClassifier,
    *,
    scale_factor: float = FACE_DETECT_SCALE,
    min_neighbors: int = FACE_DETECT_MIN_NEIGHBORS,
    min_size: tuple[int, int] = FACE_DETECT_MIN_SIZE,
) -> tuple[Box, ...]:
    """Detect faces in a frame with a caller-supplied OpenCV cascade.

    The classifier is intentionally supplied by the caller so this function
    remains easy to test and does not load model files as an import side
    effect. Haar detection is only a local signal; it is not face recognition.
    """
    if frame is None or frame.size == 0 or frame.ndim < 2:
        return ()

    if frame.ndim == 2:
        grayscale = frame
    else:
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.equalizeHist(grayscale)

    detections = classifier.detectMultiScale(
        grayscale,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )
    return tuple(
        (int(x), int(y), int(width), int(height))
        for x, y, width, height in detections
    )


def detect_face_features(
    frame: np.ndarray,
    face_boxes: tuple[Box, ...] | list[Box],
    eye_classifier: cv2.CascadeClassifier,
    smile_classifier: cv2.CascadeClassifier,
) -> tuple[tuple[Box, ...], tuple[Box, ...]]:
    """Find eye and smile regions inside already-detected face boxes."""
    if frame is None or frame.size == 0 or not face_boxes:
        return (), ()
    grayscale = (
        frame
        if frame.ndim == 2
        else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    )
    grayscale = cv2.equalizeHist(grayscale)
    frame_height, frame_width = grayscale.shape[:2]
    eyes: list[Box] = []
    smiles: list[Box] = []

    for face_x, face_y, face_width, face_height in face_boxes:
        left = max(0, face_x)
        top = max(0, face_y)
        right = min(frame_width, face_x + face_width)
        bottom = min(frame_height, face_y + face_height)
        if right <= left or bottom <= top:
            continue
        face_region = grayscale[top:bottom, left:right]
        region_height, region_width = face_region.shape[:2]
        detected_eyes = eye_classifier.detectMultiScale(
            face_region,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(max(8, region_width // 10), max(8, region_height // 12)),
        )
        eyes.extend(
            (
                int(left + x),
                int(top + y),
                int(width),
                int(height),
            )
            for x, y, width, height in detected_eyes
        )

        mouth_top = int(region_height * 0.45)
        lower_face = face_region[mouth_top:]
        detected_smiles = smile_classifier.detectMultiScale(
            lower_face,
            scaleFactor=1.2,
            minNeighbors=12,
            minSize=(max(12, region_width // 5), max(8, region_height // 12)),
        )
        smiles.extend(
            (
                int(left + x),
                int(top + mouth_top + y),
                int(width),
                int(height),
            )
            for x, y, width, height in detected_smiles
        )

    return tuple(eyes), tuple(smiles)


def find_template_match(
    frame: np.ndarray,
    template: np.ndarray,
    scale_factors: tuple[float, ...] = (0.85, 1.0, 1.15),
) -> tuple[Box, float] | None:
    """Find a saved target crop in a later frame at nearby scales."""
    if (
        frame is None
        or template is None
        or frame.size == 0
        or template.size == 0
    ):
        return None

    frame_gray = (
        frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    )
    template_gray = (
        template
        if template.ndim == 2
        else cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    )
    if float(np.std(template_gray)) < 2.0:
        return None

    frame_height, frame_width = frame_gray.shape[:2]
    template_height, template_width = template_gray.shape[:2]
    best: tuple[Box, float] | None = None
    for scale in scale_factors:
        search_width = int(round(template_width * scale))
        search_height = int(round(template_height * scale))
        if (
            search_width < 12
            or search_height < 12
            or search_width > frame_width
            or search_height > frame_height
        ):
            continue
        search_template = cv2.resize(
            template_gray,
            (search_width, search_height),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        scores = cv2.matchTemplate(
            frame_gray,
            search_template,
            cv2.TM_CCOEFF_NORMED,
        )
        _, score, _, location = cv2.minMaxLoc(scores)
        if best is None or score > best[1]:
            best = (
                (int(location[0]), int(location[1]), search_width, search_height),
                float(score),
            )
    return best


def human_match_score(
    target_box: Box,
    face_boxes: tuple[Box, ...] | list[Box],
    eye_boxes: tuple[Box, ...] | list[Box] = (),
    smile_boxes: tuple[Box, ...] | list[Box] = (),
) -> float | None:
    """Return how strongly a detected face belongs to the selected target.

    A face whose center is inside the selected target box is a strong local
    match. Faces just outside the box receive a tapered score, which makes
    this useful as an arbitration signal without making it a hard tracker
    requirement. ``None`` means no face was detected, not that the target was
    lost.
    """
    if not face_boxes and not eye_boxes and not smile_boxes:
        return None

    target_x, target_y, target_width, target_height = target_box
    if target_width <= 0 or target_height <= 0:
        return 0.0

    target_right = target_x + target_width
    target_bottom = target_y + target_height
    target_diagonal = max(
        1.0,
        (target_width**2 + target_height**2) ** 0.5,
    )
    best_score = 0.0
    for face_x, face_y, face_width, face_height in face_boxes:
        if face_width <= 0 or face_height <= 0:
            continue
        face_center_x = face_x + face_width / 2
        face_center_y = face_y + face_height / 2
        closest_x = min(max(face_center_x, target_x), target_right)
        closest_y = min(max(face_center_y, target_y), target_bottom)
        distance = (
            (face_center_x - closest_x) ** 2
            + (face_center_y - closest_y) ** 2
        ) ** 0.5
        score = max(0.0, 1.0 - distance / target_diagonal)
        best_score = max(best_score, score)

    feature_boxes = tuple(eye_boxes) + tuple(smile_boxes)
    if not feature_boxes:
        return best_score

    feature_matches = 0
    for feature_x, feature_y, feature_width, feature_height in feature_boxes:
        feature_center_x = feature_x + feature_width / 2
        feature_center_y = feature_y + feature_height / 2
        if (
            target_x <= feature_center_x <= target_right
            and target_y <= feature_center_y <= target_bottom
        ):
            feature_matches += 1
    feature_score = feature_matches / len(feature_boxes)
    return max(
        best_score,
        best_score * (1.0 - FEATURE_MATCH_WEIGHT)
        + feature_score * FEATURE_MATCH_WEIGHT,
    )


@dataclass(frozen=True)
class TargetReference:
    """The selected target's in-memory appearance reference."""

    initial_box: Box
    histogram: np.ndarray
    target_kind: str = "generic"
    reference_crop: np.ndarray | None = None

    def appearance_score(self, frame: np.ndarray, candidate_box: Box) -> float:
        """Return a 0..1 similarity score for a candidate box."""
        candidate_crop = crop_box(frame, candidate_box)
        if candidate_crop is None:
            return 0.0

        candidate_histogram = appearance_histogram(candidate_crop)
        distance = float(
            cv2.compareHist(
                self.histogram,
                candidate_histogram,
                cv2.HISTCMP_BHATTACHARYYA,
            )
        )
        return max(0.0, min(1.0, 1.0 - distance))


def create_target_reference(
    frame: np.ndarray,
    box: Box,
    target_kind: str = "generic",
) -> TargetReference:
    """Capture a target reference from the current frame."""
    crop = crop_box(frame, box)
    if crop is None:
        raise ValueError("The selected target is outside the current frame.")
    return TargetReference(
        initial_box=box,
        histogram=appearance_histogram(crop),
        target_kind=target_kind,
        reference_crop=crop,
    )

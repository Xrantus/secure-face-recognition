"""OpenCV overlay helpers for live face recognition preview."""

from __future__ import annotations

import cv2
import numpy as np

from .face_recognizer import SimilarityMetric

WINDOW_TITLE = "Face Recognition (YOLO + InsightFace)"


def draw_face_label(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    name: str,
    score: float,
    metric: SimilarityMetric,
) -> None:
    """Draw bounding box and identity label on frame."""
    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
    label = f"{name} {score:.2f}" if metric == "cosine" else f"{name} {score:.3f}"
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(0, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def show_frame(frame: np.ndarray, window_title: str = WINDOW_TITLE) -> bool:
    """Show frame in OpenCV window. Returns False if user pressed 'q'."""
    cv2.imshow(window_title, frame)
    return (cv2.waitKey(1) & 0xFF) != ord("q")

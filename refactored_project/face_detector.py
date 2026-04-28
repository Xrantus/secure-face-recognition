"""Face detection module based on YOLO (Ultralytics).

This module is intentionally *detection-only* and does not depend on the recognizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class Detection:
    """A single detection result.

    Attributes:
        bbox: Bounding box as (x1, y1, x2, y2) in pixel coordinates.
        conf: Confidence score.
    """

    bbox: tuple[int, int, int, int]
    conf: float


class FaceDetector:
    """YOLO-based face detector that returns filtered bounding boxes."""

    def __init__(
        self,
        model_path: str,
        img_size: int,
        pred_conf: float,
        iou: float,
        max_det: int,
        det_threshold: float,
    ) -> None:
        """Initialize the YOLO model and detection parameters."""

        from ultralytics import YOLO

        self._model_path = model_path
        self._yolo = YOLO(model_path, task="detect")
        self._img_size = int(img_size)
        self._pred_conf = float(pred_conf)
        self._iou = float(iou)
        self._max_det = int(max_det)
        self._det_threshold = float(det_threshold)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Detect faces in a BGR frame and return filtered detections."""

        results = self._yolo.predict(
            frame_bgr,
            imgsz=self._img_size,
            verbose=False,
            conf=self._pred_conf,
            iou=self._iou,
            max_det=self._max_det,
        )
        if not results:
            return []

        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        dets: list[Detection] = []
        for i in range(len(boxes)):
            conf = float(boxes.conf[i])
            if conf < self._det_threshold:
                continue

            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            dets.append(Detection(bbox=(x1, y1, x2, y2), conf=conf))

        return dets

    @staticmethod
    def best_by_conf(detections: Iterable[Detection]) -> Detection | None:
        """Return the detection with highest confidence (or None)."""

        best: Detection | None = None
        for d in detections:
            if best is None or d.conf > best.conf:
                best = d
        return best


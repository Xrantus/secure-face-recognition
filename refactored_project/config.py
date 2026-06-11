"""Central configuration for hardware, models, and similarity metrics.

This module is intentionally split into three logical groups:
- HARDWARE_ENV: camera backend selection only (MAC vs RPI).
- MODEL_CONFIG: model selection and inference knobs (hardware-agnostic).
- METRIC_CONFIG: similarity metric selection and thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
# Windows/Mac gelistirme icin WIN; Pi'de run_system otomatik RPI algilar.
HARDWARE_ENV: Literal["MAC", "RPI", "WIN"] = "WIN"


@dataclass(frozen=True)
class ModelConfig:
    """Model and inference parameters (hardware-agnostic)."""

    yolo_model_path: str
    recognizer_model_name: str = "buffalo_s"

    yolo_img_size: int = 640
    yolo_pred_conf: float = 0.01
    yolo_det_threshold: float = 0.15
    yolo_iou: float = 0.45
    max_det: int = 100

    det_size: tuple[int, int] = (160, 160)
    landmark_pad: float = 0.20
    min_face_size: int = 35
    frame_skip: int = 2

    db_path: str = "known_faces_embeddings.npz"


@dataclass(frozen=True)
class MetricConfig:
    """Similarity metric selection and thresholds."""

    similarity_metric: Literal["cosine", "euclidean"] = "cosine"
    cosine_threshold: float = 0.50
    euclidean_threshold: float = 1.00


@dataclass(frozen=True)
class CameraConfig:
    """Camera/backend configuration (hardware-dependent)."""

    opencv_camera_index: int = 0
    opencv_frame_width: int = 1280
    opencv_frame_height: int = 720
    rpi_preview_size: tuple[int, int] = (640, 480)


def default_model_config(project_root: Path) -> ModelConfig:
    """Return a sensible default model config for this repository."""

    yolo_dir = project_root / "yolo11-modes"
    preferred = yolo_dir / "face_yolo11n_int8.onnx"
    if preferred.is_file():
        return ModelConfig(yolo_model_path=str(preferred))

    # Fallback: pick the first ONNX file found in yolo11-modes/.
    if yolo_dir.is_dir():
        onnx_files = sorted(p for p in yolo_dir.iterdir() if p.suffix.lower() == ".onnx")
        if onnx_files:
            return ModelConfig(yolo_model_path=str(onnx_files[0]))

    # As a last resort keep the preferred path (will fail loudly at runtime).
    return ModelConfig(yolo_model_path=str(preferred))


MODEL_CONFIG: ModelConfig = default_model_config(Path(__file__).resolve().parents[1])
METRIC_CONFIG = MetricConfig()
CAMERA_CONFIG = CameraConfig()


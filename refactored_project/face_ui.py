"""OpenCV / Picamera2 overlay helpers for live face recognition preview."""

from __future__ import annotations

import os

import cv2
import numpy as np

from .face_recognizer import SimilarityMetric

WINDOW_TITLE = "Face Recognition (YOLO + InsightFace)"

_gui_warned = False
_rpi_preview_mode: str | None = None


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


def start_rpi_preview(picam) -> str | None:
    """Start HDMI preview on Raspberry Pi. Returns backend name or None."""
    global _rpi_preview_mode

    try:
        from picamera2 import Preview
    except ImportError:
        print("[UI] Picamera2 Preview modulu bulunamadi.")
        _rpi_preview_mode = None
        return None

    for preview_cls, name in ((Preview.QTGL, "QTGL"), (Preview.DRM, "DRM")):
        try:
            picam.start_preview(preview_cls)
            _rpi_preview_mode = name
            print(
                f"[UI] RPi onizleme baslatildi ({name}). "
                "Goruntu Pi'ye bagli HDMI/monitorde acilir (SSH penceresinde degil)."
            )
            return name
        except Exception as exc:
            print(f"[UI] {name} onizleme basarisiz: {exc}")

    _rpi_preview_mode = None
    print(
        "[UI] UYARI: HDMI onizleme acilamadi. "
        "Pi'ye monitor bagli mi? Masaustu oturumu acik mi? (SSH disinda)"
    )
    return None


def show_frame_rpi(picam, frame: np.ndarray) -> None:
    """Push annotated frame to Picamera2 HDMI overlay."""
    cfg = picam.camera_configuration()
    main = cfg.get("main", {})
    target_w, target_h = main.get("size", (frame.shape[1], frame.shape[0]))

    h, w = frame.shape[:2]
    if (w, h) != (target_w, target_h):
        frame = cv2.resize(frame, (target_w, target_h))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    alpha = np.full((target_h, target_w, 1), 255, dtype=np.uint8)
    picam.set_overlay(np.concatenate([rgb, alpha], axis=2))


def stop_rpi_preview(picam) -> None:
    """Clear overlay and stop Picamera2 preview."""
    global _rpi_preview_mode

    try:
        picam.set_overlay(None)
    except Exception:
        pass
    try:
        picam.stop_preview()
    except Exception:
        pass
    _rpi_preview_mode = None


def _try_opencv_window(frame: np.ndarray, window_title: str) -> bool:
    """Show frame in OpenCV window. Returns False if user pressed 'q'."""
    global _gui_warned

    if not os.environ.get("DISPLAY"):
        os.environ["DISPLAY"] = ":0"

    try:
        cv2.imshow(window_title, frame)
        return (cv2.waitKey(1) & 0xFF) != ord("q")
    except cv2.error as exc:
        if not _gui_warned:
            print(
                f"[UI] OpenCV penceresi acilamadi ({exc}). "
                "RPi'de HDMI monitor veya masaustu oturumu gerekir."
            )
            _gui_warned = True
        return True


def show_frame(
    frame: np.ndarray,
    window_title: str = WINDOW_TITLE,
    picam=None,
) -> bool:
    """
    Show annotated frame.
    On RPi with active preview uses HDMI overlay; otherwise tries OpenCV.
    Returns False if user pressed 'q' (OpenCV mode only).
    """
    global _gui_warned

    if picam is not None and _rpi_preview_mode is not None:
        try:
            show_frame_rpi(picam, frame)
            return True
        except Exception as exc:
            if not _gui_warned:
                print(f"[UI] RPi overlay hatasi: {exc}")
                _gui_warned = True

    return _try_opencv_window(frame, window_title)

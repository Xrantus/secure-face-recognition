"""FastAPI Web Server for RPi - Backend Integration.

Provides endpoints for the Spring Boot backend to interact with the RPi.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException

from .face_recognizer import FaceRecognizer

app = FastAPI(title="RPi Face Recognition API", description="Integration API between RPi and Central Backend")

# Global variables that will be injected by the main system script
_recognizer: FaceRecognizer | None = None
_inference_lock: threading.Lock | None = None
_db_lock: threading.Lock | None = None
_reload_callback = None


def setup_api(
    recognizer: FaceRecognizer,
    inference_lock: threading.Lock,
    db_lock: threading.Lock,
    reload_callback: Callable,
) -> None:
    """Inject dependencies from the main run loop."""
    global _recognizer, _inference_lock, _db_lock, _reload_callback
    _recognizer = recognizer
    _inference_lock = inference_lock
    _db_lock = db_lock
    _reload_callback = reload_callback


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint for Spring Boot."""
    return {"status": "UP"}


@app.post("/generate")
async def generate_embedding(file: UploadFile = File(...)) -> dict[str, Any]:
    """Extract a 512-dimensional vector from the uploaded image."""
    if _recognizer is None or _inference_lock is None:
        raise HTTPException(status_code=503, detail="API is not fully initialized.")

    try:
        # Read the image file into memory
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        # Thread-safe inference to prevent crashing if camera loop is also processing
        with _inference_lock:
            # Note: We use the base FaceRecognizer which produces 512D embeddings as requested
            emb = _recognizer.embed_from_roi(img_bgr)

        if emb is None:
            raise HTTPException(status_code=400, detail="No face detected in the image.")

        # emb is a 512-dimensional numpy array, convert to list for JSON serialization
        return {"embedding": emb.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload")
def reload_cache(background_tasks: BackgroundTasks) -> dict[str, str]:
    """Triggered by Backend to refresh personnel embeddings."""
    if _reload_callback is None:
        raise HTTPException(status_code=503, detail="Reload callback is not set.")

    # We add the actual download and reload task to background tasks 
    # so we can return 200 OK immediately.
    background_tasks.add_task(_reload_callback)
    return {"message": "Reload process started in the background."}


if __name__ == "__main__":
    import uvicorn
    # RPi starts this web server on port 8000.
    uvicorn.run(app, host="0.0.0.0", port=8000)

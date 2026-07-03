"""Face recognition module based on InsightFace.

Responsibilities:
- Extract a normalized embedding from a cropped ROI.
- Compute similarity/distance using a selectable metric (cosine/euclidean).
- Load DB embeddings and perform identity prediction.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


SimilarityMetric = Literal["cosine", "euclidean"]


class FaceRecognizer:
    """InsightFace recognizer that can embed and compare faces."""

    def __init__(
        self,
        det_size: tuple[int, int],
        model_name: str,
        providers: list[str] | None = None,
    ) -> None:
        """Initialize InsightFace FaceAnalysis with detection+recognition modules."""

        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise SystemExit(
                "Could not initialize InsightFace. Most likely `onnxruntime` is missing.\n"
                "Fix (inside virtual environment):\n"
                "  pip install onnxruntime\n"
                "Alternative: install using `requirements.txt` from the project root."
            ) from e

        kwargs = {}
        if providers is not None:
            kwargs["providers"] = providers

        self._app = FaceAnalysis(
            name=model_name,
            root=".",
            allowed_modules=["detection", "recognition"],
            **kwargs,
        )
        self._app.prepare(ctx_id=-1, det_size=det_size)

    def embed_from_roi(self, roi_bgr: np.ndarray) -> np.ndarray | None:
        """Compute a L2-normalized embedding from a BGR ROI.

        The method detects landmarks inside ROI, performs 112x112 alignment, then
        runs the recognition model.
        """

        from insightface.utils import face_align

        try:
            _, kpss = self._app.det_model.detect(roi_bgr, max_num=1, metric="default")
        except Exception:
            return None

        kps = self._first_landmarks(kpss)
        if kps is None:
            return None

        aligned = face_align.norm_crop(roi_bgr, landmark=kps)
        emb = self._app.models["recognition"].get_feat(aligned)[0]
        return self.l2_normalize(emb)

    @staticmethod
    def l2_normalize(vec: np.ndarray) -> np.ndarray:
        """Return L2-normalized copy of a vector (safe for near-zero norms)."""

        v = np.asarray(vec, dtype=np.float32)
        n = float(np.linalg.norm(v))
        if n <= 1e-12:
            return v
        return v / n

    @staticmethod
    def calculate_similarity(emb1: np.ndarray, emb2: np.ndarray, metric: SimilarityMetric) -> float:
        """Compute similarity/distance between two embeddings.

        - cosine: dot product (larger is better) assuming L2-normalized vectors.
        - euclidean: L2 distance (smaller is better).
        """

        a = np.asarray(emb1, dtype=np.float32)
        b = np.asarray(emb2, dtype=np.float32)
        if metric == "cosine":
            return float(np.dot(a, b))
        if metric == "euclidean":
            return float(np.linalg.norm(a - b))
        raise ValueError(f"Unsupported metric: {metric}")

    @staticmethod
    def load_db(npz_path: str) -> tuple[np.ndarray, np.ndarray]:
        """Load DB embeddings and names from an `.npz` file.

        Expected keys: `encodings`, `names`.
        DB embeddings are row-wise L2-normalized on load for stable scoring.
        """

        db = np.load(npz_path, allow_pickle=True)
        embs = np.asarray(db["encodings"], dtype=np.float32)
        names = np.asarray(db["names"])

        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        embs = embs / norms
        return embs, names

    @staticmethod
    def predict_identity(
        emb: np.ndarray,
        db_embs: np.ndarray,
        db_names: np.ndarray,
        metric: SimilarityMetric,
        threshold: float,
    ) -> tuple[str, float]:
        """Predict identity from DB using selected metric and threshold.

        Returns:
            (best_name_or_unknown, best_score)

        Threshold logic:
        - cosine: accept if best_score >= threshold
        - euclidean: accept if best_score <= threshold
        """

        if db_embs.size == 0:
            return "Unknown", float("nan")

        if metric == "cosine":
            sims = np.dot(db_embs, emb.astype(np.float32))
            idx = int(np.argmax(sims))
            best = float(sims[idx])
            name = str(db_names[idx])
            return (name, best) if best >= threshold else ("Unknown", best)

        if metric == "euclidean":
            dists = np.linalg.norm(db_embs - emb.astype(np.float32), axis=1)
            idx = int(np.argmin(dists))
            best = float(dists[idx])
            name = str(db_names[idx])
            return (name, best) if best <= threshold else ("Unknown", best)

        raise ValueError(f"Unsupported metric: {metric}")

    @staticmethod
    def _first_landmarks(kpss: object) -> np.ndarray | None:
        """Extract the first (5,2) landmarks array from InsightFace output."""

        if kpss is None:
            return None
        if isinstance(kpss, np.ndarray):
            if kpss.size == 0:
                return None
            if kpss.ndim == 2 and kpss.shape == (5, 2):
                return kpss
            if kpss.ndim == 3 and kpss.shape[1:] == (5, 2):
                return kpss[0]
            return None
        try:
            kpss_list = list(kpss)  # type: ignore[arg-type]
        except Exception:
            return None
        if not kpss_list:
            return None
        k0 = np.asarray(kpss_list[0])
        return k0 if k0.shape == (5, 2) else None


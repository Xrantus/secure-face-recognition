"""128-dimensional Face Recognizer wrapper.

This file provides a specialized FaceRecognizer that forces the output
embeddings to be 128 dimensions by slicing the original 512d vector
and re-normalizing it. It inherits the rest of the logic from the base class.
"""

from __future__ import annotations

import numpy as np

from .face_recognizer import FaceRecognizer


class FaceRecognizer128(FaceRecognizer):
    """An InsightFace wrapper that slices the 512d embedding to 128d."""

    def embed_from_roi(self, roi_bgr: np.ndarray) -> np.ndarray | None:
        """Compute a 128d L2-normalized embedding from a BGR ROI."""
        
        emb = super().embed_from_roi(roi_bgr)
        if emb is None:
            return None
            
        # 128-boyuta indirge ve L2 normalizasyonu yap
        emb_128 = emb[:128]
        return self.l2_normalize(emb_128)

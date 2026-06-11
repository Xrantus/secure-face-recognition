"""Access log policy with lightweight bbox tracking for multi-face scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class FaceObservation:
    bbox: BBox
    name: str
    score: float


@dataclass
class _FaceTrack:
    track_id: int
    bbox: BBox
    last_seen: float
    last_identity: str = "Unknown"
    unknown_streak: int = 0
    unknown_logged: bool = False
    last_unknown_score: float | None = None
    ever_recognized_user: str | None = None
    suppress_unknown_until: float = 0.0


def _iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / (union + 1e-6)


class AccessLogPolicy:
    """Per-userId authorized log cooldown and per-track unknown confirmation/logging."""

    def __init__(
        self,
        on_authorized: Callable[[str], None],
        on_unknown: Callable[[int, float | None], None],
        absence_reset_seconds: float = 4.0,
        unknown_confirm_cycles: int = 3,
        auth_unknown_suppress_seconds: float = 8.0,
        authorized_log_cooldown_seconds: float = 60.0,
        track_iou_threshold: float = 0.3,
        spatial_suppress_iou: float = 0.25,
    ) -> None:
        self._on_authorized = on_authorized
        self._on_unknown = on_unknown
        self._absence_reset = absence_reset_seconds
        self._unknown_confirm = unknown_confirm_cycles
        self._auth_unknown_suppress = auth_unknown_suppress_seconds
        self._authorized_log_cooldown = authorized_log_cooldown_seconds
        self._track_iou_threshold = track_iou_threshold
        self._spatial_suppress_iou = spatial_suppress_iou

        self._tracks: dict[int, _FaceTrack] = {}
        self._next_track_id = 1
        self._authorized_last_seen: dict[str, float] = {}
        self._last_authorized_log_sent: dict[str, float] = {}
        self._user_unknown_suppress_until: dict[str, float] = {}

    def update(self, observations: list[FaceObservation], t_now: float) -> None:
        self._prune_expired(t_now)

        matched, unmatched_obs = self._match_observations(observations)

        for track_id, obs in matched.items():
            self._apply_observation(self._tracks[track_id], obs, t_now)

        for obs in unmatched_obs:
            track = _FaceTrack(
                track_id=self._next_track_id,
                bbox=obs.bbox,
                last_seen=t_now,
            )
            self._next_track_id += 1
            self._tracks[track.track_id] = track
            self._apply_observation(track, obs, t_now)

    def _prune_expired(self, t_now: float) -> None:
        expired_track_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if t_now - track.last_seen > self._absence_reset
        ]
        for track_id in expired_track_ids:
            del self._tracks[track_id]

        for user_id in list(self._user_unknown_suppress_until):
            if t_now - self._authorized_last_seen.get(user_id, 0.0) > self._absence_reset:
                self._user_unknown_suppress_until.pop(user_id, None)

    def _match_observations(
        self,
        observations: list[FaceObservation],
    ) -> tuple[dict[int, FaceObservation], list[FaceObservation]]:
        matched: dict[int, FaceObservation] = {}
        used_track_ids: set[int] = set()
        unmatched_obs: list[FaceObservation] = []

        for obs in observations:
            best_track_id: int | None = None
            best_iou = 0.0
            for track_id, track in self._tracks.items():
                if track_id in used_track_ids:
                    continue
                overlap = _iou(obs.bbox, track.bbox)
                if overlap > best_iou:
                    best_iou = overlap
                    best_track_id = track_id

            if best_track_id is not None and best_iou >= self._track_iou_threshold:
                matched[best_track_id] = obs
                used_track_ids.add(best_track_id)
            else:
                unmatched_obs.append(obs)

        return matched, unmatched_obs

    def _unknown_suppressed(self, bbox: BBox, track: _FaceTrack, t_now: float) -> bool:
        if t_now < track.suppress_unknown_until:
            return True

        for other in self._tracks.values():
            if other.track_id == track.track_id:
                continue
            user_id = other.ever_recognized_user
            if not user_id:
                continue
            if t_now >= self._user_unknown_suppress_until.get(user_id, 0.0):
                continue
            if _iou(bbox, other.bbox) >= self._spatial_suppress_iou:
                return True
        return False

    def _apply_observation(self, track: _FaceTrack, obs: FaceObservation, t_now: float) -> None:
        prev_identity = track.last_identity
        track.bbox = obs.bbox
        track.last_seen = t_now
        track.last_identity = obs.name

        if obs.name != "Unknown":
            if prev_identity == "Unknown" and not track.unknown_logged:
                track.unknown_streak = 0

            self._authorized_last_seen[obs.name] = t_now
            last_log_sent = self._last_authorized_log_sent.get(obs.name, 0.0)
            if t_now - last_log_sent >= self._authorized_log_cooldown:
                self._last_authorized_log_sent[obs.name] = t_now
                self._on_authorized(obs.name)

            track.ever_recognized_user = obs.name
            track.suppress_unknown_until = t_now + self._auth_unknown_suppress
            self._user_unknown_suppress_until[obs.name] = t_now + self._auth_unknown_suppress
            return

        if track.unknown_logged or self._unknown_suppressed(obs.bbox, track, t_now):
            return

        track.unknown_streak += 1
        track.last_unknown_score = obs.score
        if track.unknown_streak >= self._unknown_confirm:
            track.unknown_logged = True
            self._on_unknown(track.track_id, track.last_unknown_score)

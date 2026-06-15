"""Backend Client for communicating with the Spring Boot Server.

Handles fetching embeddings, sending access logs, and offline log synchronization.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from typing import Any

import numpy as np
import requests

from .face_recognizer import FaceRecognizer

# TODO: Gercek backend IP ve Portunuzu buraya yazin
BACKEND_BASE_URL = "http://127.0.0.1:8080"
OFFLINE_LOGS_FILE = "offline_logs.json"

_log_lock = threading.Lock()

def _get_current_time_iso() -> str:
    """Returns current time in ISO 8601 format."""
    return datetime.now().isoformat()


def fetch_and_save_embeddings(db_abs: str) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Fetch all active personnel embeddings from the backend and save to .npz.
    Returns (embs, names) if successful, otherwise None.
    """
    url = f"{BACKEND_BASE_URL}/api/embedding/all-active"
    print(f"[Backend Client] Yuz verileri cekiliyor: {url}")
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # NOTE: Backend'in dondurdugu JSON yapisina gore asagidaki anahtarlari (keys) degistirmeniz gerekebilir.
        # Varsayim: [{"id": "Ahmet", "embedding": [0.1, 0.2, ...]}, ...]
        new_names = []
        new_embs = []
        unique_user_ids: set[str] = set()
        
        for item in data:
            # Backend'in dondurdugu DTO: EmbeddingDTO {id, userId, userName, photoId, embedding, dimension}
            person_id = str(item.get("userId", "Unknown"))
            user_name = str(item.get("userName", "Unknown"))
            emb_vector = item.get("embedding", [])
            
            if len(emb_vector) > 0:
                combined_name = f"{person_id}:{user_name}"
                new_names.append(combined_name)
                new_embs.append(np.array(emb_vector, dtype=np.float32))
                unique_user_ids.add(person_id)
                
        if not new_names:
            print("[Backend Client] Uyari: Backend'den bos veri dondu veya format hatali.")
            return None
            
        # Npz olarak kaydet
        np.savez(db_abs, encodings=new_embs, names=new_names)
        print(f"[Backend Client] {len(unique_user_ids)} personelin yuz verisi guncellendi.")
        
        # Bellege yukle ve geri don
        return FaceRecognizer.load_db(db_abs)

    except requests.RequestException as e:
        print(f"[Backend Client] Baglanti hatasi (fetch_embeddings): {e}")
        return None
    except Exception as e:
        print(f"[Backend Client] Beklenmeyen hata (fetch_embeddings): {e}")
        return None


def send_access_log(person_id: str) -> None:
    """
    Sends an instant access log to the backend. 
    If offline, saves it locally.
    """
    url = f"{BACKEND_BASE_URL}/api/access-logs"
    payload = {
        "userId": int(person_id) if person_id.isdigit() else None,
        "accessType": "AUTHORIZED", # RPi'den gecenler onayli sayilir (ya da is mantiginiza gore degistirin)
        "details": f"RPi cihazindan yuz tanima ile gecis",
        "deviceId": "RPI_MAIN_DOOR",
        "accessTime": _get_current_time_iso()
    }
    
    try:
        response = requests.post(url, json=payload, timeout=3)
        response.raise_for_status()
        print(f"[Backend Client] Anlik gecis logu gonderildi: {person_id}")
        
        # Basarili baglanti kurulduysa offline loglari da senkronize etmeyi dene
        sync_offline_logs()
        
    except requests.RequestException:
        print(f"[Backend Client] Baglanti yok. Log cevrimdisi kaydediliyor: {person_id}")
        _save_log_offline(payload)


def send_unknown_access_log(score: float | None = None, track_id: int | None = None) -> None:
    """
    Sends an access log for an unrecognized face detected on RPi.
    If offline, saves it locally (same as send_access_log).
    """
    url = f"{BACKEND_BASE_URL}/api/access-logs"
    score_text = ""
    if score is not None and not (isinstance(score, float) and np.isnan(score)):
        score_text = f" (Skor: {score:.3f})"
    track_text = f" track:{track_id}" if track_id is not None else ""
    payload = {
        "userId": None,
        "accessType": "UNKNOWN",
        "details": f"RPi cihazindan taninmayan yuz algilandi{track_text}{score_text}",
        "deviceId": "RPI_MAIN_DOOR",
        "accessTime": _get_current_time_iso(),
    }

    try:
        response = requests.post(url, json=payload, timeout=3)
        response.raise_for_status()
        print("[Backend Client] Taninmayan yuz logu gonderildi")

        sync_offline_logs()

    except requests.RequestException:
        print("[Backend Client] Baglanti yok. Taninmayan yuz logu cevrimdisi kaydediliyor")
        _save_log_offline(payload)


def _save_log_offline(payload: dict[str, Any]) -> None:
    """Thread-safe append log to offline storage."""
    with _log_lock:
        logs = []
        if os.path.exists(OFFLINE_LOGS_FILE):
            try:
                with open(OFFLINE_LOGS_FILE, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            except json.JSONDecodeError:
                pass
                
        logs.append(payload)
        
        with open(OFFLINE_LOGS_FILE, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)


def sync_offline_logs() -> None:
    """
    Reads offline logs and sends them to the batch endpoint.
    If successful, clears the offline logs file.
    """
    with _log_lock:
        if not os.path.exists(OFFLINE_LOGS_FILE):
            return
            
        try:
            with open(OFFLINE_LOGS_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            logs = []
            
        if not logs:
            return
            
    # Send batch request
    url = f"{BACKEND_BASE_URL}/api/access-logs/batch"
    try:
        print(f"[Backend Client] {len(logs)} adet cevrimdisi log senkronize ediliyor...")
        response = requests.post(url, json=logs, timeout=5)
        response.raise_for_status()
        
        print("[Backend Client] Cevrimdisi loglar basariyla senkronize edildi.")
        
        # Dosyayi temizle
        with _log_lock:
            if os.path.exists(OFFLINE_LOGS_FILE):
                os.remove(OFFLINE_LOGS_FILE)
                
    except requests.RequestException as e:
        print(f"[Backend Client] Toplu (batch) log gonderimi basarisiz: {e}")

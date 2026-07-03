"""Backend Connection Test Script.

This file simulates requests that the RPi system sends to the Spring Boot backend
(such as fetch embeddings, send logs, etc.) without requiring a camera.
It simplifies development and debugging of backend integrations.
"""

import time
import os
from .backend_client import fetch_and_save_embeddings, send_access_log, sync_offline_logs, _save_log_offline, OFFLINE_LOGS_FILE

def run_tests():
    # Eski hatali testlerden kalan dosyalari temizle
    if os.path.exists(OFFLINE_LOGS_FILE):
        os.remove(OFFLINE_LOGS_FILE)

    print("=== BACKEND COMMUNICATION TEST STARTING ===\n")
    
    # Test 1: Fetch Face Embeddings (GET /api/embedding/all-active)
    print("--- Test 1: Initial Sync (Fetching Face Embeddings) ---")
    print("Description: Fetching registered personnel faces from Spring Boot...")
    # Saves results to "test_known_faces.npz"
    result = fetch_and_save_embeddings("test_known_faces.npz")
    if result is not None:
        embs, names = result
        print(f"-> SUCCESS: Retrieved face data for {len(names)} personnel.")
        print(f"-> Received names: {names}")
    else:
        print("-> FAILED: Data could not be retrieved. Ensure backend is running and IP is correct.")
        
    print("\n--------------------------------------------------------------\n")

    # Test 2: Instant Access Log (POST /api/access-logs)
    test_personnel_id = "Ahmet-Yilmaz-123"
    print("--- Test 2: Instant Access Log (Camera Detected Face) ---")
    print(f"Description: Sending access log to backend for personnel ID '{test_personnel_id}'...")
    send_access_log(test_personnel_id)
    print("Note: If console output shows 'Instant access log sent', the request reached Spring Boot successfully.")
    
    print("\n--------------------------------------------------------------\n")
    
    # Test 3: Offline Logs Synchronization (Offline -> Online)
    print("--- Test 3: Offline Logs Synchronization (Offline -> Online) ---")
    print("Description: Simulating disconnected internet scenario...")
    
    # Manually cache two offline logs for simulation
    _save_log_offline({
        "userId": None, 
        "accessType": "AUTHORIZED", 
        "details": "Offline Test Mehmet",
        "deviceId": "RPI_TEST",
        "accessTime": "2026-05-18T12:00:00"
    })
    _save_log_offline({
        "userId": None, 
        "accessType": "AUTHORIZED", 
        "details": "Offline Test Ayse",
        "deviceId": "RPI_TEST",
        "accessTime": "2026-05-18T12:05:00"
    })
    
    print("-> Created 2 mock offline logs.")
    print("-> Restoring simulated connection and trying batch synchronization...")
    time.sleep(1) # Short sleep to make output readable
    sync_offline_logs()
    print("Note: If console output shows 'Offline logs successfully synchronized', Spring Boot successfully processed the batch logs.")
    
    print("\n=== TEST COMPLETED ===")

if __name__ == "__main__":
    run_tests()

# RPi and Backend Integration: Real-World Test Scenarios

This document contains real-world testing scenarios designed to verify the integration and communication between the Raspberry Pi (RPi) edge face recognition system and the Spring Boot backend server. These tests must be executed in a live environment where both systems (the `run_system.py` script on RPi and the Spring Boot application) are running on the same network.

## Prerequisites

1. Ensure the Spring Boot backend application is running and accessible.
2. Ensure the RPi (or the testing MacBook/Windows PC) is connected to the same local network as the backend server.
3. Verify that `BACKEND_BASE_URL` in `src/backend_client.py` is configured with the correct IP and port of the backend server (e.g., `http://192.168.1.100:8080`).

---

### Scenario 1: Initial Boot and Database Synchronization
**Objective:** Verify that the RPi pulls the latest personnel embedding database from the backend at startup.

1. Start the face recognition application on the RPi:
   ```bash
   python -m src.run_system
   ```
2. Monitor the backend server logs; a `GET` request should be logged at the `/api/embedding/all-active` endpoint.
3. Look for the following console logs on the RPi:
   ```
   [Backend Client] Fetching face database: http://<backend_ip>:8080/api/embedding/all-active
   [Backend Client] Updated face database for X personnel.
   ```
4. **Verification:** Confirm that the `.npz` database file (e.g., `known_faces_embeddings.npz`) specified in the config is created or its modification timestamp is updated in the application directory.

---

### Scenario 2: Real-time Verification and Log Transmission (Online Mode)
**Objective:** Verify that verified face logs are immediately sent to the backend database when the system is online.

1. Keep both the RPi system and backend server online.
2. Show an enrolled face to the camera.
3. Verify the RPi console logs show:
   ```
   [SUCCESS] {Name} (AUTHORIZED) detected! (Score: ...)
   [Backend Client] Instant access log sent: {PersonId}
   ```
4. **Verification:** Query the backend database or logs to confirm a new access log record is written with status `AUTHORIZED` containing the correct timestamp.

---

### Scenario 3: Unregistered (Unknown) Face Detection
**Objective:** Verify that unregistered faces are handled appropriately without spamming unnecessary network calls.

1. Present an unregistered face (not enrolled in the database) to the camera.
2. Verify the RPi console logs show:
   ```
   [WARNING] Unrecognized face track:X (log sending)
   [Backend Client] Unrecognized face log sent
   ```
3. **Verification:** Confirm that an access log with status `UNKNOWN` is recorded in the backend with a null user ID.

---

### Scenario 4: Network Outage and Offline Log Caching
**Objective:** Verify that access logs are cached locally in the offline storage when the backend connection is lost.

1. Stop the backend Spring Boot server or disconnect the network interface of the RPi.
2. Show an enrolled face to the camera.
3. Verify the face is recognized successfully.
4. Verify the connection timeout/refusal triggers the following RPi console logs:
   ```
   [Backend Client] Connection offline. Logging locally for: {PersonId}
   ```
5. Detect an enrolled or unrecognized face a few more times to generate multiple cached records.
6. **Verification:** Verify that the file `offline_logs.json` is created in the RPi directory and contains the pending logs as a JSON array.

---

### Scenario 5: Connection Recovery and Batch Synchronization
**Objective:** Verify that cached offline logs are synchronized in bulk with the backend as soon as connectivity is restored.

1. Restart the backend Spring Boot server or restore the network interface.
2. Show a registered face to the camera to trigger a recognition event.
3. The RPi will first send the instant access log, recognize the active connection, and then sync the queue:
   ```
   [Backend Client] Instant access log sent: {PersonId}
   [Backend Client] Synchronizing X offline logs...
   [Backend Client] Offline logs successfully synchronized.
   ```
4. **Verification:**
   - Verify that the backend processed the bulk records at `/api/access-logs/batch`.
   - Confirm that all offline logs are recorded in the database with their original timestamps.
   - Verify that the local `offline_logs.json` cache file has been deleted.

---

### Scenario 6: Database Reload Webhook
**Objective:** Verify that the backend can push a webhook request to force the RPi to refresh its local embeddings cache without restarting the main loop.

1. Add or delete a personnel photo in the backend administrative panel.
2. The backend should trigger a `POST` request to the RPi at `http://<RPI_IP>:8000/reload`.
3. Check the RPi console logs. You should see FastAPI process the request and launch a background reload task:
   ```
   INFO:     192.168.1.100:54321 - "POST /reload HTTP/1.1" 200 OK
   [Backend Client] Fetching face database: http://...
   [API] Database cache successfully updated!
   ```
4. **Verification:** Present the newly enrolled person to the camera; they should be recognized immediately without a system restart.

---

### Scenario 7: Vector Generation Endpoint (Generate API)
**Objective:** Verify that the backend can request the RPi to process an uploaded photo and return a 512-dimensional vector (embedding) for enrollment.

1. Send a test HTTP request from a tool like Postman:
   - **Method:** `POST`
   - **URL:** `http://<RPI_IP>:8000/generate`
   - **Body:** `form-data` containing key `file` (File type) and value (a face image, e.g., `.jpg` or `.png`).
2. **Verification:** Confirm that the RPi returns a `200 OK` with a JSON payload containing the 512-dimensional vector:
   ```json
   {
     "embedding": [0.0123, -0.0456, ..., 0.089]
   }
   ```
3. **Error Handling Verification:** Upload a photo containing no faces (e.g., a landscape). Verify that the API returns `400 Bad Request` with:
   ```json
   {
     "detail": "No face detected in the image."
   }
   ```

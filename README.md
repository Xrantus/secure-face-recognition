# SecureFace: Robust Face Recognition for Surveillance Cameras

SecureFace is an autonomous, contactless, and offline-capable facial recognition system designed for edge-based real-time surveillance and secure access control. Developed as a Senior Design Thesis Project (CENG415) at İzmir Institute of Technology, this repository implements the modular Python-based edge software deployed on autonomous edge units (e.g., Raspberry Pi 5 / PC / macOS) integrated with a centralized Spring Boot backend server.

---

## 🚀 Key Features

- **Distributed Edge-Server Architecture:** Heavy compute tasks (face detection and feature extraction) run locally on edge devices, while synchronization, administrative controls, and persistent logging are handled by a central server.
- **Deep Learning Pipeline:** Employs state-of-the-art models for face detection (YOLOv11 optimized to INT8) and high-dimensional biometric embedding extraction (InsightFace).
- **Proximity Sensor Gating:** Integrates an HC-SR04 ultrasonic sensor to activate the neural pipeline only when a user is within range (e.g., < 100 cm), saving CPU cycles and reducing heat.
- **Resilient & Offline-First:** Recognizes users locally from a cache database (`.npz`) during network outages. Logs are cached locally in `offline_logs.json` and synchronized in bulk once the backend connection is restored.
- **Interactive UI Dashboard:** A custom OpenCV HUD rendering real-time camera feeds, proximity sensor readings, last recognized face badge, access status levels (Authorized, Denied, Unknown), and access logs history.
- **Biometric Benchmarking Suite:** Modules to sweep resolutions, evaluate embedding models, optimize similarity thresholds (EER, F1, Youden's J), and compare metrics (Cosine, Euclidean, Manhattan, Mahalanobis).

---

## 📂 Project Structure

```
.
├── docs/                        # Academic term reports and testing manuals
│   ├── thesis_term_report.pdf   # Complete senior design term report
│   ├── thesis_testing.md        # Detailed optimization and benchmark guide
│   ├── rpi_backend_test_scenarios.md # Step-by-step physical integration scenarios
│   └── architecture_and_flows.md # System flowcharts and lock mechanisms
│
├── src/                         # Core Python package source code
│   ├── benchmarks/              # System evaluation and analysis scripts
│   │   ├── benchmark_metrics.py     # Similarity metrics sweeper (FAR/TAR)
│   │   ├── benchmark_models.py      # Latency & throughput benchmark per model
│   │   ├── benchmark_resolutions.py # Resolution resolution evaluations
│   │   ├── benchmark_vggface2_recognition.py # Benchmark on VGGFace2 dataset
│   │   ├── benchmark_video.py       # YOLO baseline vs. INT8 model comparison
│   │   ├── compare_models_visual.py # Visual comparison of detection models
│   │   ├── compare_metrics.py       # Distance metrics performance comparator
│   │   └── find_best_thresholds.py  # Sweeps data to compute optimal thresholds
│   │
│   ├── config.py                # Hardware, model, metrics central config
│   ├── run_system.py            # Production entrypoint (FastAPI Server + Camera)
│   ├── main.py                  # Standalone live camera runner
│   ├── face_ui.py               # Dashboard renderer and GUI drawing modules
│   ├── face_detector.py         # YOLO bounding box detection class
│   ├── face_recognizer.py       # InsightFace alignment & embedding generator
│   ├── distance_sensor.py       # HC-SR04 ultrasonic distance sensor listener
│   ├── api_server.py            # FastAPI endpoints (/reload, /generate)
│   ├── backend_client.py        # Spring Boot client (sync & logging requests)
│   ├── access_log_policy.py     # Throttle logs cooldown policy per user/track
│   └── test_backend.py          # Backend mock requests validation script
│
├── db_images/                   # Folder for local offline enrollment
├── test_videos/                 # Mock video files for testing/benchmarks
├── yolo11-models/               # ONNX quantized/unquantized detection models
├── models/                      # InsightFace embedding model weights
├── requirements.txt             # Python packages requirements
└── LICENSE                      # Project license
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+ (Python 3.11 recommended)
- OpenCV, NumPy, Ultralytics YOLO, InsightFace
- On Raspberry Pi: `picamera2` and `lgpio` library

### Setup Virtual Environment
In the repository root directory:
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
python -m pip install -U pip
pip install -r requirements.txt
```

Verify that the system imports correctly:
```bash
python -c "from src.face_recognizer import FaceRecognizer; print('SecureFace modules initialized successfully!')"
```

---

## ⚙️ CENTRALIZED CONFIGURATION (`src/config.py`)

All thresholds, ports, and models are hardware-agnostic and controlled in a single configuration file:
1. **`HARDWARE_ENV`**: Selects camera reading backend (`"MAC"`, `"WIN"` uses OpenCV `VideoCapture`; `"RPI"` uses `Picamera2` overlay).
2. **`ModelConfig`**: Adjusts YOLO ONNX weights path, `buffalo_s` parameters, detection thresholds, skipped frames, and database path.
3. **`MetricConfig`**: Sweeps similarity metric between `"cosine"` (dot product similarity) and `"euclidean"` (L2 distance), and defines corresponding validation thresholds.
4. **`ProximityConfig`**: Sets HC-SR04 pins (e.g., TRIG=GPIO17, ECHO=GPIO27), distance parameters, poll interval, and error fallbacks.

## 🚀 Running the System

SecureFace supports running in two configurations: **Standalone (Offline) Mode** and **Centralized Backend Mode**. Both configurations leverage **`src.run_system`** as the primary execution engine to render the interactive UI Dashboard, with automatic offline fallback if the backend is unreachable.

### Step 1: Initialize the Local Biometric Database (Required)
Before running either mode, populate the local enrollment photo directories:
1. Create directories under `db_images/<PersonName>/` and upload face photos:
   ```
   db_images/
   ├── Buket/
   │   ├── photo1.jpg
   │   └── photo2.png
   └── Kerem/
       └── photo1.jpg
   ```
2. Process and vectorize the database locally:
   ```bash
   python -m src.db_create
   ```
   This extracts embeddings, calculates average vectors, and caches them to `known_faces_embeddings.npz`.

---

### Step 2: Run the UI Dashboard

#### Option A: Standalone (Offline) Mode [RECOMMENDED for Local Testing]
Run the complete system offline with the premium UI Dashboard rendering (circular face crop, logs database HUD, proximity sensor graphs). It automatically detects if the central server is offline and uses the local `.npz` database:
```bash
python -m src.run_system
```
*To run using a mock video file instead of a webcam:*
```bash
python -m src.run_system --video tar1.h264
```

#### Option B: Centralized Backend Mode (Online)
Integrates with the Spring Boot server for webhooks sync, real-time logging, and remote reload:
1. Configure the server IP/Port in `src/backend_client.py`:
   ```python
   BACKEND_BASE_URL = "http://192.168.1.100:8080"
   ```
2. Start the system:
   ```bash
   python -m src.run_system
   ```

---

### 🔍 Debug / Lightweight CLI Mode
For simple webcam verification with raw OpenCV bounding boxes (no advanced HUD dashboard/Figma-like panel):
- To run with live webcam:
  ```bash
  python -m src.main
  ```
- To run with mock video file:
  ```bash
  python -m src.main --video tar1.h264
  ```

---

## 🎓 Experimental Results & Academic Evaluation

SecureFace was evaluated under rigorous biometric benchmarks to determine the optimal configuration for real-time edge processing on resource-constrained hardware. Below are the consolidated empirical findings from the senior thesis report.

### 1. Face Detection Performance (YOLO11n ONNX)
Trained and validated on a filtered subset of the **WIDERFace** dataset (comprising 9,980 training and 1,761 validation images):

| Model Format | Precision | Recall | F1-Score | mAP@0.5 | Inference Time | Model Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLO11n FP32** | 0.9710 ± 0.0057 | 0.8868 ± 0.0092 | 0.9270 ± 0.0066 | 0.8853 ± 0.0083 | **3.23 ms** | 5.20 MB |
| **YOLO11n INT8** | **0.9712 ± 0.0061** | **0.8858 ± 0.0103** | **0.9265 ± 0.0071** | **0.8848 ± 0.0081** | 3.87 ms | **2.87 MB** |

> [!NOTE]
> The **INT8 quantized model** was selected as the final deployment format because it reduces storage and memory overhead by **44.8%** with negligible accuracy degradation.

### 2. Face Embedding Extraction Models
Evaluated using a baseline video to evaluate extraction latency, system throughput, and True Acceptance Rate (TAR) at 0.00% False Acceptance Rate (FAR):

| Model Pack | Throughput (FPS) | Latency (ms) | TAR (%) | FAR (%) |
| :--- | :---: | :---: | :---: | :---: |
| **buffalo_s** | **33.3** | **30.04** | 81.59% | 0.00% |
| **buffalo_l** | 13.1 | 76.45 | 93.14% | 0.00% |
| **antelopev2** | 9.5 | 104.92 | 95.40% | 0.00% |

> [!TIP]
> **`buffalo_s`** was chosen as the default model provider since it yields a throughput of **33.3 FPS** and a latency of only **30 ms** per face, matching the real-time requirements of edge units.

### 3. Biometric Similarity Metrics
Comparison of different distance/similarity metrics using the `buffalo_s` L2-normalized vectors:

| Distance Metric | Area Under Curve (AUC) | Equal Error Rate (EER) | EER-Threshold | FDR |
| :--- | :---: | :---: | :---: | :---: |
| **Cosine Similarity** | **0.9999** | **0.0047** | **0.3001** | **32.750** |
| **Pearson Correlation** | 0.9999 | 0.0047 | 0.2993 | 32.752 |
| **Euclidean Distance (L2)** | 0.9999 | 0.0047 | 1.1831 | 26.261 |
| **Manhattan Distance (L1)** | 0.9999 | 0.0062 | 21.4972 | 25.969 |
| **Mahalanobis Distance** | 0.9999 | 0.0093 | 3.5175 | 23.898 |

* **Final Selection:** Cosine Similarity was chosen due to its high Fisher Discriminant Ratio (FDR) and computational efficiency on L2-normalized vectors.
* **Operational Decision Threshold:** While statistical metrics suggest a threshold of 0.30 to 0.40, the system defaults to **0.50** for a strict, security-oriented profile resulting in **0.00% FAR** during access control.

---

## 📊 Evaluation & Benchmarking

To benchmark accuracy, FPS, and latencies across different models, distance metrics, and resolutions, run the scripts located inside `src/benchmarks/`:

### 1. Model Latency & Throughput Benchmark
Evaluate different embedding models (e.g., `buffalo_s`, `buffalo_l`, `antelopev2`) on a sample video:
```bash
python -m src.benchmarks.benchmark_models --video test_videos/sample.mp4 --video_type authorized --model buffalo_s
```

### 2. Resolution Performance Sweep
Determine how input resolution impacts YOLO inference latency, detection accuracy (F1, mAP), and downstream recognition success:
```bash
python -m src.benchmarks.benchmark_resolutions --video test_videos/sample.mp4
```

### 3. Metric Decision Optimization
Determine the best similarity threshold for Cosine or L2 distance by sweeping confidence levels to identify the Equal Error Rate (EER) or Youden's J:
```bash
python -m src.benchmarks.benchmark_metrics --video test_videos/sample.mp4 --video_type authorized
```

---

## 📜 License
Licensed under the [MIT License](LICENSE).

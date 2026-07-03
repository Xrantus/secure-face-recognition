# Smart Access Control System: Performance and Optimization Test Plan

This testing plan is structured into four main phases to evaluate both the software accuracy and hardware efficiency of the developed face recognition system on the edge device (Raspberry Pi 5).

---

## 1. Preparing the Benchmark Video
Rather than using generic open-source footage, it is highly recommended to record a custom benchmark video that mimics the actual deployment environment. Presenting a custom-made scenario looks much more professional in a thesis defense.

### Recommended Video Recording Scenario (1 - 1.5 Minutes, MP4):
- **0–15s**: Person A (enrolled in DB) walks toward the camera, looks in different directions (left, right, up, down), and exits.
- **15–30s**: Person B (enrolled in DB) repeats the process but wears accessories like glasses or a hat.
- **30–45s**: An unregistered stranger (Unknown) walks up and looks at the camera (to prove the system rejects unauthorized access).
- **45–60s**: Enrolled and unregistered persons enter the frame simultaneously (multi-face detection and pipeline performance degradation test).
- **60–75s**: Ambient lighting changes abruptly (e.g., toggling a light switch) while a person walks quickly past the camera (motion blur and lighting robustness test).

---

## 2. Modular Architecture & Directory Structure
Structuring monolithic scripts into clean, object-oriented classes isolates tasks and simplifies independent optimization testing.

### Proposed Code Directory Layout:
```
/thesis-project-image
│
├── requirements.txt            # System dependencies
│
├── src/                        # Main source code directory
│   ├── config.py               # Thresholds, paths, and settings
│   ├── face_detector.py        # YOLO bounding box detection wrapper
│   ├── face_recognizer.py      # InsightFace embedding & comparison wrapper
│   ├── db_create.py            # Local offline enrollment utility
│   ├── run_system.py           # Production edge system runner (FastAPI + Camera)
│   └── benchmarks/             # Benchmarking tools
│       ├── compare_metrics.py  # Evaluates different distance metrics
│       └── compare_models.py   # Compares different detection models
```

---

## 3. Four-Phase Optimization & Benchmark Plan

### Phase 1: Face Detection (YOLO) Baseline Metrics
Evaluate the baseline model without any hardware-specific compression (Unquantized FP32 or FP16).
- **Target Model**: Trained YOLOv11 Face model.
- **Environment**: PC / GPU Environment (to find maximum potential capacity).
- **Metrics to Measure**:
  - **Precision**: Of all predicted faces, how many are actually faces?
  - **Recall**: What percentage of ground-truth faces did the model detect?
  - **F1-Score**: The harmonic mean of Precision and Recall.
  - **mAP@0.5**: Standard object detection average precision.

### Phase 2: Model Optimization (Pruning & Quantization)
Compress and accelerate the model using pruning and quantization. Re-evaluate metrics after each step to compare with the Phase 1 Baseline.
- **Step 2.1 - Pruning**: Reducing model density (sparsity) and observing mAP drops.
- **Step 2.2 - Quantization**: Converting model weights to 8-bit integers (INT8).
- **Evaluation**: Generate trade-off graphs (Accuracy vs. Inference Speed) comparing FP32 (Original), Pruned, and INT8 models.

### Phase 3: Face Recognition Pipeline Customizations
Benchmark the InsightFace downstream pipeline under various configurations.
- **Embedding Model Comparison**: Compare recognition accuracy and inference times of different weight configurations (e.g., `buffalo_s` vs. `buffalo_l` or `MobileFaceNet`).
- **Input Size Experiments**: Analyze the impact of face crop size and InsightFace's `det_size` parameter (e.g., 160x160 vs. 320x320 pixels) on inference latency and accuracy.
- **Distance Metrics**: Compare Cosine Similarity and Euclidean Distance (L2 Norm) to determine which metric yields a more stable decision threshold:
  $$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

### Phase 4: Edge Deployment & Hardware Benchmarks
Deploy models on Raspberry Pi 5 to record actual real-world hardware metrics:
- **Inference Time (Latency)**: Time taken to process a single frame (ms).
- **FPS (Frames Per Second)**: System throughput (Detection FPS vs. End-to-End Pipeline FPS).
- **Resource Utilization**: CPU Usage (%) and Memory Footprint (RAM).
- **Thermal Throttling**: The impact of processor heat on throughput under continuous workload.

---

## 4. Verification & Optimization Checklist

- **Embedding Model Evaluation**
  - [ ] **Model 1: `buffalo_s`**: The lightweight default model optimized for edge devices. (Baseline).
  - [ ] **Model 2: `buffalo_l`**: Deep ResNet-based model. Used as a high-accuracy reference point to measure the accuracy/speed trade-off against `buffalo_s`.
  - [ ] **Model 3: `MobileFaceNet`**: Ultra-lightweight embedding network specifically designed for embedded devices.

- **Input Dimension Benchmarks**
  - [ ] **Dimension 1 (112x112)**: Standard InsightFace alignment crop. Balanced speed/accuracy.
  - [ ] **Dimension 2 (160x160)**: Larger resolution to see if retaining finer details improves embedding accuracy.
  - [ ] **Dimension 3 (320x320)**: High resolution. Used to locate computational bottlenecks on the Raspberry Pi.

- **Distance & Similarity Metric Validation**
  - [ ] **Cosine Similarity**: Measures the angular distance between vectors (currently implemented).
  - [ ] **Euclidean Distance (L2)**: Measures straight-line distance in Euclidean space.
  - [ ] **Threshold Calibration**: Find the optimal acceptance thresholds (e.g., Cosine $\geq$ 0.50, L2 $\leq$ 1.00) that minimize unauthorized intrusions.

- **Biometric Performance Calculations**
  - [ ] **TAR (True Acceptance Rate)**: Proportion of registered users correctly authorized.
  - [ ] **FAR (False Acceptance Rate)**: Proportion of unregistered intruders incorrectly authorized (must be as close to 0.0 as possible).
  - [ ] **EER (Equal Error Rate)**: The threshold where FAR equals FRR (False Rejection Rate). Lower is better.
  - [ ] **Latency Logs**: Log average embedding extraction times in milliseconds for all models and resolutions on the Raspberry Pi 5.

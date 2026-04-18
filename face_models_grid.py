"""
Ayni kamera karesini birden fazla YOLO modeliyle isler; grid gosterir.
Kiyaslama yalnizca veritabanindaki hedef kisi (varsayilan: Kerem) ile kosinus benzerligi uzerinden yapilir:
her modelin YOLO kutusundan InsightFace embedding -> Kerem vektoru ile dot product.

Cikis: face_models_grid_log.jsonl, face_models_grid_summary.txt (Kerem ortalamalari)
"""
import json
import os
import threading
import time
from datetime import datetime, timezone

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from ultralytics import YOLO

# --- Kiyaslama hedefi (db-images klasor adi / known_faces_embeddings.npz names) ---
COMPARE_TARGET_NAME = "kerem"
LANDMARK_PAD = 0.20
MIN_FACE_ROI = 35

# (panel basligi, yol, ultralytics imgsz)
MODEL_ENTRIES = [
    ("widerface FP32", "face_yolo11_widerface_best.onnx", 640),
    ("widerface INT8", "face_yolo11_widerface_best_int8.onnx", 320),
    ("widerface INT8 OPT", "face_yolo11_widerface_best_int8_OPTIMIZED.onnx", 320),
    ("widerface INT8 alt", "face_yolo11_widerface_int8.onnx", 640),
    ("face best ONNX", "face_yolo11_best.onnx", 320),
    ("face best INT8", "face_yolo11_best_int8.onnx", 320),
    ("face best PT", "face_yolo11_best.pt", 320),
    ("widerface PT", "face_yolo11_widerface_best.pt", 640),
]

FRAME_SKIP = 2
YOLO_PRED_CONF = 0.01
YOLO_IOU = 0.45
MAX_DET = 100
CAM_INDEX = 0

WINDOW_TITLE = f"YOLO grid — sadece {COMPARE_TARGET_NAME} benzerligi"
GRID_MAX_W = 1600
GRID_MAX_H = 900
HEADER_H = 36
BOX_COLORS = [
    (0, 255, 100),
    (255, 180, 0),
    (255, 80, 180),
    (0, 200, 255),
    (180, 255, 255),
    (100, 100, 255),
]

# --- Log dosyalari ---
LOG_JSONL = "face_models_grid_log.jsonl"
SUMMARY_TXT = "face_models_grid_summary.txt"

DB_PATH = "known_faces_embeddings.npz"
RECOG_THRESHOLD = 0.50
DET_SIZE = (320, 320)


def _landmarks_list(kpss):
    if kpss is None:
        return []
    if isinstance(kpss, np.ndarray):
        if kpss.size == 0:
            return []
        if kpss.ndim == 2 and kpss.shape == (5, 2):
            return [kpss]
        if kpss.ndim == 3 and kpss.shape[1:] == (5, 2):
            return [kpss[i] for i in range(kpss.shape[0])]
        return []
    return [np.asarray(k) for k in kpss] if kpss else []


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _grid_layout(n: int):
    if n <= 0:
        return 1, 1
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def _paint_boxes(img: np.ndarray, result, color: tuple) -> tuple[int, int, float]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return 0, 0, 0.0
    n_raw = len(boxes)
    drawn = 0
    best = 0.0
    for i in range(n_raw):
        conf = float(boxes.conf[i])
        best = max(best, conf)
        if conf < YOLO_PRED_CONF:
            continue
        x1, y1, x2, y2 = map(int, boxes.xyxy[i])
        H, W = img.shape[:2]
        x1 = max(0, min(x1, W - 1))
        x2 = max(0, min(x2, W - 1))
        y1 = max(0, min(y1, H - 1))
        y2 = max(0, min(y2, H - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        drawn += 1
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            f"{conf:.2f}",
            (x1, max(22, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return drawn, n_raw, best


def _detection_list(result, yolo: YOLO) -> list[dict]:
    """YOLO sonucundan sinif adi + conf listesi (ozet)."""
    out = []
    names = getattr(yolo, "names", None) or {}
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return out
    for i in range(len(boxes)):
        cid = int(boxes.cls[i])
        conf = float(boxes.conf[i])
        label = names.get(cid, str(cid)) if isinstance(names, dict) else str(cid)
        out.append({"class": str(label), "conf": round(conf, 4)})
    return out[:50]


def _kerem_reference_embedding(db_embs: np.ndarray, db_names) -> np.ndarray | None:
    """Ayni isimli tum kayitlarin ortalama embedding'i (L2 normalize)."""
    name_l = COMPARE_TARGET_NAME.strip().lower()
    idx = [i for i, n in enumerate(db_names) if str(n).strip().lower() == name_l]
    if not idx:
        return None
    v = np.mean(db_embs[idx], axis=0)
    nrm = np.linalg.norm(v)
    return (v / nrm).astype(np.float32) if nrm > 0 else None


def _kerem_similarity_from_yolo_box(
    frame_bgr: np.ndarray, result, app: FaceAnalysis, kerem_emb: np.ndarray
) -> dict:
    """En buyuk gecerli YOLO kutusundan embedding; sadece Kerem ile kosinus."""
    out: dict = {"kerem_cosine": None, "kerem_status": "no_yolo", "yolo_conf_best": None}
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return out

    H, W = frame_bgr.shape[:2]
    best_i = -1
    best_area = 0
    for i in range(len(boxes)):
        conf = float(boxes.conf[i])
        if conf < YOLO_PRED_CONF:
            continue
        x1, y1, x2, y2 = map(int, boxes.xyxy[i])
        x1, x2 = clamp(x1, 0, W - 1), clamp(x2, 0, W - 1)
        y1, y2 = clamp(y1, 0, H - 1), clamp(y2, 0, H - 1)
        if x2 <= x1 or y2 <= y1:
            continue
        fw, fh = x2 - x1, y2 - y1
        if min(fw, fh) < MIN_FACE_ROI:
            continue
        area = fw * fh
        if area > best_area:
            best_area = area
            best_i = i

    if best_i < 0:
        out["kerem_status"] = "no_valid_box"
        return out

    out["yolo_conf_best"] = round(float(boxes.conf[best_i]), 4)
    x1, y1, x2, y2 = map(int, boxes.xyxy[best_i])
    x1, x2 = clamp(x1, 0, W - 1), clamp(x2, 0, W - 1)
    y1, y2 = clamp(y1, 0, H - 1), clamp(y2, 0, H - 1)
    face_w, face_h = x2 - x1, y2 - y1
    pw = int(face_w * LANDMARK_PAD)
    ph = int(face_h * LANDMARK_PAD)
    roi = frame_bgr[max(0, y1 - ph) : min(H, y2 + ph), max(0, x1 - pw) : min(W, x2 + pw)]
    if roi.size == 0:
        out["kerem_status"] = "empty_roi"
        return out

    try:
        _, kpss = app.det_model.detect(roi, max_num=1, metric="default")
        lm_list = _landmarks_list(kpss)
        if not lm_list:
            out["kerem_status"] = "no_landmark"
            return out
        aligned = face_align.norm_crop(roi, landmark=lm_list[0])
        emb = app.models["recognition"].get_feat(aligned)[0]
        emb = emb / np.linalg.norm(emb)
        cos = float(np.dot(emb, kerem_emb))
        out["kerem_cosine"] = round(cos, 5)
        out["kerem_status"] = "ok"
        out["kerem_tantildi"] = cos >= RECOG_THRESHOLD
    except Exception as e:
        out["kerem_status"] = f"err:{e}"

    return out


def _make_tile(
    frame_bgr: np.ndarray,
    title: str,
    yolo: YOLO,
    imgsz: int,
    color: tuple,
    cell_w: int,
    cell_h: int,
    app: FaceAnalysis | None,
    kerem_emb: np.ndarray | None,
) -> tuple[np.ndarray, dict]:
    panel = frame_bgr.copy()
    t0 = time.perf_counter()
    stats: dict = {
        "panel_title": title,
        "n_raw": 0,
        "drawn": 0,
        "max_conf": 0.0,
        "infer_ms": 0.0,
        "detections": [],
        "error": None,
        "kerem_cosine": None,
        "kerem_status": None,
        "kerem_tantildi": None,
    }
    try:
        r = yolo.predict(panel, imgsz=imgsz, conf=YOLO_PRED_CONF, iou=YOLO_IOU, max_det=MAX_DET, verbose=False)[0]
        drawn, n_raw, best = _paint_boxes(panel, r, color)
        stats["n_raw"] = n_raw
        stats["drawn"] = drawn
        stats["max_conf"] = round(best, 5)
        stats["detections"] = _detection_list(r, yolo)
        if app is not None and kerem_emb is not None:
            ks = _kerem_similarity_from_yolo_box(panel, r, app, kerem_emb)
            stats.update(ks)
            kc = ks.get("kerem_cosine")
            if kc is not None and ks.get("kerem_status") == "ok":
                st_ok = ks.get("kerem_tantildi", False)
                col = (0, 255, 80) if st_ok else (0, 165, 255)
                cv2.putText(
                    panel,
                    f"{COMPARE_TARGET_NAME} {kc:.3f}",
                    (8, panel.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    col,
                    2,
                    cv2.LINE_AA,
                )
    except Exception as e:
        cv2.putText(panel, f"HATA: {e}", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        stats["error"] = str(e)
    stats["infer_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

    avail_h = max(60, cell_h - HEADER_H)
    h0, w0 = panel.shape[:2]
    scale = min(cell_w / w0, avail_h / h0)
    nw, nh = int(w0 * scale), int(h0 * scale)
    thumb = cv2.resize(panel, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    out[:] = (40, 40, 45)
    y0 = HEADER_H + (avail_h - nh) // 2
    x0 = (cell_w - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = thumb

    line1 = title[:42] + ("…" if len(title) > 42 else "")
    kc = stats.get("kerem_cosine")
    ks = stats.get("kerem_status")
    if kc is not None:
        td = stats.get("kerem_tantildi")
        line2 = (
            f"{COMPARE_TARGET_NAME} cos={kc:.3f} "
            f"({'OK' if td else 'dusuk'}) | YOLO maxc={stats['max_conf']:.2f} | {stats['infer_ms']:.0f}ms"
        )
    else:
        line2 = f"{COMPARE_TARGET_NAME}: — ({ks}) | YOLO maxc={stats['max_conf']:.3f} | {stats['infer_ms']:.0f}ms"
    cv2.putText(out, line1, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(out, line2, (6, HEADER_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 200, 180), 1, cv2.LINE_AA)
    return out, stats


def _load_models():
    loaded = []
    for title, path, imgsz in MODEL_ENTRIES:
        if not os.path.isfile(path):
            print(f"[atlandi] yok: {path}")
            continue
        try:
            m = YOLO(path, task="detect")
            loaded.append((title, path, m, imgsz))
            print(f"[yuklendi] {title} <- {path} (imgsz={imgsz})")
        except Exception as e:
            print(f"[hata] {path}: {e}")
    return loaded


def _load_recognition():
    if not os.path.isfile(DB_PATH):
        raise SystemExit(f"{DB_PATH} bulunamadi. {COMPARE_TARGET_NAME} kiyasi icin gerekli.")
    try:
        app = FaceAnalysis(name="buffalo_s", root=".", allowed_modules=["detection", "recognition"])
        app.prepare(ctx_id=-1, det_size=DET_SIZE)
        db = np.load(DB_PATH, allow_pickle=True)
        embs = db["encodings"].astype(np.float32)
        names = db["names"]
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        kerem_vec = _kerem_reference_embedding(embs, names)
        if kerem_vec is None:
            raise SystemExit(
                f"Veritabaninda '{COMPARE_TARGET_NAME}' adinda kayit yok (names). "
                "db-images/{isim}/ ile new_db.py calistirin."
            )
        print(f"[kiyaslama] hedef={COMPARE_TARGET_NAME} | {DB_PATH} | {len(names)} kayit")
        return app, embs, names, kerem_vec
    except SystemExit:
        raise
    except Exception as e:
        raise SystemExit(f"Veritabani / InsightFace: {e}") from e


loaded = _load_models()
if not loaded:
    raise SystemExit("Hic model yuklenemedi. MODEL_ENTRIES ve dosya yollarini kontrol edin.")

recog_app, _, _, KEREM_EMB = _load_recognition()

rows, cols = _grid_layout(len(loaded))
cell_w = max(200, GRID_MAX_W // cols)
cell_h = max(180, GRID_MAX_H // rows)

# Oturum istatistikleri (kayit icin)
session_t0 = time.time()
log_sample_idx = 0
sum_ui_fps = 0.0
n_ui_fps = 0
per_model_sum_kerem = [0.0] * len(loaded)
per_model_kerem_samples = [0] * len(loaded)
ema_ui_fps = 0.0
EMA_ALPHA = 0.08

log_f = open(LOG_JSONL, "a", encoding="utf-8")
log_f.write(
    json.dumps(
        {
            "event": "session_start",
            "iso_time": datetime.now(timezone.utc).isoformat(),
            "compare_target": COMPARE_TARGET_NAME,
            "models": [{"title": t, "path": p, "imgsz": iz} for t, p, _, iz in loaded],
        },
        ensure_ascii=False,
    )
    + "\n"
)
log_f.flush()

latest_frame = None
frame_lock = threading.Lock()
running = True


def frame_reader_thread(cap):
    global latest_frame, running
    while running:
        ret, frame = cap.read()
        if not ret:
            running = False
            break
        with frame_lock:
            latest_frame = frame
    cap.release()


cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise SystemExit("Webcam acilamadi.")

reader_t = threading.Thread(target=frame_reader_thread, args=(cap,), daemon=True)
reader_t.start()
while latest_frame is None and running:
    time.sleep(0.02)

frame_counter = 0
fps_t0 = time.time()
fps_n = 0
ui_fps = 0.0
last_grid = None
print(f"Grid: {rows}x{cols} | hedef: {COMPARE_TARGET_NAME} | log: {LOG_JSONL} | Cikis: q")

while running:
    with frame_lock:
        if latest_frame is None:
            time.sleep(0.01)
            continue
        frame = latest_frame.copy()

    frame = cv2.flip(frame, 1)
    fps_n += 1

    if frame_counter % FRAME_SKIP == 0:
        cycle_t0 = time.perf_counter()
        tiles = []
        panel_stats: list[dict] = []
        for idx, (title, _path, yolo, imgsz) in enumerate(loaded):
            color = BOX_COLORS[idx % len(BOX_COLORS)]
            tile, st = _make_tile(
                frame, title, yolo, imgsz, color, cell_w, cell_h, recog_app, KEREM_EMB
            )
            st["model_path"] = _path
            st["imgsz"] = imgsz
            tiles.append(tile)
            panel_stats.append(st)

        row_imgs = []
        for r in range(rows):
            row_tiles = tiles[r * cols : (r + 1) * cols]
            while len(row_tiles) < cols:
                pad = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                cv2.putText(pad, "-", (cell_w // 2 - 5, cell_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 80, 80), 1)
                row_tiles.append(pad)
            row_imgs.append(np.hstack(row_tiles))
        grid = np.vstack(row_imgs)

        gh, gw = grid.shape[:2]
        scale = min(GRID_MAX_W / gw, GRID_MAX_H / gh, 1.0)
        if scale < 1.0:
            grid = cv2.resize(grid, (int(gw * scale), int(gh * scale)), interpolation=cv2.INTER_AREA)

        now = time.time()
        if now - fps_t0 >= 1.0:
            ui_fps = fps_n / (now - fps_t0)
            fps_n = 0
            fps_t0 = now
            sum_ui_fps += ui_fps
            n_ui_fps += 1
            ema_ui_fps = EMA_ALPHA * ui_fps + (1 - EMA_ALPHA) * ema_ui_fps if ema_ui_fps > 0 else ui_fps

        for i, st in enumerate(panel_stats):
            kc = st.get("kerem_cosine")
            if kc is not None and st.get("kerem_status") == "ok":
                per_model_sum_kerem[i] += float(kc)
                per_model_kerem_samples[i] += 1

        grid_cycle_ms = (time.perf_counter() - cycle_t0) * 1000.0

        avg_fps_so_far = sum_ui_fps / max(1, n_ui_fps)
        avg_kerem = [
            per_model_sum_kerem[i] / max(1, per_model_kerem_samples[i]) for i in range(len(loaded))
        ]

        log_sample_idx += 1
        record = {
            "sample": log_sample_idx,
            "compare_target": COMPARE_TARGET_NAME,
            "iso_time": datetime.now(timezone.utc).isoformat(),
            "ui_fps_instant": round(ui_fps, 3),
            "ui_fps_ema": round(ema_ui_fps, 3),
            "avg_ui_fps_so_far": round(avg_fps_so_far, 3),
            "grid_cycle_ms": round(grid_cycle_ms, 2),
            "panels": panel_stats,
            "avg_kerem_cosine_so_far": {
                loaded[i][0]: round(avg_kerem[i], 5) for i in range(len(loaded))
            },
            "kerem_samples_so_far": {
                loaded[i][0]: per_model_kerem_samples[i] for i in range(len(loaded))
            },
        }
        log_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_f.flush()

        cv2.putText(
            grid,
            f"{COMPARE_TARGET_NAME} kiyasi | UI ~{ui_fps:.1f} FPS | {LOG_JSONL}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        last_grid = grid

    if last_grid is not None:
        cv2.imshow(WINDOW_TITLE, last_grid)

    frame_counter += 1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

running = False
reader_t.join(timeout=1.0)
cv2.destroyAllWindows()

elapsed = time.time() - session_t0
avg_fps_final = sum_ui_fps / max(1, n_ui_fps)
avg_kerem_final = [
    per_model_sum_kerem[i] / max(1, per_model_kerem_samples[i]) for i in range(len(loaded))
]
ranked = sorted(
    range(len(loaded)),
    key=lambda i: avg_kerem_final[i] if per_model_kerem_samples[i] > 0 else -1.0,
    reverse=True,
)

log_f.write(
    json.dumps(
        {
            "event": "session_end",
            "compare_target": COMPARE_TARGET_NAME,
            "iso_time": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(elapsed, 2),
            "samples": log_sample_idx,
            "avg_ui_fps": round(avg_fps_final, 3),
            "per_model_avg_kerem_cosine": {
                loaded[i][0]: round(avg_kerem_final[i], 5) for i in range(len(loaded))
            },
            "per_model_kerem_sample_count": {
                loaded[i][0]: per_model_kerem_samples[i] for i in range(len(loaded))
            },
            "ranking_by_kerem": [loaded[i][0] for i in ranked],
        },
        ensure_ascii=False,
    )
    + "\n"
)
log_f.close()

summary_lines = [
    f"Kiyaslama hedefi: {COMPARE_TARGET_NAME} (kosinus benzerligi)",
    f"Oturum suresi (yaklasik): {elapsed:.1f} s",
    f"Kayit ornek sayisi: {log_sample_idx}",
    f"Ortalama UI FPS: {avg_fps_final:.2f}",
    "",
    "Model basina ortalama Kerem benzerligi (gecerli ornek ortalamasi):",
]
for i in range(len(loaded)):
    n_s = per_model_kerem_samples[i]
    ak = avg_kerem_final[i]
    summary_lines.append(f"  - {loaded[i][0]}: {ak:.4f}  (n={n_s} gecerli ornek)")
summary_lines.extend(
    [
        "",
        "Kerem skoruna gore siralama (yuksek iyi):",
        *[f"  {r+1}. {loaded[ranked[r]][0]}" for r in range(len(ranked))],
        "",
        f"Detay: {LOG_JSONL}",
    ]
)
with open(SUMMARY_TXT, "w", encoding="utf-8") as sf:
    sf.write("\n".join(summary_lines) + "\n")

print("Kapatildi.")
print(f"Ozet: {SUMMARY_TXT} | Detay: {LOG_JSONL}")

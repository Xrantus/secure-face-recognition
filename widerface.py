import cv2
import os

from ultralytics import YOLO

MODEL_PATH = "face_yolo11_widerface_best_int8.tflite"  # INT8 model (sabit 640x640 bekliyor)
IMGSZ = 640
CONF = 0.01
IOU = 0.70
CAM_INDEX = 0

if not os.path.exists(MODEL_PATH):
    raise SystemExit(f"Model bulunamadi: {MODEL_PATH}")

print(f"YOLO INT8 ONNX yukleniyor: {MODEL_PATH}")
yolo = YOLO(MODEL_PATH, task="detect")

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise SystemExit("Webcam baslatilamadi.")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)

    results = yolo(frame, imgsz=IMGSZ, conf=CONF, iou=IOU, verbose=False)
    r = results[0]

    if r.boxes is not None and len(r.boxes) > 0:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            conf = float(b.conf[0])
            cls_id = int(b.cls[0])
            name = yolo.names.get(cls_id, str(cls_id)) if hasattr(yolo, "names") else str(cls_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{name} {conf:.2f}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    cv2.imshow("INT8 Face YOLO", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
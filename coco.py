from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from onnx import load as onnx_load
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static


class YOLOCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_paths, input_name, img_size=320, max_samples=64):
        self.input_name = input_name
        self.image_paths = image_paths[:max_samples]
        self.img_size = img_size
        self.index = 0

        # Kalibrasyon gorseli bulunamazsa minimum birkac dummy sample uret.
        self.use_dummy = len(self.image_paths) == 0
        self.dummy_count = 16

    def _preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)   # CHW -> NCHW
        return img

    def get_next(self):
        if self.use_dummy:
            if self.index >= self.dummy_count:
                return None
            self.index += 1
            sample = np.random.rand(1, 3, self.img_size, self.img_size).astype(np.float32)
            return {self.input_name: sample}

        if self.index >= len(self.image_paths):
            return None

        img_path = self.image_paths[self.index]
        self.index += 1
        img = cv2.imread(str(img_path))
        if img is None:
            return self.get_next()
        return {self.input_name: self._preprocess(img)}


# 1) Modeli yukle (eger yolo11n.pt yoksa otomatik indirecek)
model = YOLO("yolo11n.pt")

# 2) Once normal ONNX (FP32) olarak disa aktar (BOYUT 320 OLARAK SABITLENDI)
fp32_path = str(model.export(format="onnx", imgsz=320))

# 3) ONNX Runtime ile QDQ static INT8 quantization uygula
onnx_graph = onnx_load(fp32_path)
input_name = onnx_graph.graph.input[0].name

dataset_dir = Path("db-images")
image_paths = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
    image_paths.extend(dataset_dir.rglob(ext))

# KALIBRASYON VERI OKUYUCUSUNA 320 BOYUTU GONDERILDI
reader = YOLOCalibrationDataReader(image_paths=image_paths, input_name=input_name, img_size=320, max_samples=64)
int8_path = "yolo11n_int8.onnx"

quantize_static(
    model_input=fp32_path,
    model_output=int8_path,
    calibration_data_reader=reader,
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    per_channel=True,
)

print(f"FP32 model: {fp32_path}")
print(f"INT8 model olusturuldu: {int8_path}")
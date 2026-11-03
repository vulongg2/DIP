"""
YOLO Person Detector Wrapper.
Đóng gói mô hình YOLOv8 với cơ chế tự động tìm kiếm weights trong models/ hoặc root.
"""

import os
import torch
from ultralytics import YOLO


def resolve_yolo_weights(weights_name="yolov8s.pt"):
    candidates = [
        weights_name,
        os.path.join("models", weights_name),
        os.path.join("models", "checkpoints", weights_name),
        os.path.join(os.path.dirname(__file__), "..", "..", weights_name),
        os.path.join(os.path.dirname(__file__), "..", "..", "models", weights_name),
    ]
    for p in candidates:
        abs_p = os.path.abspath(p)
        if os.path.exists(abs_p):
            return abs_p
    return weights_name


class YOLODetector:
    def __init__(self, model_name="yolov8s.pt", conf=0.45, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights_path = resolve_yolo_weights(model_name)
        print(f"[*] Nạp mô hình YOLO Detection từ: {weights_path} (Thiết bị: {self.device})")
        self.model = YOLO(weights_path)
        self.conf = conf

    def detect_and_track(self, frame, tracker_cfg=None):
        """
        Thực hiện inference và tracking trên frame.
        Chỉ lọc class 0 (Person).
        """
        results = self.model.track(
            source=frame,
            persist=True,
            classes=[0],  # Chỉ detect người
            conf=self.conf,
            tracker=tracker_cfg,
            verbose=False,
            device=self.device
        )
        return results[0] if results else None

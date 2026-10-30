"""
ReID Feature Extractor Module.
Hỗ trợ 2 Backend:
1. 'custom': Customized Deep Neural Network (CustomReIDNet) với kiến trúc Metric Learning + BNNeck tự xây dựng.
2. 'onnx': Model ONNX có sẵn (yolo26n-reid.onnx).
"""

import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from ultralytics.trackers.utils.reid import ReID

from core.custom_reid_net import CustomReIDNet


class ReIDExtractor:
    def __init__(self, backend="custom", model_path="yolo26n-reid.onnx", device=None):
        self.backend = backend
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if self.backend == "custom":
            # Khởi tạo model tự thiết kế (Customized ReID Net)
            print(f"[*] Khởi tạo Customized ReID Model (Metric Learning + BNNeck) trên {self.device}...")
            self.model = CustomReIDNet(embedding_dim=512, pretrained=True).to(self.device)
            
            # Nếu đã có checkpoint weights đã train thì load vào
            ckpt_path = "custom_reid_net.pth"
            if os.path.exists(ckpt_path):
                try:
                    state_dict = torch.load(ckpt_path, map_location=self.device)
                    self.model.load_state_dict(state_dict, strict=False)
                    print(f"[*] Đã nạp thành công trọng số customized weights từ {ckpt_path}!")
                except Exception as e:
                    print(f"[!] Chú ý: Dùng weights mặc định do không nạp được {ckpt_path}: {e}")

            self.model.eval()

            # Preprocessing chuẩn Person ReID (kích thước 256x128 chuẩn tỷ lệ dáng người)
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # Dùng file ONNX mặc định
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Không tìm thấy model ReID: {model_path}")
            self.encoder = ReID(model_path)
            self.input_size = (128, 256)

    def extract_crop_feature(self, crop_bgr):
        """
        Trích xuất vector embedding L2-normalized từ ảnh crop BGR.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        h, w = crop_bgr.shape[:2]
        if h < 20 or w < 10:
            return None

        if self.backend == "custom":
            # Chuyển BGR sang RGB
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model(tensor)
            feat_np = feat.cpu().numpy().flatten()
            return feat_np
        else:
            img_resized = cv2.resize(crop_bgr, self.input_size)
            box_full = np.array([[0, 0, img_resized.shape[1], img_resized.shape[0]]], dtype=np.float32)
            feat = self.encoder(img_resized, box_full)[0].flatten()
            norm = np.linalg.norm(feat)
            if norm > 1e-8:
                feat = feat / norm
            return feat

    @staticmethod
    def cosine_similarity(feat1, feat2):
        if feat1 is None or feat2 is None:
            return 0.0
        return float(np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-8))

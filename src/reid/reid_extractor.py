"""
ReID Feature Extractor Module.
Hỗ trợ 2 Backend:
1. 'custom': Customized Deep Neural Network (CustomReIDNet) với kiến trúc Metric Learning + BNNeck tự xây dựng.
2. 'onnx': Model ONNX (yolo26n-reid.onnx).
"""

import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from ultralytics.trackers.utils.reid import ReID

from .custom_reid_net import CustomReIDNet


def resolve_model_path(filename):
    """Tìm đường dẫn model linh hoạt (root, models/, hoặc models/checkpoints/)"""
    candidates = [
        filename,
        os.path.join("models", filename),
        os.path.join("models", "checkpoints", filename),
        os.path.join(os.path.dirname(__file__), "..", "..", filename),
        os.path.join(os.path.dirname(__file__), "..", "..", "models", filename),
        os.path.join(os.path.dirname(__file__), "..", "..", "models", "checkpoints", filename),
    ]
    for p in candidates:
        abs_p = os.path.abspath(p)
        if os.path.exists(abs_p):
            return abs_p
    return filename


class ReIDExtractor:
    def __init__(self, backend="custom", model_path="custom_reid_net.pth", device=None):
        self.backend = backend
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if self.backend == "custom":
            print(f"[*] Khởi tạo Customized ReID Model (Metric Learning + BNNeck) trên {self.device}...")
            self.model = CustomReIDNet(embedding_dim=512, pretrained=True).to(self.device)
            
            resolved_ckpt = resolve_model_path(model_path)
            if os.path.exists(resolved_ckpt):
                try:
                    state_dict = torch.load(resolved_ckpt, map_location=self.device)
                    self.model.load_state_dict(state_dict, strict=False)
                    print(f"[*] Đã nạp thành công trọng số customized weights từ {resolved_ckpt}!")
                except Exception as e:
                    print(f"[!] Chú ý: Dùng weights mặc định do không nạp được {resolved_ckpt}: {e}")
            else:
                print(f"[!] Không tìm thấy file trọng số {model_path}, khởi tạo với ImageNet weights.")

            self.model.eval()

            # Preprocessing chuẩn Person ReID (256x128 chuẩn tỷ lệ dáng người)
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            resolved_onnx = resolve_model_path(model_path if model_path.endswith('.onnx') else "yolo26n-reid.onnx")
            if not os.path.exists(resolved_onnx):
                raise FileNotFoundError(f"Không tìm thấy model ReID ONNX: {resolved_onnx}")
            self.encoder = ReID(resolved_onnx)
            self.input_size = (128, 256)

    def extract_crop_feature(self, crop_bgr):
        """
        Trích xuất vector embedding L2-normalized từ ảnh crop BGR.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        if self.backend == "custom":
            try:
                rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                tensor = self.transform(rgb).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feat = self.model(tensor)
                    feat_np = feat.cpu().squeeze(0).numpy()
                    norm = np.linalg.norm(feat_np)
                    if norm > 1e-8:
                        feat_np = feat_np / norm
                    return feat_np
            except Exception as e:
                print(f"[!] Lỗi trích xuất feature (Custom): {e}")
                return None
        else:
            try:
                features = self.encoder._extract_features([crop_bgr])
                if features is not None and len(features) > 0:
                    feat = np.ascontiguousarray(features[0], dtype=np.float32)
                    norm = np.linalg.norm(feat)
                    if norm > 1e-8:
                        feat = feat / norm
                    return feat
            except Exception as e:
                print(f"[!] Lỗi trích xuất feature (ONNX): {e}")
                return None
        return None

    @staticmethod
    def cosine_similarity(f1, f2):
        if f1 is None or f2 is None:
            return 0.0
        return float(np.dot(f1, f2))

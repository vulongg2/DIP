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
import onnxruntime as ort

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
    def __init__(self, backend="onnx", model_path="yolo26n-reid.onnx", device=None):
        self.backend = backend
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if self.backend == "custom":
            print(f"[*] Khoi tao Customized ReID Model (Metric Learning + BNNeck) tren {self.device}...")
            self.model = CustomReIDNet(embedding_dim=512, pretrained=True).to(self.device)
            
            resolved_ckpt = resolve_model_path(model_path if model_path.endswith('.pth') else "custom_reid_net.pth")
            if os.path.exists(resolved_ckpt):
                try:
                    state_dict = torch.load(resolved_ckpt, map_location=self.device)
                    self.model.load_state_dict(state_dict, strict=False)
                    print(f"[*] Nap thanh cong trong so customized weights tu {resolved_ckpt}!")
                except Exception as e:
                    print(f"[!] Chu y: Dung weights mac dinh do khong nap duoc {resolved_ckpt}: {e}")
            else:
                print(f"[!] Khong tim thay file trong so {model_path}, khoi tao voi ImageNet weights.")

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
            
            # Khởi tạo ONNX Runtime Session an toàn
            available_providers = ort.get_available_providers()
            providers = [p for p in ['CUDAExecutionProvider', 'CPUExecutionProvider'] if p in available_providers]
            if not providers:
                providers = ['CPUExecutionProvider']
            self.ort_session = ort.InferenceSession(resolved_onnx, providers=providers)
            
            self.input_name = self.ort_session.get_inputs()[0].name
            self.output_name = self.ort_session.get_outputs()[0].name
            print(f"[*] Nap thanh cong ReID ONNX Model tu {resolved_onnx} (Provider: {self.ort_session.get_providers()[0]})")

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
                # Preprocessing cho ONNX ReID model (128x256, RGB, ImageNet mean/std)
                resized = cv2.resize(crop_bgr, (128, 256))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                norm = (rgb - mean) / std
                blob = np.transpose(norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
                
                out = self.ort_session.run([self.output_name], {self.input_name: blob})[0][0]
                norm_val = np.linalg.norm(out)
                if norm_val > 1e-8:
                    out = out / norm_val
                return out
            except Exception as e:
                print(f"[!] Lỗi trích xuất feature (ONNX): {e}")
                return None
        return None

    @staticmethod
    def cosine_similarity(f1, f2):
        if f1 is None or f2 is None:
            return 0.0
        return float(np.dot(f1, f2))

"""
ReID Feature Extractor & DIP Preprocessing Module (Bước 3 & Module 4).
Đảm nhiệm:
1. Tiền xử lý ảnh số (DIP): Đổi hệ màu BGR -> RGB, Nội suy Bilinear Resize (128x256), Chuẩn hóa Z-Score ImageNet.
2. Trích xuất vector đặc trưng 512 chiều chuẩn hóa L2 qua 2 Backend:
   - 'custom': Mạng PyTorch CustomReIDNet (MobileNetV3 + BNNeck + Triplet Loss).
   - 'onnx'  : Mô hình ONNX Runtime tốc độ cao (yolo26n-reid.onnx).
"""

import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import onnxruntime as ort

from .custom_reid_net import CustomReIDNet


def resolve_model_path(filename):
    """Tự động tìm đường dẫn file trọng số trong thư mục gốc, models/, hoặc models/checkpoints/."""
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
        """
        Khởi tạo bộ trích xuất đặc trưng ReID.
        - backend: 'custom' (dùng mạng tự huấn luyện custom_reid_net.pth) hoặc 'onnx' (dùng yolo26n-reid.onnx).
        """
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

            self.model.eval()  # Chuyển mạng sang chế độ suy luận (cố định BatchNorm & tắt Dropout)

            # Chuỗi tiền xử lý ảnh số chuẩn Person ReID (Chương 2 & Chương 3 giáo trình DIP):
            # 1. Resize (256, 128): Nội suy song tuyến tính đưa về tỷ lệ dáng người đứng 2:1.
            # 2. ToTensor(): Chuyển ma trận [0, 255] sang số thực [0.0, 1.0] và đảo trục thành [3, 256, 128].
            # 3. Normalize(mean, std): Chuẩn hóa Z-Score khử độ lệch sáng và cân bằng độ tương phản giữa các phòng.
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            resolved_onnx = resolve_model_path(model_path if model_path.endswith('.onnx') else "yolo26n-reid.onnx")
            if not os.path.exists(resolved_onnx):
                raise FileNotFoundError(f"Khong tim thay model ReID ONNX: {resolved_onnx}")
            
            # Khởi tạo phiên chạy ONNX Runtime
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
        Biến một bức ảnh cắt người (crop_bgr) thành Vector đặc trưng 512 chiều chuẩn hóa L2.
        - Input : crop_bgr (np.ndarray, hệ màu BGR, kích thước [h, w, 3] bất kỳ).
        - Output: feat_np  (np.ndarray, shape=(512,), dtype=float32, độ dài ||f||_2 = 1.0).
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        if self.backend == "custom":
            try:
                # 1. Chuyển hệ màu từ BGR (OpenCV) sang RGB (PyTorch)
                rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                
                # 2. Tiền xử lý Resize + Chuẩn hóa Z-Score và thêm chiều Batch ở đầu -> [1, 3, 256, 128]
                tensor = self.transform(rgb).unsqueeze(0).to(self.device)
                
                # 3. Đưa qua mạng CustomReIDNet (tắt tính đạo hàm torch.no_grad để chạy nhanh và tiết kiệm VRAM)
                with torch.no_grad():
                    feat = self.model(tensor)
                    feat_np = feat.cpu().squeeze(0).numpy()
                    
                    # 4. Đảm bảo chuẩn hóa L2 đưa vector về mặt cầu đơn vị (||f||_2 = 1.0)
                    norm = np.linalg.norm(feat_np)
                    if norm > 1e-8:
                        feat_np = feat_np / norm
                    return feat_np
            except Exception as e:
                print(f"[!] Loi trich xuat feature (Custom): {e}")
                return None
        else:
            try:
                # 1. Nội suy song tuyến tính (Bilinear Interpolation) về chuẩn (Rộng 128, Cao 256)
                resized = cv2.resize(crop_bgr, (128, 256))
                
                # 2. Chuyển BGR sang RGB và chia 255 đưa về miền [0.0, 1.0]
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                
                # 3. Chuẩn hóa Z-Score: (I - mean) / std
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                norm = (rgb - mean) / std
                
                # 4. Đảo trục từ [H, W, C] sang [1, C, H, W] tức [1, 3, 256, 128]
                blob = np.transpose(norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
                
                # 5. Chạy suy luận đồ thị ONNX và chuẩn hóa L2
                out = self.ort_session.run([self.output_name], {self.input_name: blob})[0][0]
                norm_val = np.linalg.norm(out)
                if norm_val > 1e-8:
                    out = out / norm_val
                return out
            except Exception as e:
                print(f"[!] Loi trich xuat feature (ONNX): {e}")
                return None

    @staticmethod
    def cosine_similarity(f1, f2):
        """
        Tính độ tương đồng Cosine giữa 2 vector đặc trưng 512 chiều:
            CosineSim(f1, f2) = (f1 · f2) / (||f1|| * ||f2||)
        Vì cả f1 và f2 đều đã được chuẩn hóa L2 (||f1|| = ||f2|| = 1.0),
        công thức rút gọn chỉ còn phép nhân tích vô hướng np.dot(f1, f2)!
        """
        if f1 is None or f2 is None:
            return 0.0
        return float(np.dot(f1, f2))

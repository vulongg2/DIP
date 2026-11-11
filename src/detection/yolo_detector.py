"""
YOLO Person Detector & Local Tracker Wrapper (Module 1 & 2).
Đóng gói mô hình YOLOv8 kết hợp cùng bộ bám vết cục bộ BoT-SORT / ByteTrack.
"""

import os
import torch
from ultralytics import YOLO


def resolve_yolo_weights(weights_name="yolov8s.pt"):
    """
    Tự động dò tìm đường dẫn file trọng số YOLO (.pt) trong thư mục gốc hoặc thư mục models/.
    Đặc biệt: Tự động phát hiện và xử lý trường hợp người dùng tải repo bằng nút 'Download ZIP'
    trên GitHub khiến file .pt bị biến thành file con trỏ Git LFS (chỉ nặng ~130 bytes, bắt đầu bằng chữ 'version').
    Khi phát hiện file < 10KB, hàm sẽ tự xóa file con trỏ lỗi đó và trả về tên chuẩn (VD: 'yolov8s.pt')
    để thư viện Ultralytics tự động tải file trọng số thật về máy.
    """
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
            try:
                # Kiểm tra nếu file nhỏ hơn 10KB -> Chắc chắn là file text con trỏ Git LFS (130 bytes)
                if os.path.getsize(abs_p) < 10240:
                    print(f"[!] Phat hien file Git LFS pointer (<10KB) tai: {abs_p}")
                    print(f"[*] Dang xoa file pointer loi va tu dong tai {os.path.basename(weights_name)} tu Ultralytics...")
                    os.remove(abs_p)
                    continue
            except OSError:
                pass
            return abs_p
    # Trả về tên file gốc (vd: 'yolov8s.pt') để Ultralytics tự tải từ internet nếu chưa có trên máy
    return os.path.basename(weights_name)


class YOLODetector:
    def __init__(self, model_name="yolov8s.pt", conf=0.45, device=None):
        """
        Khởi tạo mô hình phát hiện người YOLOv8.
        - model_name: Tên file trọng số (yolov8s.pt hoặc yolov8n.pt).
        - conf: Ngưỡng độ tự tin tối thiểu (mặc định 0.45 = 45%).
        - device: 'cuda' (chạy trên GPU NVIDIA) hoặc 'cpu'.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights_path = resolve_yolo_weights(model_name)
        print(f"[*] Nap mo hinh YOLO Detection tu: {weights_path} (Thiet bi: {self.device})")
        self.model = YOLO(weights_path)
        self.conf = conf

    def detect_and_track(self, frame, tracker_cfg=None):
        """
        Hàm 2-trong-1: Thực hiện cả Phát hiện đối tượng (Detection) và Bám vết cục bộ (Local Tracking).
        Bên trong hàm self.model.track() của Ultralytics sẽ tự động chạy 2 bước nối tiếp:
          + Bước 1: Đưa ảnh qua mạng CNN YOLOv8 để tìm các hộp bao [x1, y1, x2, y2].
          + Bước 2: Đưa các hộp bao vào thuật toán BoT-SORT (configs/tracker.yaml) để chạy
            bộ lọc Kalman dự đoán vị trí và ghép cặp Hungarian IoU, từ đó gán số local_track_id.
        - Input:
            + frame: Khung hình BGR uint8 [H, W, 3].
            + tracker_cfg: Đường dẫn tới file cấu hình configs/tracker.yaml.
        - Output:
            + Đối tượng Results chứa:
                * results[0].boxes.xyxy : Tọa độ hộp bao [x1, y1, x2, y2] (Output của YOLO)
                * results[0].boxes.id   : Số thứ tự tạm thời local_track_id (Output của Tracker)
                * results[0].boxes.conf : Độ tự tin của từng hộp bao
        """
        results = self.model.track(
            source=frame,
            persist=True,       # BẮT BUỘC = True để giữ bộ nhớ Kalman Filter từ frame trước sang frame sau
            classes=[0],        # Chỉ lọc class 0 ('person' trong bộ dữ liệu COCO 80 lớp)
            conf=self.conf,     # Lọc bỏ các phát hiện có độ tự tin thấp hơn ngưỡng
            tracker=tracker_cfg,# File cấu hình siêu tham số của BoT-SORT / ByteTrack
            verbose=False,      # Tắt in log thừa ra màn hình console để tăng tốc
            device=self.device
        )
        return results[0] if results else None

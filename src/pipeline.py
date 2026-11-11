"""
Unified MTMCT Pipeline Module (Bộ điều phối trung tâm của hệ thống).
Điều phối toàn diện quy trình 5 bước của Xử lý ảnh số (DIP):
1. Thu nhận khung hình (Frame Ingestion)
2. Phát hiện người bằng YOLOv8 (Detection) + Bám vết cục bộ Kalman Filter (BoT-SORT / ByteTrack)
3. Cắt vùng quan tâm ROI (Cropping) & Trích xuất vector đặc trưng 512 chiều (ReID Feature Extraction)
4. Kiểm tra ràng buộc đồ thị không - thời gian (NetworkX Spatial-Temporal Topology)
5. Quản lý danh tính toàn cục (Global ID Management) & Quy đổi tọa độ Bản đồ 2D Floor Plan.
"""

import time
import cv2
import numpy as np

from src.detection.yolo_detector import YOLODetector              # Module 1: Phát hiện người (YOLOv8)
from src.tracking import resolve_tracker_config                   # Module 2: Tìm cấu hình thuật toán bám vết
from src.association.global_manager import GlobalIdentityManager  # Module 5: Ghép nối Global ID, EMA & Topology
from src.visualization.floor_plan_renderer import FloorPlanRenderer  # Module 7: Dựng bản đồ mặt bằng 2D


class MTMCTPipeline:
    def __init__(self,
                 detector_weights="yolov8s.pt",
                 reid_backend="custom",
                 reid_weights="custom_reid_net.pth",
                 tracker_cfg=None,
                 similarity_thresh=0.65,
                 use_topology=True,
                 conf_thresh=0.45):
        """
        Khởi tạo toàn bộ các mô hình AI và cấu trúc dữ liệu (chỉ chạy 1 lần khi bật chương trình).
        - detector_weights: Trọng số YOLOv8 phát hiện người.
        - reid_backend: 'custom' (mạng PyTorch BNNeck tự thiết kế) hoặc 'onnx' (chạy qua ONNX Runtime).
        - similarity_thresh: Ngưỡng tương đồng Cosine (>= 0.65 thì gán cùng 1 Global ID).
        - use_topology: Bật/tắt kiểm tra luật di chuyển giữa các phòng bằng đồ thị NetworkX.
        """
        # 1. Nạp mô hình YOLOv8 lên GPU (CUDA) hoặc CPU
        self.detector = YOLODetector(model_name=detector_weights, conf=conf_thresh)
        
        # 2. Xác định đường dẫn file cấu hình BoT-SORT / ByteTrack (configs/tracker.yaml)
        self.tracker_cfg = tracker_cfg or resolve_tracker_config()
        
        # 3. Khởi tạo bộ quản lý danh tính toàn cục (chứa Gallery, mạng ReID và đồ thị Topology)
        self.global_manager = GlobalIdentityManager(
            reid_backend=reid_backend,
            reid_model_path=reid_weights,
            similarity_thresh=similarity_thresh,
            use_topology=use_topology
        )
        
        # 4. Khởi tạo bộ vẽ sơ đồ mặt bằng 2D
        self.floor_plan = FloorPlanRenderer()
        print(f"[*] MTMCT Pipeline da san sang (Tracker Config: {self.tracker_cfg})")

    def process_frame(self, frame, cam_id="Cam_1", current_time=None):
        """
        Xử lý trọn vẹn 1 khung hình (frame) từ 1 camera cụ thể.
        - Input:
            + frame: Ma trận ảnh BGR uint8 kích thước [H, W, 3] từ OpenCV.
            + cam_id: Tên định danh camera hiện tại (VD: 'Cam_1', 'Cam_2', 'Cam_3').
            + current_time: Thời gian thực (giây) để tính khoảng thời gian di chuyển Δt.
        - Output:
            + annotated_frame: Ảnh đã vẽ khung Bounding Box, Global ID và điểm tương đồng.
            + detections_info: Danh sách thông tin tọa độ của từng người để vẽ lên bản đồ 2D.
        """
        if current_time is None:
            current_time = time.time()

        # Lấy chiều cao (h) và chiều rộng (w) của khung hình
        h, w = frame.shape[:2]

        # BƯỚC 1 & 2: Chạy hàm 2-trong-1 (YOLOv8 phát hiện tọa độ + Tracker Kalman gán Local ID)
        res = self.detector.detect_and_track(frame, tracker_cfg=self.tracker_cfg)

        # Tạo bản sao của ảnh gốc để vẽ vời (giữ nguyên frame gốc sạch để cắt ảnh crop không bị dính nét vẽ)
        annotated_frame = frame.copy()
        detections_info = []

        # Kiểm tra xem trong khung hình có phát hiện được đối tượng nào không
        if res is not None and res.boxes is not None and len(res.boxes) > 0:
            # Chuyển kết quả từ GPU Tensor về mảng NumPy trên RAM để xử lý
            boxes = res.boxes.xyxy.cpu().numpy()  # Tọa độ hộp bao [N, 4]: (x1, y1, x2, y2)
            track_ids = res.boxes.id.int().cpu().numpy() if res.boxes.id is not None else None  # Local Track ID
            clses = res.boxes.cls.int().cpu().numpy()  # Nhãn lớp đối tượng (0 = Person)

            # Duyệt qua từng đối tượng phát hiện được trong khung hình
            for i, box in enumerate(boxes):
                if clses[i] != 0:
                    continue  # Bỏ qua nếu không phải là người (class != 0)

                # Ép kiểu tọa độ sang số nguyên (vì chỉ số dòng/cột của pixel phải là số nguyên)
                x1, y1, x2, y2 = map(int, box)

                # Kẹp biên an toàn (Boundary Clamping): Đảm bảo tọa độ không vượt ngoài phạm vi bức ảnh
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                # Lọc nhiễu hình thái: Bỏ qua các hộp bao quá nhỏ (rộng < 15px hoặc cao < 30px)
                if (x2 - x1) < 15 or (y2 - y1) < 30:
                    continue

                # Lấy số thứ tự tạm thời trong phòng (Local ID do BoT-SORT cấp)
                local_id = int(track_ids[i]) if track_ids is not None else (i + 1)

                # BƯỚC 3: Cắt vùng quan tâm ROI (Cropping) từ ảnh gốc sạch
                # Cắt từ hàng y1 đến y2, từ cột x1 đến x2 -> Loại bỏ 95% phông nền thừa
                crop = frame[y1:y2, x1:x2]

                # BƯỚC 4 & 5: Đưa ảnh crop vào Global Manager để trích xuất ReID 512D,
                # kiểm tra 3 chốt chặn (Mutual Exclusion + Topology + Cosine Sim) và lấy Global ID
                gid, color, is_new, sim = self.global_manager.match_or_create(
                    cam_id=cam_id,
                    local_track_id=local_id,
                    crop_bgr=crop,
                    box=(x1, y1, x2, y2),
                    current_time=current_time
                )

                # BƯỚC 6: Vẽ khung Bounding Box và nhãn Global ID lên bản sao khung hình
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{gid} (L:{local_id}) {sim:.2f}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - lh - 6), (x1 + lw + 6, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # BƯỚC 7: Đóng gói dữ liệu vị trí chuẩn hóa [0.0 -> 1.0] làm Input cho bộ vẽ Bản đồ 2D
                # - norm_x: Hoành độ tâm cơ thể chia cho chiều rộng ảnh w
                # - norm_y: Tung độ đáy hộp bao (vị trí bàn chân chạm sàn y2) chia cho chiều cao ảnh h
                detections_info.append({
                    "global_id": gid,
                    "local_id": local_id,
                    "box": (x1, y1, x2, y2),
                    "norm_x": float(((x1 + x2) / 2) / w),
                    "norm_y": float(y2 / h),
                    "color": color,
                    "cam_id": cam_id,
                    "sim": sim
                })

        return annotated_frame, detections_info

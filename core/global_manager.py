"""
Global Identity Manager (Online Cross-Camera & Single-Camera ReID Association).
Quản lý tập hợp danh tính toàn cục (Global ID Gallery), cập nhật đặc trưng động (EMA),
và lưu trữ hành trình di chuyển (Trajectory) liên camera.
"""

import time
import numpy as np
from core.topology import CameraTopology
from core.reid_extractor import ReIDExtractor


class GlobalIdentity:
    def __init__(self, global_id, feature, initial_cam, timestamp, color=None, initial_local_id=None):
        self.global_id = global_id
        self.feature = feature  # L2-normalized feature vector
        self.last_cam = initial_cam
        self.last_local_id = initial_local_id
        self.last_seen_time = timestamp
        self.first_seen_time = timestamp
        self.total_detections = 1
        self.color = color or self._generate_color(global_id)
        
        # Lịch sử hành trình: list các dict {timestamp, cam_id, box, center}
        self.trajectory = []

    def update_feature(self, new_feature, alpha=0.85):
        """Cập nhật feature vector theo Exponential Moving Average (EMA)."""
        if new_feature is None:
            return
        updated = alpha * self.feature + (1.0 - alpha) * new_feature
        norm = np.linalg.norm(updated)
        if norm > 1e-8:
            self.feature = updated / norm

    def add_trajectory_point(self, cam_id, box, timestamp, local_track_id=None):
        x1, y1, x2, y2 = box
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        self.trajectory.append({
            "timestamp": timestamp,
            "cam_id": cam_id,
            "box": (x1, y1, x2, y2),
            "center": center,
        })
        self.last_cam = cam_id
        if local_track_id is not None:
            self.last_local_id = local_track_id
        self.last_seen_time = timestamp
        self.total_detections += 1

    @staticmethod
    def _generate_color(idx):
        colors = [
            (0, 255, 0),    # Lime
            (255, 128, 0),  # Orange
            (0, 200, 255),  # Cyan-Yellow
            (255, 0, 128),  # Magenta
            (128, 0, 255),  # Purple
            (0, 255, 255),  # Yellow
            (255, 255, 0),  # Cyan
            (0, 128, 255),  # Orange-Red
        ]
        return colors[(idx - 1) % len(colors)]


class GlobalIdentityManager:
    def __init__(self, reid_backend="custom", reid_model_path="yolo26n-reid.onnx", similarity_thresh=0.65, use_topology=True):
        self.extractor = ReIDExtractor(backend=reid_backend, model_path=reid_model_path)
        self.topology = CameraTopology() if use_topology else None
        self.similarity_thresh = similarity_thresh
        
        # Gallery lưu trữ các đối tượng: dict {global_id: GlobalIdentity}
        self.gallery = {}
        self.next_global_id = 1
        
        # Mapping cục bộ: dict {(cam_id, local_track_id): global_id}
        self.local_to_global = {}

        # Log hành trình tổng hợp để xuất báo cáo
        self.event_log = []

    def reset(self):
        """Xóa trắng bộ nhớ gallery (dùng khi muốn test lại từ đầu)."""
        self.gallery.clear()
        self.local_to_global.clear()
        self.event_log.clear()
        self.next_global_id = 1

    def match_or_create(self, cam_id, local_track_id, crop_bgr, box, current_time=None):
        """
        Nhận vào 1 detection đã có local_track_id từ tracker ở 1 camera cụ thể.
        Trả về: (global_id, color, is_new, similarity)
        """
        if current_time is None:
            current_time = time.time()

        key = (cam_id, local_track_id)
        
        # 1. Nếu local_track này đã được gán Global ID trước đó trong cùng luồng camera
        if key in self.local_to_global:
            gid = self.local_to_global[key]
            identity = self.gallery[gid]
            identity.add_trajectory_point(cam_id, box, current_time, local_track_id=local_track_id)
            
            # Cập nhật nhẹ nhàng feature định kỳ
            if identity.total_detections % 15 == 0:
                feat = self.extractor.extract_crop_feature(crop_bgr)
                if feat is not None:
                    identity.update_feature(feat, alpha=0.90)

            return gid, identity.color, False, 1.0

        # 2. Local track mới xuất hiện -> cần trích xuất ReID feature để so khớp với Gallery
        feat = self.extractor.extract_crop_feature(crop_bgr)
        best_gid = None
        best_sim = -1.0

        if feat is not None and len(self.gallery) > 0:
            for gid, identity in self.gallery.items():
                time_delta = current_time - identity.last_seen_time
                
                # RÀNG BUỘC LOẠI TRỪ LẪN NHAU (MUTUAL EXCLUSION):
                # Nếu Global ID này đang hiện diện trong cùng camera (vừa nhìn thấy trong vòng 1.5s)
                # và thuộc về một người khác (local_track khác) -> KHÔNG THỂ LÀ CÙNG 1 NGƯỜI ĐƯỢC!
                # (Một người không thể phân thân làm 2 người cùng đứng trong khung hình)
                if identity.last_cam == cam_id and time_delta < 1.5:
                    if identity.last_local_id != local_track_id:
                        continue

                # Kiểm tra ràng buộc Topology (nếu có bật)
                if self.topology is not None:
                    if not self.topology.is_transition_valid(identity.last_cam, cam_id, time_delta):
                        continue  # Bỏ qua ứng viên vi phạm luật di chuyển giữa các camera

                sim = self.extractor.cosine_similarity(feat, identity.feature)
                if sim > best_sim:
                    best_sim = sim
                    best_gid = gid

        # 3. Quyết định: Gán vào Global ID cũ hay tạo mới
        if best_gid is not None and best_sim >= self.similarity_thresh:
            # Khớp với đối tượng đã có
            gid = best_gid
            identity = self.gallery[gid]
            identity.update_feature(feat, alpha=0.80)
            identity.add_trajectory_point(cam_id, box, current_time, local_track_id=local_track_id)
            self.local_to_global[key] = gid
            
            self.event_log.append({
                "time": current_time,
                "global_id": gid,
                "event": f"Re-identified at {cam_id} (sim={best_sim:.2f})",
                "cam": cam_id
            })
            return gid, identity.color, False, best_sim
        else:
            # Tạo Global ID mới
            gid = self.next_global_id
            self.next_global_id += 1
            identity = GlobalIdentity(gid, feat, cam_id, current_time, initial_local_id=local_track_id)
            identity.add_trajectory_point(cam_id, box, current_time, local_track_id=local_track_id)
            self.gallery[gid] = identity
            self.local_to_global[key] = gid

            self.event_log.append({
                "time": current_time,
                "global_id": gid,
                "event": f"New identity initialized at {cam_id}",
                "cam": cam_id
            })
            return gid, identity.color, True, (best_sim if best_sim > 0 else 0.0)

    def get_summary(self):
        """Tóm tắt thông tin các đối tượng đã nhận diện."""
        summary = []
        for gid, idt in self.gallery.items():
            cams_visited = list(dict.fromkeys([p["cam_id"] for p in idt.trajectory]))
            duration_sec = idt.last_seen_time - idt.first_seen_time
            summary.append({
                "global_id": gid,
                "first_seen": idt.first_seen_time,
                "last_seen": idt.last_seen_time,
                "duration_sec": duration_sec,
                "total_points": len(idt.trajectory),
                "cameras_visited": cams_visited,
                "last_camera": idt.last_cam
            })
        return summary

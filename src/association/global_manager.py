"""
Global Identity Manager (Online Cross-Camera & Single-Camera ReID Association).
Quản lý tập hợp danh tính toàn cục (Global ID Gallery), cập nhật đặc trưng động (EMA),
ràng buộc loại trừ không gian (Spatial Mutual Exclusion) và lưu trữ hành trình (Trajectory).

========================================================================================
LUỒNG LOGIC CỐT LÕI CỦA BỘ QUẢN LÝ DANH TÍNH TOÀN CỤC (`GlobalIdentityManager`):
Khi nhận vào 1 đối tượng `(cam_id, local_track_id, crop_bgr, box, current_time)` từ `pipeline.py`:
1. Tra cứu bộ nhớ đệm `(cam_id, local_track_id)` trong `local_to_global`:
   - Nếu `local_track_id` này đã được cấp `Global ID` từ các frame trước -> Giữ nguyên `Global ID` đó!
   - Cứ mỗi 15 frame, trích xuất lại vector đặc trưng của ảnh `crop_bgr` và cập nhật làm mượt bằng
     bộ lọc thông thấp EMA (Exponential Moving Average) với hệ số alpha = 0.90.

2. Nếu là `local_track_id` MỚI (vừa bước vào camera hoặc vừa bị mất dấu rồi xuất hiện lại):
   - Gọi `ReIDExtractor.extract_crop_feature(crop_bgr)` để lấy vector đặc trưng 512 chiều `feat`.
   - Duyệt qua toàn bộ các đối tượng đang lưu trong `Gallery` (`self.gallery`) và lọc qua 3 tầng:
     + Tầng 1 - Ràng buộc Loại trừ Không gian (Spatial Mutual Exclusion):
       Nếu ứng viên trong Gallery đang xuất hiện CÙNG LÚC trên cùng camera (trong vòng 1.5 giây)
       với một `local_track_id` khác -> Chắc chắn là 2 người khác nhau đứng cạnh nhau -> Bỏ qua!
     + Tầng 2 - Ràng buộc Đồ thị Không gian - Thời gian (`CameraTopology.is_transition_valid`):
       Kiểm tra thời gian di chuyển giữa `last_cam` và `cam_id` hiện tại có nằm trong `[min_sec, max_sec]` không.
     + Tầng 3 - So khớp Độ tương đồng Cosine (`cosine_similarity`):
       Tìm ứng viên có điểm tương đồng Cosine cao nhất (`best_sim`).

3. Ra quyết định:
   - Nếu `best_sim >= similarity_thresh` (mặc định 0.65): Khớp thành công (`REID_MATCH`) -> Gán lại `Global ID` cũ,
     cập nhật EMA vector đặc trưng với alpha = 0.80.
   - Ngược lại (`best_sim < similarity_thresh`): Cấp mới hoàn toàn một `Global ID` (`NEW_GLOBAL_ID`).
========================================================================================
"""

import time
import numpy as np

try:
    from src.topology.camera_graph import CameraTopology
    from src.reid.reid_extractor import ReIDExtractor
except ImportError:
    from ..topology.camera_graph import CameraTopology
    from ..reid.reid_extractor import ReIDExtractor


class GlobalIdentity:
    """
    Lớp lưu trữ hồ sơ của 1 người (1 Global ID) xuyên suốt toàn hệ thống đa camera.
    """
    def __init__(self, global_id, feature, initial_cam, timestamp, color=None, initial_local_id=None):
        self.global_id = global_id                  # Mã định danh toàn cục duy nhất (1, 2, 3, ...)
        self.feature = feature                      # Vector đặc trưng ngoại hình 512 chiều (đã chuẩn hóa L2)
        self.last_cam = initial_cam                 # Camera gần nhất nhìn thấy người này
        self.last_local_id = initial_local_id       # Local Track ID gần nhất tại camera đó
        self.last_seen_time = timestamp             # Thời điểm gần nhất xuất hiện (giây)
        self.first_seen_time = timestamp            # Thời điểm đầu tiên phát hiện trong hệ thống
        self.total_detections = 1                   # Tổng số frame đã theo dõi người này
        self.color = color or self._generate_color(global_id)  # Màu sắc cố định (BGR) để vẽ BBox
        
        # Lịch sử hành trình: danh sách các điểm {timestamp, cam_id, box, center}
        self.trajectory = []

    def update_feature(self, new_feature, alpha=0.85):
        """
        Cập nhật vector đặc trưng ngoại hình theo Trung bình trượt hàm mũ (Exponential Moving Average - EMA).
        Đóng vai trò như một BỘ LỌC THÔNG THẤP (Low-pass Filter) trong xử lý tín hiệu số:
            f_t = normalize( alpha * f_{t-1} + (1 - alpha) * f_new )
        Giúp giữ lại 85% đặc trưng gốc ổn định của người đó và chỉ hấp thụ 15% sự thay đổi góc nhìn mới,
        chống lại hiện tượng trôi đặc trưng (Feature Drift) khi người đó bị che khuất tạm thời.
        """
        if new_feature is None:
            return
        updated = alpha * self.feature + (1.0 - alpha) * new_feature
        # Chuẩn hóa L2 lại để vector luôn nằm trên mặt cầu đơn vị (phục vụ tính tích vô hướng Cosine)
        norm = np.linalg.norm(updated)
        if norm > 1e-8:
            self.feature = updated / norm

    def add_trajectory_point(self, cam_id, box, timestamp, local_track_id=None):
        """Ghi nhận thêm 1 tọa độ vị trí mới vào lịch sử di chuyển của đối tượng."""
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
        """Bảng màu BGR tươi sáng cố định cho từng Global ID để dễ phân biệt bằng mắt thường."""
        colors = [
            (0, 255, 0),    # Xanh lá (Lime)
            (255, 128, 0),  # Xanh dương sáng
            (0, 200, 255),  # Vàng cam
            (255, 0, 128),  # Hồng tím (Magenta)
            (128, 0, 255),  # Tím (Purple)
            (0, 255, 255),  # Vàng (Yellow)
            (255, 255, 0),  # Xanh lơ (Cyan)
            (0, 128, 255),  # Cam đỏ (Orange-Red)
        ]
        return colors[(idx - 1) % len(colors)]


class GlobalIdentityManager:
    """
    Bộ điều phối trung tâm quản lý Gallery danh tính toàn cục và thực hiện Re-Identification liên camera.
    """
    def __init__(self, reid_backend="custom", reid_model_path="custom_reid_net.pth", similarity_thresh=0.65, use_topology=True):
        # Khởi tạo bộ trích xuất đặc trưng ReID (hỗ trợ ONNX hoặc Custom PyTorch Model)
        self.extractor = ReIDExtractor(backend=reid_backend, model_path=reid_model_path)
        # Khởi tạo đồ thị không gian - thời gian NetworkX
        self.topology = CameraTopology() if use_topology else None
        # Ngưỡng tương đồng Cosine tối thiểu để công nhận là cùng 1 người
        self.similarity_thresh = similarity_thresh
        
        # Bộ nhớ Gallery lưu trữ toàn bộ các đối tượng đã xuất hiện: dict {global_id: GlobalIdentity}
        self.gallery = {}
        self.next_global_id = 1
        
        # Bảng ánh xạ nhanh từ Track cục bộ của từng camera sang Global ID:
        # dict {(cam_id, local_track_id): global_id}
        self.local_to_global = {}

        # Nhật ký sự kiện (Event Log) lưu lịch sử cấp mới / tái nhận diện để xuất báo cáo
        self.event_log = []

    def reset(self):
        """Xóa trắng bộ nhớ Gallery (dùng khi muốn chạy lại từ đầu)."""
        self.gallery.clear()
        self.local_to_global.clear()
        self.event_log.clear()
        self.next_global_id = 1

    def _find_best_candidate(self, feat, cam_id, local_track_id, current_time, exclude_gid=None):
        """Tìm Global ID phù hợp nhất trong Gallery thỏa mãn cả 3 tầng lọc."""
        best_gid = None
        best_sim = -1.0
        if feat is None or len(self.gallery) == 0:
            return best_gid, best_sim

        for gid, identity in self.gallery.items():
            if exclude_gid is not None and gid == exclude_gid:
                continue
            time_delta = current_time - identity.last_seen_time

            # Bỏ qua các danh tính "bóng ma" thoáng qua (phản chiếu kính / false positive < 4 frame đã tắt > 3s)
            if identity.total_detections < 4 and time_delta > 3.0:
                continue

            # [TẦNG LỌC 1] Ràng buộc loại trừ không gian (Spatial Mutual Exclusion)
            if identity.last_cam == cam_id and time_delta < 1.5:
                if identity.last_local_id != local_track_id:
                    continue

            # [TẦNG LỌC 2] Ràng buộc đồ thị Topology không gian - thời gian
            if self.topology is not None:
                if not self.topology.is_transition_valid(identity.last_cam, cam_id, time_delta):
                    continue

            # [TẦNG LỌC 3] Đo độ tương đồng Cosine Similarity
            sim = self.extractor.cosine_similarity(feat, identity.feature)
            if sim > best_sim:
                best_sim = sim
                best_gid = gid

        return best_gid, best_sim

    def match_or_create(self, cam_id, local_track_id, crop_bgr, box, current_time=None):
        """
        Hàm cốt lõi nhất của MTMCT:
        Nhận vào 1 đối tượng đã có `local_track_id` từ BoT-SORT/ByteTrack ở 1 camera cụ thể,
        quyết định xem người này thuộc `Global ID` nào trong Gallery hoặc tạo mới.

        Đầu vào:
            cam_id: Tên camera (vd: "Cam_1", "Cam_2")
            local_track_id: ID cục bộ do Tracker gán trong luồng camera đó
            crop_bgr: Ảnh cắt (crop) vùng chứa người (NumPy BGR)
            box: Tọa độ (x1, y1, x2, y2)
            current_time: Thời gian thực (giây)
        Đầu ra:
            Tuple (global_id, color, is_new, similarity)
        """
        if current_time is None:
            current_time = time.time()

        key = (cam_id, local_track_id)
        
        # =====================================================================
        # TRƯỜNG HỢP 1: Local Track này ĐÃ ĐƯỢC GÁN Global ID ở các frame trước
        # =====================================================================
        if key in self.local_to_global:
            gid = self.local_to_global[key]
            identity = self.gallery[gid]
            identity.add_trajectory_point(cam_id, box, current_time, local_track_id=local_track_id)

            # Tự hiệu chỉnh sớm (Grace-period Self-Correction):
            # Nếu track mới tạo ở các frame đầu bị dính mép/khuất một phần (Sim thấp tạm thời),
            # khi đối tượng bước trọn vào khung hình (frame thứ 5 và 10), tự động kiểm tra lại xem
            # có khớp mạnh với một Global ID cũ hợp lệ từ camera trước đó không để gộp lại ID.
            if identity.total_detections in (5, 10) and len(self.gallery) > 1:
                feat_check = self.extractor.extract_crop_feature(crop_bgr)
                if feat_check is not None:
                    cand_gid, cand_sim = self._find_best_candidate(
                        feat_check, cam_id, local_track_id, current_time, exclude_gid=gid
                    )
                    merge_thresh = max(self.similarity_thresh + 0.12, 0.40)
                    if cand_gid is not None and cand_gid < gid and cand_sim >= merge_thresh:
                        target_ident = self.gallery[cand_gid]
                        target_ident.update_feature(feat_check, alpha=0.80)
                        target_ident.add_trajectory_point(cam_id, box, current_time, local_track_id=local_track_id)
                        self.local_to_global[key] = cand_gid
                        if gid not in self.local_to_global.values():
                            self.gallery.pop(gid, None)
                        self.event_log.append({
                            "time": current_time,
                            "event": "REID_MATCH",
                            "global_id": cand_gid,
                            "cam_id": cam_id,
                            "local_track_id": local_track_id,
                            "similarity": cand_sim
                        })
                        print(f"[*] {cam_id} Track_{local_track_id}: Tu hieu chinh ID {gid} -> Global ID {cand_gid} (Sim={cand_sim:.2f})")
                        return cand_gid, target_ident.color, False, cand_sim
                    identity.update_feature(feat_check, alpha=0.80)

            # Định kỳ mỗi 15 frame mới chạy mạng CNN ReID 1 lần để cập nhật EMA đặc trưng (tiết kiệm GPU/CPU)
            elif identity.total_detections % 15 == 0:
                feat = self.extractor.extract_crop_feature(crop_bgr)
                if feat is not None:
                    identity.update_feature(feat, alpha=0.90)

            return gid, identity.color, False, 1.0

        # =====================================================================
        # TRƯỜNG HỢP 2: Local Track MỚI XUẤT HIỆN -> Trích xuất ReID & Tìm kiếm trong Gallery
        # =====================================================================
        feat = self.extractor.extract_crop_feature(crop_bgr)
        best_gid, best_sim = self._find_best_candidate(feat, cam_id, local_track_id, current_time)

        # =====================================================================
        # TRƯỜNG HỢP 3: RA QUYẾT ĐỊNH (GÁN LẠI ID CŨ HOẶC CẤP ID MỚI)
        # =====================================================================
        if best_gid is not None:
            print(f"[*] {cam_id} Track_{local_track_id}: So khop voi Global ID {best_gid} dat Sim={best_sim:.2f} (Nguong: {self.similarity_thresh:.2f})")
        else:
            print(f"[*] {cam_id} Track_{local_track_id}: Khong co ung vien hop le (Gallery co {len(self.gallery)} IDs)")

        if best_gid is not None and best_sim >= self.similarity_thresh:
            # ---> KHỚP THÀNH CÔNG VỚI NGƯỜI ĐÃ CÓ TRONG GALLERY (REID_MATCH)
            gid = best_gid
            identity = self.gallery[gid]
            identity.update_feature(feat, alpha=0.80)
            identity.add_trajectory_point(cam_id, box, current_time, local_track_id=local_track_id)
            self.local_to_global[key] = gid
            
            self.event_log.append({
                "time": current_time,
                "event": "REID_MATCH",
                "global_id": gid,
                "cam_id": cam_id,
                "local_track_id": local_track_id,
                "similarity": best_sim
            })
            return gid, identity.color, False, best_sim
        else:
            # ---> KHÔNG KHỚP AI TRONG GALLERY -> CẤP GLOBAL ID MỚI (NEW_GLOBAL_ID)
            gid = self.next_global_id
            self.next_global_id += 1
            
            # Dự phòng nếu ảnh crop quá mờ không trích được đặc trưng thì khởi tạo vector đơn vị ngẫu nhiên
            if feat is None:
                feat = np.random.randn(512).astype(np.float32)
                feat /= np.linalg.norm(feat)

            new_identity = GlobalIdentity(
                global_id=gid,
                feature=feat,
                initial_cam=cam_id,
                timestamp=current_time,
                initial_local_id=local_track_id
            )
            new_identity.add_trajectory_point(cam_id, box, current_time, local_track_id=local_track_id)
            
            self.gallery[gid] = new_identity
            self.local_to_global[key] = gid

            self.event_log.append({
                "time": current_time,
                "event": "NEW_GLOBAL_ID",
                "global_id": gid,
                "cam_id": cam_id,
                "local_track_id": local_track_id,
                "similarity": best_sim
            })
            return gid, new_identity.color, True, best_sim

    def get_active_identities(self, timeout_sec=30.0):
        """Lấy danh sách các Global ID đang hoạt động (xuất hiện trong vòng `timeout_sec` giây gần nhất)."""
        current_time = time.time()
        active = {}
        for gid, identity in self.gallery.items():
            if (current_time - identity.last_seen_time) <= timeout_sec:
                active[gid] = identity
        return active

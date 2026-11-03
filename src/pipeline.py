"""
Unified MTMCT Pipeline Module.
Điều phối toàn diện quá trình:
Frame Ingestion -> YOLOv8 Detection -> Local Tracker (ByteTrack/BoT-SORT) -> ReID Feature Extraction -> NetworkX Spatial-Temporal Topology -> Global ID Management.
"""

import time
import cv2
import numpy as np

from src.detection.yolo_detector import YOLODetector
from src.tracking import resolve_tracker_config
from src.association.global_manager import GlobalIdentityManager
from src.visualization.floor_plan_renderer import FloorPlanRenderer


class MTMCTPipeline:
    def __init__(self,
                 detector_weights="yolov8s.pt",
                 reid_backend="custom",
                 reid_weights="custom_reid_net.pth",
                 tracker_cfg=None,
                 similarity_thresh=0.65,
                 use_topology=True,
                 conf_thresh=0.45):
        
        self.detector = YOLODetector(model_name=detector_weights, conf=conf_thresh)
        self.tracker_cfg = tracker_cfg or resolve_tracker_config()
        self.global_manager = GlobalIdentityManager(
            reid_backend=reid_backend,
            reid_model_path=reid_weights,
            similarity_thresh=similarity_thresh,
            use_topology=use_topology
        )
        self.floor_plan = FloorPlanRenderer()
        print(f"[*] MTMCT Pipeline đã sẵn sàng (Tracker Config: {self.tracker_cfg})")

    def process_frame(self, frame, cam_id="Cam_1", current_time=None):
        """
        Xử lý 1 frame của 1 camera cụ thể.
        Trả về:
            annotated_frame: Frame đã vẽ BBox, Global ID và màu sắc
            detections_info: Danh sách các đối tượng phát hiện
        """
        if current_time is None:
            current_time = time.time()

        h, w = frame.shape[:2]
        res = self.detector.detect_and_track(frame, tracker_cfg=self.tracker_cfg)
        annotated_frame = frame.copy()
        detections_info = []

        if res is not None and res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            track_ids = res.boxes.id.int().cpu().numpy() if res.boxes.id is not None else None
            clses = res.boxes.cls.int().cpu().numpy()

            for i, box in enumerate(boxes):
                if clses[i] != 0:
                    continue  # Chỉ xử lý người

                x1, y1, x2, y2 = map(int, box)
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                if (x2 - x1) < 15 or (y2 - y1) < 30:
                    continue

                local_id = int(track_ids[i]) if track_ids is not None else (i + 1)
                crop = frame[y1:y2, x1:x2]

                gid, color, is_new, sim = self.global_manager.match_or_create(
                    cam_id=cam_id,
                    local_track_id=local_id,
                    crop_bgr=crop,
                    box=(x1, y1, x2, y2),
                    current_time=current_time
                )

                # Vẽ Bounding Box & nhãn
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{gid} (L:{local_id}) {sim:.2f}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - lh - 6), (x1 + lw + 6, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

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

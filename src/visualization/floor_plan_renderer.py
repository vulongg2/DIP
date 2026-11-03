"""
Floor Plan Renderer & Visualizer.
Vẽ sơ đồ mặt bằng 2D, hiển thị vị trí thời gian thực và quỹ đạo di chuyển xuyên camera.
"""

import cv2
import numpy as np


class FloorPlanRenderer:
    def __init__(self, width=480, height=360):
        self.width = width
        self.height = height
        self.zones = {
            "Cam_1": {"rect": (25, 40, 140, 240), "color": (230, 240, 255), "label": "Loi vao (Zone 1)"},
            "Cam_2": {"rect": (170, 80, 310, 200), "color": (255, 245, 230), "label": "Hanh lang (Zone 2)"},
            "Cam_3": {"rect": (340, 40, 455, 240), "color": (235, 255, 235), "label": "Phong hoc (Zone 3)"},
        }

    def render(self, active_positions, topology=None):
        """
        Vẽ sơ đồ mặt bằng khu vực 2D (Floor Plan) hiển thị vị trí và đường đi của các đối tượng.
        active_positions: dict {global_id: {"cam_id": ..., "color": ..., "box": ..., "center": ...}}
        """
        plan = np.ones((self.height, self.width, 3), dtype=np.uint8) * 245  # Nền xám trắng

        # Vẽ nền các phòng
        for cam_id, zinfo in self.zones.items():
            x1, y1, x2, y2 = zinfo["rect"]
            cv2.rectangle(plan, (x1, y1), (x2, y2), zinfo["color"], -1)
            cv2.rectangle(plan, (x1, y1), (x2, y2), (180, 180, 180), 2)
            cv2.putText(plan, zinfo["label"], (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

        # Vẽ đường hành lang nối giữa các phòng
        cv2.line(plan, (140, 140), (170, 140), (100, 100, 200), 2, cv2.LINE_AA)
        cv2.line(plan, (310, 140), (340, 140), (100, 100, 200), 2, cv2.LINE_AA)

        # Vẽ tiêu đề sơ đồ
        cv2.rectangle(plan, (0, 0), (self.width, 30), (50, 50, 50), -1)
        cv2.putText(plan, "2D INDOOR FLOOR PLAN - MULTI-ZONE TRAJECTORY", (15, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Vẽ vị trí các đối tượng đang hoạt động
        for gid, pos in active_positions.items():
            cam_id = pos["cam_id"]
            if cam_id not in self.zones:
                continue
            z_rect = self.zones[cam_id]["rect"]
            color = pos.get("color", (0, 255, 0))

            # Tính tọa độ trong phòng dựa theo tỷ lệ box
            rx1, ry1, rx2, ry2 = z_rect
            norm_x = pos.get("norm_x", 0.5)
            norm_y = pos.get("norm_y", 0.5)

            px = int(rx1 + 15 + norm_x * (rx2 - rx1 - 30))
            py = int(ry1 + 35 + norm_y * (ry2 - ry1 - 50))

            # Vẽ điểm đối tượng và viền phát sáng
            cv2.circle(plan, (px, py), 12, color, -1)
            cv2.circle(plan, (px, py), 15, (50, 50, 50), 2)
            cv2.putText(plan, f"ID {gid}", (px - 14, py - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

        # Legend chú thích bên dưới
        cv2.rectangle(plan, (0, self.height - 40), (self.width, self.height), (230, 230, 230), -1)
        cv2.putText(plan, "Red/Blue/Green: Global IDs | NetworkX Topology Graph Guard Active",
                    (10, self.height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1)

        return plan


def draw_hud_header(frame, title, fps, tracker_name="BoT-SORT+ReID"):
    """Vẽ thanh thông tin HUD trên đầu khung hình"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Engine: {tracker_name}", (w - 320, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return frame

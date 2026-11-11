"""
Floor Plan Renderer & Visualizer.
Vẽ sơ đồ mặt bằng 2D (2D Indoor Floor Plan), ánh xạ tọa độ chuẩn hóa từ khung hình camera
sang tọa độ mặt bằng thực tế và hiển thị vị trí thời gian thực xuyên camera.
"""

import cv2
import numpy as np


class FloorPlanRenderer:
    def __init__(self, width=480, height=360):
        """
        Khởi tạo bản đồ mặt bằng 2D kích thước 480x360 gồm 3 khu vực (Zones):
        - Cam_1 (Zone 1): Lối vào (Entrance)
        - Cam_2 (Zone 2): Hành lang (Hallway)
        - Cam_3 (Zone 3): Phòng học / Văn phòng (Office)
        """
        self.width = width
        self.height = height
        self.zones = {
            "Cam_1": {"rect": (25, 40, 140, 240), "color": (230, 240, 255), "label": "Loi vao (Zone 1)"},
            "Cam_2": {"rect": (170, 80, 310, 200), "color": (255, 245, 230), "label": "Hanh lang (Zone 2)"},
            "Cam_3": {"rect": (340, 40, 455, 240), "color": (235, 255, 235), "label": "Phong hoc (Zone 3)"},
        }

    def render(self, active_positions, topology=None):
        """
        Vẽ sơ đồ mặt bằng khu vực 2D (Floor Plan) hiển thị vị trí của các đối tượng.
        Đầu vào:
            active_positions: dict {global_id: {"cam_id": ..., "color": ..., "norm_x": ..., "norm_y": ...}}
        Đầu ra:
            plan: Ảnh BGR (360, 480, 3) biểu diễn bản đồ 2D thời gian thực.
        """
        # Tạo ảnh nền xám trắng (giá trị mức xám 245)
        plan = np.ones((self.height, self.width, 3), dtype=np.uint8) * 245

        # 1. Vẽ hình chữ nhật biểu diễn từng phòng/khu vực (Zone)
        for cam_id, zinfo in self.zones.items():
            x1, y1, x2, y2 = zinfo["rect"]
            cv2.rectangle(plan, (x1, y1), (x2, y2), zinfo["color"], -1)
            cv2.rectangle(plan, (x1, y1), (x2, y2), (180, 180, 180), 2)
            cv2.putText(plan, zinfo["label"], (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

        # 2. Vẽ đường nối biểu diễn cửa thông nhau giữa các khu vực (Edges trong Topology)
        cv2.line(plan, (140, 140), (170, 140), (100, 100, 200), 2, cv2.LINE_AA)
        cv2.line(plan, (310, 140), (340, 140), (100, 100, 200), 2, cv2.LINE_AA)

        # 3. Vẽ thanh tiêu đề trên cùng
        cv2.rectangle(plan, (0, 0), (self.width, 30), (50, 50, 50), -1)
        cv2.putText(plan, "2D INDOOR FLOOR PLAN - MULTI-ZONE TRAJECTORY", (15, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # 4. Ánh xạ tuyến tính tọa độ (norm_x, norm_y) của từng đối tượng vào bên trong ô phòng tương ứng
        for gid, pos in active_positions.items():
            cam_id = pos["cam_id"]
            if cam_id not in self.zones:
                continue
            z_rect = self.zones[cam_id]["rect"]
            color = pos.get("color", (0, 255, 0))

            # Phép biến đổi Affine 2D đơn giản: nội suy vị trí (px, py) trong hình chữ nhật (rx1, ry1, rx2, ry2)
            rx1, ry1, rx2, ry2 = z_rect
            norm_x = pos.get("norm_x", 0.5)
            norm_y = pos.get("norm_y", 0.5)

            px = int(rx1 + 15 + norm_x * (rx2 - rx1 - 30))
            py = int(ry1 + 35 + norm_y * (ry2 - ry1 - 50))

            # Vẽ chấm tròn đại diện cho người và nhãn Global ID
            cv2.circle(plan, (px, py), 12, color, -1)
            cv2.circle(plan, (px, py), 15, (50, 50, 50), 2)
            cv2.putText(plan, f"ID {gid}", (px - 14, py - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

        # 5. Vẽ thanh chú thích (Legend) ở cạnh đáy
        cv2.rectangle(plan, (0, self.height - 40), (self.width, self.height), (230, 230, 230), -1)
        cv2.putText(plan, "Red/Blue/Green: Global IDs | NetworkX Topology Graph Guard Active",
                    (10, self.height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1)

        return plan


def draw_hud_header(frame, title, fps, tracker_name="BoT-SORT+ReID"):
    """
    Vẽ thanh trạng thái bán trong suốt (Alpha Blending HUD) trên đầu khung hình camera.
    Công thức trộn ảnh số (Alpha Blending):
        I_out = 0.75 * I_overlay + 0.25 * I_frame
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Engine: {tracker_name}", (w - 320, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return frame

"""
Camera Topology Module using NetworkX.
Mô hình hóa đồ thị không gian - thời gian (Spatial-Temporal Topology Graph) giữa các khu vực camera trong nhà.
Loại bỏ trường hợp so khớp sai lệch (False Positive ReID) và dịch chuyển phi thực tế (Teleport) giữa các phòng.

========================================================================================
GIẢI THÍCH LÝ THUYẾT ĐỒ THỊ TRONG BÀI TOÁN LIÊN CAMERA (MTMCT):
- Hệ thống camera trong nhà được biểu diễn bằng đồ thị vô hướng có trọng số G = (V, E):
  + Tập đỉnh V (Nodes): Mỗi đỉnh là 1 Camera gắn với 1 khu vực (Zone), ví dụ:
    Cam_1 (Cửa vào - Zone 1) --- Cam_2 (Hành lang - Zone 2) --- Cam_3 (Văn phòng - Zone 3).
  + Tập cạnh E (Edges): Mỗi cạnh nối 2 camera có lối đi thông nhau trực tiếp, mang trọng số:
    * `min_sec` (T_min): Thời gian tối thiểu để đi bộ từ camera này sang camera kia.
    * `max_sec` (T_max): Thời gian tối đa cho phép trước khi coi là đối tượng đã rời khỏi khu vực.
- Khi 1 người biến mất ở `prev_cam` tại thời điểm t_1 và xuất hiện ở `new_cam` tại t_2:
  + Độ trễ chuyển tiếp: delta_t = t_2 - t_1.
  + Tìm đường đi ngắn nhất (Shortest Path) trên đồ thị từ `prev_cam` đến `new_cam`.
  + Cộng tổng [T_min, T_max] dọc theo các cạnh trên đường đi.
  + Nếu delta_t < T_min (đi nhanh hơn tốc độ người chạy -> Teleport phi vật lý) -> LOẠI BỎ!
  + Nếu delta_t > T_max (biến mất quá lâu) -> LOẠI BỎ!
========================================================================================
"""

import os
import json
import networkx as nx


class CameraTopology:
    def __init__(self, config_path=None):
        """
        Khởi tạo đồ thị Topology bằng thư viện NetworkX.
        Nếu có file cấu hình JSON (`configs/topology.json`), đọc từ file;
        ngược lại tự động tạo cấu hình 3 khu vực trong nhà mặc định.
        """
        self.graph = nx.Graph()
        if config_path and os.path.exists(config_path):
            self._load_from_json(config_path)
        else:
            self._build_default_topology()

    def _build_default_topology(self):
        """
        Xây dựng đồ thị 3 camera trong nhà mặc định:
        Cam_1 (Khu vực cửa / Entrance) <---> Cam_2 (Hành lang / Hallway) <---> Cam_3 (Phòng làm việc / Office)
        Chú ý: Cam_1 và Cam_3 KHÔNG nối trực tiếp với nhau mà phải đi qua Cam_2.
        """
        # 1. Thêm các đỉnh (Nodes) đại diện cho từng vùng camera
        self.graph.add_node("Cam_1", name="Entrance (Cua vao)", zone_id=1, coords=(100, 200))
        self.graph.add_node("Cam_2", name="Hallway (Hanh lang)", zone_id=2, coords=(300, 200))
        self.graph.add_node("Cam_3", name="Office (Van phong)", zone_id=3, coords=(500, 200))

        # Ràng buộc thời gian di chuyển (giây) giữa các camera liền kề:
        # min_sec: thời gian tối thiểu (0.2s để chấp nhận cả khi 2 cam gần nhau)
        # max_sec: thời gian tối đa (300s = 5 phút để thoải mái đi lại giữa các phòng)
        self.graph.add_edge("Cam_1", "Cam_2", min_sec=0.2, max_sec=300.0, distance_m=10.0)
        self.graph.add_edge("Cam_2", "Cam_3", min_sec=0.2, max_sec=300.0, distance_m=12.0)

    def _load_from_json(self, config_path):
        """Đọc cấu trúc đỉnh (nodes) và cạnh (edges) từ file JSON cấu hình."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for node in data.get("nodes", []):
                self.graph.add_node(
                    node["id"],
                    name=node.get("name", node["id"]),
                    zone_id=node.get("zone_id", 1),
                    coords=tuple(node.get("coords", (100, 100)))
                )
            for edge in data.get("edges", []):
                self.graph.add_edge(
                    edge["source"],
                    edge["target"],
                    min_sec=edge.get("min_sec", 0.5),
                    max_sec=edge.get("max_sec", 60.0),
                    distance_m=edge.get("distance_m", 10.0)
                )
        except Exception as e:
            print(f"[!] Loi doc topology json ({e}), dung cau hinh mac dinh.")
            self._build_default_topology()

    def is_adjacent(self, cam_a, cam_b):
        """Kiểm tra 2 camera có lối đi nối trực tiếp với nhau trên đồ thị không."""
        if cam_a == cam_b:
            return True
        return self.graph.has_edge(cam_a, cam_b)

    def is_transition_valid(self, prev_cam, new_cam, time_delta_sec):
        """
        Kiểm tra chuyển tiếp của đối tượng từ `prev_cam` sang `new_cam` sau khoảng thời gian `time_delta_sec`
        có hợp lệ về mặt vật lý (không gian - thời gian) hay không.

        Trả về:
            True  - Hợp lệ, cho phép so khớp ReID Cosine Similarity.
            False - Vi phạm vật lý (dịch chuyển tức thời hoặc quá hạn), chặn không cho gán cùng Global ID.
        """
        # Trường hợp 1: Xuất hiện lại trên CÙNG 1 camera (ví dụ bị khuất sau cột nhà rồi bước ra)
        if prev_cam == new_cam:
            return time_delta_sec <= 120.0

        # Nếu tên camera không nằm trong đồ thị thì mặc định cho phép (Fallback an toàn)
        if not self.graph.has_node(prev_cam) or not self.graph.has_node(new_cam):
            return True

        # Trường hợp 2: Di chuyển KHÁC camera -> Tìm đường đi ngắn nhất bằng thuật toán Dijkstra/BFS của NetworkX
        try:
            path = nx.shortest_path(self.graph, source=prev_cam, target=new_cam)
            total_min_sec = 0.0
            total_max_sec = 0.0
            # Cộng dồn ngưỡng thời gian tối thiểu và tối đa dọc theo hành trình các cạnh
            for i in range(len(path) - 1):
                edge_data = self.graph[path[i]][path[i + 1]]
                total_min_sec += edge_data.get("min_sec", 0.5)
                total_max_sec += edge_data.get("max_sec", 60.0)

            # Kiểm tra điều kiện biên [total_min_sec, total_max_sec]
            if time_delta_sec < total_min_sec:
                return False  # Loại bỏ: Đi nhanh hơn giới hạn vật lý (Teleport giữa các phòng)
            if time_delta_sec > total_max_sec:
                return False  # Loại bỏ: Đã biến mất quá lâu
            return True
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # Không tồn tại đường đi nối giữa 2 khu vực này
            return False

    def get_zone_info(self, cam_id):
        """Lấy thông tin tên khu vực và tọa độ 2D của camera."""
        if self.graph.has_node(cam_id):
            return self.graph.nodes[cam_id]
        return {"name": f"Camera {cam_id}", "zone_id": 0, "coords": (0, 0)}

    def get_all_cameras(self):
        """Trả về danh sách tất cả các camera trong hệ thống."""
        return list(self.graph.nodes())

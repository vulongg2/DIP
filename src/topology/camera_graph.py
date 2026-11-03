"""
Camera Topology Module using NetworkX.
Mô hình hóa đồ thị không gian - thời gian giữa các khu vực camera trong nhà.
Loại bỏ trường hợp so khớp sai lệch và dịch chuyển phi thực tế giữa các phòng.
"""

import os
import json
import networkx as nx


class CameraTopology:
    def __init__(self, config_path=None):
        self.graph = nx.Graph()
        if config_path and os.path.exists(config_path):
            self._load_from_json(config_path)
        else:
            self._build_default_topology()

    def _build_default_topology(self):
        """
        Xây dựng đồ thị 3 camera trong nhà mặc định:
        Cam_1 (Khu vực cửa / Entrance) <---> Cam_2 (Hành lang / Hallway) <---> Cam_3 (Phòng làm việc / Office)
        """
        self.graph.add_node("Cam_1", name="Entrance (Cửa vào)", zone_id=1, coords=(100, 200))
        self.graph.add_node("Cam_2", name="Hallway (Hành lang)", zone_id=2, coords=(300, 200))
        self.graph.add_node("Cam_3", name="Office (Văn phòng)", zone_id=3, coords=(500, 200))

        # Ràng buộc thời gian di chuyển (giây) giữa các camera liền kề
        self.graph.add_edge("Cam_1", "Cam_2", min_sec=0.5, max_sec=60.0, distance_m=10.0)
        self.graph.add_edge("Cam_2", "Cam_3", min_sec=0.5, max_sec=60.0, distance_m=12.0)

    def _load_from_json(self, config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for node in data.get("nodes", []):
                self.graph.add_node(node["id"], name=node.get("name", node["id"]), zone_id=node.get("zone_id", 1), coords=tuple(node.get("coords", (100, 100))))
            for edge in data.get("edges", []):
                self.graph.add_edge(edge["source"], edge["target"], min_sec=edge.get("min_sec", 0.5), max_sec=edge.get("max_sec", 60.0), distance_m=edge.get("distance_m", 10.0))
        except Exception as e:
            print(f"[!] Lỗi đọc topology json ({e}), dùng cấu hình mặc định.")
            self._build_default_topology()

    def is_adjacent(self, cam_a, cam_b):
        """Kiểm tra 2 camera có nối trực tiếp với nhau không."""
        if cam_a == cam_b:
            return True
        return self.graph.has_edge(cam_a, cam_b)

    def is_transition_valid(self, prev_cam, new_cam, time_delta_sec):
        """
        Kiểm tra chuyển tiếp giữa 2 camera có hợp lệ về mặt không gian - thời gian không.
        - Cùng 1 camera: mất dấu tối đa 120s.
        - Khác camera: kiểm tra có đường đi trong đồ thị và time_delta thuộc [min_sec, max_sec].
        """
        if prev_cam == new_cam:
            return time_delta_sec <= 120.0

        if not self.graph.has_node(prev_cam) or not self.graph.has_node(new_cam):
            return True  # Fallback nếu camera chưa đăng ký trên đồ thị

        try:
            path = nx.shortest_path(self.graph, source=prev_cam, target=new_cam)
            total_min_sec = 0.0
            total_max_sec = 0.0
            for i in range(len(path) - 1):
                edge_data = self.graph[path[i]][path[i + 1]]
                total_min_sec += edge_data.get("min_sec", 0.5)
                total_max_sec += edge_data.get("max_sec", 60.0)

            if time_delta_sec < total_min_sec:
                return False  # Teleport phi thực tế giữa các phòng
            if time_delta_sec > total_max_sec:
                return False  # Biến mất quá lâu
            return True
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return False

    def get_zone_info(self, cam_id):
        if self.graph.has_node(cam_id):
            return self.graph.nodes[cam_id]
        return {"name": f"Camera {cam_id}", "zone_id": 0, "coords": (0, 0)}

    def get_all_cameras(self):
        return list(self.graph.nodes())

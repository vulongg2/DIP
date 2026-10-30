"""
Camera Topology Module sử dụng NetworkX.
Mô hình hóa đồ thị không gian - thời gian giữa các khu vực camera trong nhà.
Giúp loại bỏ việc so khớp ReID sai lệch (ví dụ đối tượng không thể di chuyển tức thì giữa 2 phòng không kề nhau).
"""

import networkx as nx


class CameraTopology:
    def __init__(self):
        # Đồ thị vô hướng / có hướng mô tả các camera và đường đi giữa các phòng
        self.graph = nx.Graph()
        self._build_default_topology()

    def _build_default_topology(self):
        """
        Xây dựng đồ thị 3 camera trong nhà mặc định:
        Cam_1 (Cửa vào / Entrance) <---> Cam_2 (Hành lang / Hallway) <---> Cam_3 (Phòng / Office)
        """
        # Thêm các Node (Camera kèm tên khu vực)
        self.graph.add_node("Cam_1", name="Entrance (Khu vực cửa)", zone_id=1, coords=(100, 200))
        self.graph.add_node("Cam_2", name="Hallway (Hành lang chính)", zone_id=2, coords=(300, 200))
        self.graph.add_node("Cam_3", name="Office (Phòng làm việc)", zone_id=3, coords=(500, 200))

        # Thêm các Edge kèm ràng buộc thời gian di chuyển (giây)
        # min_sec: thời gian tối thiểu người đi bộ bình thường mất để đi từ cam này sang cam kia
        # max_sec: thời gian tối đa còn liên kết (quá thời gian này coi như đối tượng đã rời hẳn khỏi tòa nhà)
        self.graph.add_edge("Cam_1", "Cam_2", min_sec=0.5, max_sec=60.0, distance_m=10.0)
        self.graph.add_edge("Cam_2", "Cam_3", min_sec=0.5, max_sec=60.0, distance_m=12.0)

    def is_adjacent(self, cam_a, cam_b):
        """Kiểm tra 2 camera có nối trực tiếp với nhau không."""
        if cam_a == cam_b:
            return True
        return self.graph.has_edge(cam_a, cam_b)

    def is_transition_valid(self, prev_cam, new_cam, time_delta_sec):
        """
        Kiểm tra chuyển tiếp giữa 2 camera có hợp lệ về mặt không gian - thời gian không.
        - Nếu cùng 1 camera: hợp lệ nếu trong khoảng thời gian cho phép.
        - Nếu khác camera: kiểm tra có đường đi trong đồ thị và thời gian delta có nằm trong [min_sec, max_sec] không.
        """
        if prev_cam == new_cam:
            return time_delta_sec <= 120.0  # Mất dấu tối đa 2 phút trong cùng 1 camera

        if not self.graph.has_node(prev_cam) or not self.graph.has_node(new_cam):
            return True  # Fallback nếu tên camera không nằm trong cấu hình

        try:
            # Tìm đường đi ngắn nhất
            path = nx.shortest_path(self.graph, source=prev_cam, target=new_cam)
            # Tính tổng min_sec trên đường đi
            total_min_sec = 0.0
            total_max_sec = 0.0
            for i in range(len(path) - 1):
                edge_data = self.graph[path[i]][path[i + 1]]
                total_min_sec += edge_data.get("min_sec", 0.5)
                total_max_sec += edge_data.get("max_sec", 60.0)

            # Ràng buộc không-thời gian:
            if time_delta_sec < total_min_sec:
                # Đi quá nhanh (teleport phi thực tế giữa các phòng)
                return False
            if time_delta_sec > total_max_sec:
                # Biến mất quá lâu
                return False
            return True
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return False

    def get_zone_info(self, cam_id):
        """Lấy thông tin mô tả khu vực."""
        if self.graph.has_node(cam_id):
            return self.graph.nodes[cam_id]
        return {"name": f"Camera {cam_id}", "zone_id": 0, "coords": (0, 0)}

    def get_all_cameras(self):
        return list(self.graph.nodes())

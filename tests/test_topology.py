import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.topology.camera_graph import CameraTopology


class TestTopology(unittest.TestCase):
    def setUp(self):
        self.topo = CameraTopology()

    def test_default_cameras(self):
        cams = self.topo.get_all_cameras()
        self.assertIn("Cam_1", cams)
        self.assertIn("Cam_2", cams)
        self.assertIn("Cam_3", cams)

    def test_adjacency(self):
        self.assertTrue(self.topo.is_adjacent("Cam_1", "Cam_2"))
        self.assertTrue(self.topo.is_adjacent("Cam_2", "Cam_3"))
        self.assertFalse(self.topo.is_adjacent("Cam_1", "Cam_3"))

    def test_transition_validity(self):
        # Đi quá nhanh từ Cam 1 sang Cam 3 (khoảng cách cần ít nhất 1.0s)
        self.assertFalse(self.topo.is_transition_valid("Cam_1", "Cam_3", 0.1))

        # Đi trong khoảng thời gian hợp lý (ví dụ 5.0s)
        self.assertTrue(self.topo.is_transition_valid("Cam_1", "Cam_3", 5.0))

        # Biến mất quá lâu (> 120s)
        self.assertFalse(self.topo.is_transition_valid("Cam_1", "Cam_3", 150.0))


if __name__ == "__main__":
    unittest.main()

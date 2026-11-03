import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.association.global_manager import GlobalIdentityManager


class TestAssociation(unittest.TestCase):
    def setUp(self):
        self.manager = GlobalIdentityManager(similarity_thresh=0.65)

    def test_new_identity_creation(self):
        dummy_crop = np.zeros((100, 50, 3), dtype=np.uint8)
        box = (10, 10, 60, 110)
        gid, color, is_new, sim = self.manager.match_or_create(
            cam_id="Cam_1",
            local_track_id=1,
            crop_bgr=dummy_crop,
            box=box
        )
        self.assertEqual(gid, 1)
        self.assertTrue(is_new)

    def test_spatial_mutual_exclusion(self):
        dummy_crop = np.zeros((100, 50, 3), dtype=np.uint8)
        
        # Người thứ nhất xuất hiện
        gid1, _, _, _ = self.manager.match_or_create("Cam_1", 1, dummy_crop, (10, 10, 60, 110))
        
        # Người thứ hai xuất hiện cùng lúc trong Cam_1 nhưng local track khác
        # Dù ảnh giống hệt (crop giống nhau), không được phép gán chung ID
        gid2, _, is_new2, _ = self.manager.match_or_create("Cam_1", 2, dummy_crop, (70, 10, 120, 110))
        
        self.assertNotEqual(gid1, gid2)
        self.assertTrue(is_new2)


if __name__ == "__main__":
    unittest.main()

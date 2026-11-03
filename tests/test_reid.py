import os
import sys
import unittest
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.reid.custom_reid_net import CustomReIDNet, TripletLoss
from src.reid.reid_extractor import ReIDExtractor


class TestReID(unittest.TestCase):
    def test_forward_dimension(self):
        model = CustomReIDNet(embedding_dim=512, pretrained=False)
        model.eval()
        dummy_input = torch.randn(2, 3, 256, 128)
        with torch.no_grad():
            output = model(dummy_input)
        self.assertEqual(output.shape, (2, 512))
        
        # Kiểm tra L2 normalized (norm xấp xỉ 1.0)
        norm = torch.norm(output, p=2, dim=1)
        self.assertTrue(torch.allclose(norm, torch.ones_like(norm), atol=1e-5))

    def test_triplet_loss(self):
        loss_fn = TripletLoss(margin=0.3)
        features = torch.tensor([
            [1.0, 0.0],
            [0.9, 0.1],
            [-1.0, 0.0]
        ], dtype=torch.float32)
        labels = torch.tensor([0, 0, 1], dtype=torch.long)
        loss = loss_fn(features, labels)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_extractor_dummy_crop(self):
        extractor = ReIDExtractor(backend="custom")
        dummy_crop = np.zeros((100, 50, 3), dtype=np.uint8)
        feat = extractor.extract_crop_feature(dummy_crop)
        self.assertIsNotNone(feat)
        self.assertEqual(len(feat), 512)
        norm = np.linalg.norm(feat)
        self.assertAlmostEqual(norm, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()

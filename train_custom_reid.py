"""
Script Huấn luyện / Fine-tune Customized ReID Model với Metric Learning (Triplet Loss + BNNeck).
Đáp ứng yêu cầu 'customized model' và 'metric learning' trong đề bài BTL DIP.

Huấn luyện trực tiếp trên các ảnh crop đã thu thập trong outputs/crops/
hoặc bất kỳ bộ dataset ReID nào (Market-1501, Duke, dữ liệu tự thu thập).
"""

import os
import sys
import glob
import time

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from core.custom_reid_net import CustomReIDNet, TripletLoss


class CropReIDDataset(Dataset):
    """Dataset đọc các folder ảnh crop theo person_id: outputs/crops/<track_id>/*.jpg"""
    def __init__(self, root_dir="outputs/crops", transform=None):
        self.samples = []  # list of (img_path, label_idx)
        self.transform = transform

        person_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.label_map = {d: idx for idx, d in enumerate(person_dirs)}

        for p_dir in person_dirs:
            p_path = os.path.join(root_dir, p_dir)
            images = glob.glob(os.path.join(p_path, "*.jpg"))
            for img_p in images:
                self.samples.append((img_p, self.label_map[p_dir]))

        print(f"[*] Đã tải {len(self.samples)} ảnh từ {len(person_dirs)} danh tính (classes).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((256, 128, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print(f"[*] HUẤN LUYỆN CUSTOMIZED REID MODEL VỚI METRIC LEARNING")
    print(f"[*] Thiết bị: {device}")
    print("=" * 65)

    data_dir = "outputs/crops"
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print(f"[!] Không tìm thấy dữ liệu trong {data_dir}. Hãy chạy localv.py hoặc live_cam.py trước để có ảnh crop.")
        return

    # Transforms
    train_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 128)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CropReIDDataset(root_dir=data_dir, transform=train_transform)
    num_classes = max(1, len(dataset.label_map))
    dataloader = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=True, drop_last=False)

    # Khởi tạo model và loss
    model = CustomReIDNet(embedding_dim=512, num_classes=num_classes, pretrained=True).to(device)
    model.train()

    criterion_id = nn.CrossEntropyLoss()
    criterion_triplet = TripletLoss(margin=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=5e-4)

    epochs = 5
    print(f"\n[*] Bắt đầu huấn luyện {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0.0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            feat_norm, cls_score = model(imgs)

            loss_id = criterion_id(cls_score, labels)
            # Triplet loss nếu batch có nhiều hơn 1 class
            loss_triplet = criterion_triplet(feat_norm, labels) if len(torch.unique(labels)) > 1 else torch.tensor(0.0, device=device)
            
            loss = loss_id + loss_triplet
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_l = total_loss / max(1, len(dataloader))
        print(f"  Epoch [{epoch+1}/{epochs}] - Loss: {avg_l:.4f}")

    # Lưu trọng số customized model
    save_path = "custom_reid_net.pth"
    torch.save(model.state_dict(), save_path)
    print("=" * 65)
    print(f"[*] HUẤN LUYỆN HOÀN TẤT! Đã lưu model tại: {save_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()

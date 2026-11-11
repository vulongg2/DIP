"""
Customized Person Re-Identification Neural Network with Metric Learning (Module 4).
Đáp ứng trực tiếp yêu cầu 'customized model' và 'metric learning' của đề bài BTL DIP.

Kiến trúc mạng 5 tầng:
1. Backbone: MobileNetV3-Small Feature Extractor (Nhẹ, tối ưu cho thời gian thực > 50 FPS).
2. Global Average Pooling (GAP): Nén không gian Feature Map [B, 576, 8, 4] xuống [B, 576, 1, 1].
3. BNNeck (Batch Normalization Neck): Tách biệt không gian đặc trưng cho Metric Loss và ID Loss.
4. Metric Learning Projection: Chiếu lên mặt cầu đơn vị 512 chiều (L2-normalized, ||f||_2 = 1.0).
5. Hard-Mining Triplet Loss + Cross-Entropy Loss: Kéo gần ảnh cùng người, đẩy xa ảnh khác người.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MetricLearningHead(nn.Module):
    """
    Cổ chai BNNeck (Batch Normalization Neck) chuẩn cho bài toán Person ReID:
    - Tắt hệ số tự do bias (bias=False) ở cả lớp Linear và BatchNorm để ép các siêu phẳng
      phân lớp đi qua gốc tọa độ O, giúp tối ưu hóa góc cos(θ) trên mặt cầu đơn vị L2.
    """
    def __init__(self, in_features, embedding_dim=512, num_classes=None):
        super().__init__()
        # Lớp chiếu tuyến tính giảm chiều từ 576 xuống 512 chiều (không dùng bias)
        self.fc = nn.Linear(in_features, embedding_dim, bias=False)
        
        # Lớp chuẩn hóa theo lô (BatchNorm1d) làm cổ chai điều hòa phân bố đặc trưng
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.bn.bias.requires_grad_(False)  # Đóng băng bias của BatchNorm theo chuẩn BNNeck
        
        # Nhánh phân loại ID (chỉ kích hoạt trong quá trình huấn luyện nếu có nhãn num_classes > 0)
        if num_classes is not None and num_classes > 0:
            self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
        else:
            self.classifier = None

    def forward(self, x):
        # Input x: Tensor [Batch, 576] từ tầng Global Average Pooling
        feat = self.fc(x)           # Chiếu tuyến tính -> [Batch, 512]
        feat_norm = self.bn(feat)   # Đi qua cổ chai BatchNorm -> [Batch, 512]

        # Chế độ Huấn luyện (Training): Trả về cả vector đặc trưng (cho Triplet Loss) và điểm phân lớp (cho CrossEntropy)
        if self.training and self.classifier is not None:
            cls_score = self.classifier(feat_norm)
            return feat_norm, cls_score

        # Chế độ Suy luận (Inference): Chuẩn hóa L2 đưa vector về mặt cầu đơn vị có độ dài = 1.0
        # Công thức: f_unit = f / ||f||_2 -> Giúp tính Cosine Similarity chỉ bằng tích vô hướng
        return F.normalize(feat_norm, p=2, dim=1)


class CustomReIDNet(nn.Module):
    """
    Mạng nơ-ron sâu ReID tùy biến (Customized Model) cho hệ thống theo dõi liên camera trong nhà.
    - Input: Tensor ảnh người chuẩn hóa kích thước [Batch, 3, 256, 128] (Cao 256, Rộng 128).
    - Output: Vector đặc trưng 512 chiều chuẩn hóa L2 [Batch, 512].
    """
    def __init__(self, embedding_dim=512, num_classes=0, pretrained=True):
        super().__init__()
        # 1. Nạp mạng xương sống MobileNetV3-Small trích xuất đặc trưng thị giác (màu sắc, hoa văn, dáng người)
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        self.features = backbone.features  # Đầu ra có kích thước [Batch, 576, 8, 4] với ảnh vào 256x128
        
        in_channels = 576  # Số kênh đặc trưng đầu ra của MobileNetV3-Small
        
        # 2. Tầng Gộp trung bình toàn cục (Global Average Pooling - Chương 6 DIP: Nén & Giảm chiều)
        # Nén mỗi mặt phẳng đặc trưng [8 x 4] thành 1 giá trị trung bình duy nhất [1 x 1]
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # 3. Đưa vào cổ chai BNNeck để xuất ra vector 512 chiều
        self.head = MetricLearningHead(in_channels, embedding_dim=embedding_dim, num_classes=num_classes)

    def forward(self, x):
        # x: [Batch, 3, 256, 128] (Chuẩn tỷ lệ chiều cao/rộng 2:1 của dáng người đứng)
        feat_map = self.features(x)             # -> [Batch, 576, 8, 4]
        pooled = self.gap(feat_map)             # -> [Batch, 576, 1, 1]
        flattened = torch.flatten(pooled, 1)    # Duỗi phẳng thành vector 1 chiều -> [Batch, 576]
        return self.head(flattened)             # -> [Batch, 512]


class TripletLoss(nn.Module):
    """
    Hàm mất mát Hard-Mining Triplet Loss chuẩn cho Metric Learning trong Person Re-Identification.
    Công thức toán học:
        L = max( d(Anchor, Hard_Positive) - d(Anchor, Hard_Negative) + margin, 0 )
    Mục tiêu:
        - Kéo khoảng cách giữa các ảnh của CÙNG 1 NGƯỜI (d_ap) lại gần bằng 0.
        - Đẩy khoảng cách giữa các ảnh của 2 NGƯỜI KHÁC NHAU (d_an) ra xa nhau ít nhất một đoạn bằng margin (0.3).
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        - inputs: Batch các vector đặc trưng [N, 512].
        - targets: Nhãn ID thật của từng ảnh trong batch [N].
        """
        n = inputs.size(0)
        
        # BƯỚC 1: Tính ma trận khoảng cách Euclidean giữa mọi cặp ảnh (i, j) trong batch [N x N]
        # Áp dụng hằng đẳng thức: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*(a · b)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # Kẹp ngưỡng dương 1e-12 trước khi căn bậc hai để tránh lỗi NaN

        # BƯỚC 2: Tìm mẫu khó nhất (Hard-Mining) cho từng ảnh gốc i (Anchor)
        # mask[i, j] = True nếu ảnh i và ảnh j thuộc CÙNG 1 NGƯỜI (targets[i] == targets[j])
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            # Hard Positive: Ảnh cùng người với i nhưng có khoảng cách LỚN NHẤT (.max())
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            # Hard Negative: Ảnh khác người với i nhưng lại trông GIỐNG NHẤT, khoảng cách NHỎ NHẤT (.min())
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # BƯỚC 3: Tính hàm mất mát MarginRankingLoss = max(0, dist_ap - dist_an + margin)
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

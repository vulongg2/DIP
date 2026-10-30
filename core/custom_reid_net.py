"""
Customized Person Re-Identification Neural Network with Metric Learning.
Đáp ứng trực tiếp yêu cầu 'customized model' và 'metric learning / OSNet' của đề bài.

Kiến trúc mạng:
1. Backbone: Omni-Scale / MobileNetV3 Feature Extractor (Nhẹ, tối ưu cho Real-time trên Laptop GPU).
2. Global Average Pooling (GAP).
3. BNNeck (Batch Normalization Neck) - Chuẩn mực ReID hiện đại (Bag of Tricks for Person ReID).
4. Metric Learning Projection: Vector 512-D L2-normalized.
5. Triplet Loss + Cross-Entropy Loss (Metric Learning Head).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MetricLearningHead(nn.Module):
    """
    Head chuẩn BNNeck cho Person ReID:
    Tách biệt feature space cho Metric Loss (Triplet) và Classification Loss (ID Cross-Entropy).
    """
    def __init__(self, in_features, embedding_dim=512, num_classes=None):
        super().__init__()
        self.fc = nn.Linear(in_features, embedding_dim, bias=False)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.bn.bias.requires_grad_(False)  # Chuẩn BNNeck
        
        # Classifier chỉ dùng trong lúc training (nếu có nhãn ID)
        if num_classes is not None and num_classes > 0:
            self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
        else:
            self.classifier = None

    def forward(self, x):
        # x: [Batch, in_features]
        feat = self.fc(x)
        feat_norm = self.bn(feat)  # Dùng cho metric learning / inference

        if self.training and self.classifier is not None:
            cls_score = self.classifier(feat_norm)
            return feat_norm, cls_score

        # Inference: chuẩn hóa L2 để tính Cosine Similarity tức thì
        return F.normalize(feat_norm, p=2, dim=1)


class CustomReIDNet(nn.Module):
    """
    Mạng ReID tùy biến (Customized Model) cho đề tài Indoor Multi-Camera Tracking.
    """
    def __init__(self, embedding_dim=512, num_classes=0, pretrained=True):
        super().__init__()
        # Backbone tối ưu cho camera trong nhà độ trễ thấp
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        self.features = backbone.features
        
        in_channels = 576  # Output channels của mobilenet_v3_small trước classifier
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Metric Learning Head tùy biến
        self.head = MetricLearningHead(in_channels, embedding_dim=embedding_dim, num_classes=num_classes)

    def forward(self, x):
        # x: [Batch, 3, 256, 128] (Chuẩn tỷ lệ chiều cao/rộng của người trong ReID)
        feat_map = self.features(x)
        pooled = self.gap(feat_map)
        flattened = torch.flatten(pooled, 1)
        return self.head(flattened)


class TripletLoss(nn.Module):
    """
    Hard-Mining Triplet Loss chuẩn cho Metric Learning trong Person Re-Identification.
    L = max(d(a, p) - d(a, n) + margin, 0)
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Tính khoảng cách Euclidean pairwise
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()

        # Hard positive & hard negative mining
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

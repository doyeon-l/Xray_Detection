import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from PIL import Image
import os

# 모델 클래스 정의
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        efficientnet = efficientnet_b3(weights=None)

        # features 그대로 사용
        self.features = efficientnet.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        in_features = efficientnet.classifier[1].in_features
        # Dropout + Linear 그대로 복원
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import os

# 모델 클래스 정의
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        efficientnet = efficientnet_b0(weights=weights)

        self.features = efficientnet.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

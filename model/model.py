# ==========================================================
# models.py 파일의 전체 내용
# ==========================================================

import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim

from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b1, EfficientNet_B1_Weights,
    efficientnet_b2, EfficientNet_B2_Weights,
    efficientnet_b3, EfficientNet_B3_Weights
)


class EfficientNetAutoencoder(nn.Module):
    """
    EfficientNet을 인코더로 사용하는 오토인코더 모델입니다.
    """
    def __init__(self, model_version='b0', output_size=224):
        super(EfficientNetAutoencoder, self).__init__()

        # ---------------------------------------------------------------
        # 1. 인코더 (Encoder) 설정
        # ---------------------------------------------------------------
        if model_version == 'b0':
            weights = EfficientNet_B0_Weights.DEFAULT
            efficientnet = efficientnet_b0(weights=weights)
            encoder_output_dim = 1280
        elif model_version == 'b1':
            weights = EfficientNet_B1_Weights.DEFAULT
            efficientnet = efficientnet_b1(weights=weights)
            encoder_output_dim = 1280
        elif model_version == 'b2':
            weights = EfficientNet_B2_Weights.DEFAULT
            efficientnet = efficientnet_b2(weights=weights)
            encoder_output_dim = 1408
        elif model_version == 'b3':
            # 🔴 b3 버전 추가
            weights = EfficientNet_B3_Weights.DEFAULT
            efficientnet = efficientnet_b3(weights=weights)
            encoder_output_dim = 1536
        else:
            raise ValueError(f"지원하지 않는 EfficientNet 버전입니다: {model_version}. 'b0', 'b1', 'b2', 'b3' 중에서 선택하세요.")

        print(f"✅ EfficientNet-{model_version} 기반 오토인코더를 생성합니다. (인코더 출력 차원: {encoder_output_dim})")

        self.encoder = nn.Sequential(
            efficientnet.features,
            efficientnet.avgpool
        )
        
        # ---------------------------------------------------------------
        # 2. 디코더 (Decoder) 설정
        # ---------------------------------------------------------------
        if output_size == 224:
            # 224x224 디코더 구조
            self.decoder = self._build_decoder(encoder_output_dim, start_size=7, num_upsample=5)  # ✅ num_upsample 추가
        elif output_size == 256:
            # 256x256 디코더 구조
            self.decoder = self._build_decoder(encoder_output_dim, start_size=8, num_upsample=5)  # ✅ num_upsample 추가
        elif output_size == 384:
            # 384x384 디코더 구조
            self.decoder = self._build_decoder(encoder_output_dim, start_size=12, num_upsample=6) # ✅ num_upsample 추가
        else:
            raise ValueError("지원하지 않는 이미지 크기입니다. 224, 256, 384 중에서 사용하세요.")

    # 🔴 수정 부분: 디코더 빌드 함수를 고정된 레이어 목록으로 변경
    def _build_decoder(self, encoder_output_dim, start_size, num_upsample):
        if start_size == 7: # 224x224
            decoder_layers = [
                nn.Linear(encoder_output_dim, 512 * 7 * 7),
                nn.ReLU(True),
                nn.Unflatten(1, (512, 7, 7)),
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid()
            ]
        elif start_size == 8: # 256x256
            decoder_layers = [
                nn.Linear(encoder_output_dim, 512 * 8 * 8),
                nn.ReLU(True),
                nn.Unflatten(1, (512, 8, 8)),
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid()
            ]
        elif start_size == 12: # 384x384
            decoder_layers = [
                nn.Linear(encoder_output_dim, 512 * 12 * 12),
                nn.ReLU(True),
                nn.Unflatten(1, (512, 12, 12)),
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(True),
                # 🔴 이 부분에서 32 채널에서 3채널로 바로 가는 것이 올바릅니다.
                # 🔴 이전 오류는 여기에 불필요한 레이어가 추가되어 발생했습니다.
                nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid()
            ]
        else:
            raise ValueError("지원하지 않는 start_size입니다.")
        
        return nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


class MSSSIMLoss(nn.Module):
    """MS-SSIM 손실 함수 클래스입니다."""
    def __init__(self, window_size=11, size_average=True):
        super(MSSSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        
    def forward(self, img1, img2):
        return 1 - ms_ssim(img1, img2, data_range=1.0, size_average=self.size_average, win_size=self.window_size)
import torch
import torch.nn as nn

# 다른 파일에서 클래스들을 불러옵니다.
from .selector import NeuralFeatureRanker
from .subsets import HierarchicalSubsetBuilder  # 파일명이 subsets.py인 경우


class BSFSNet(nn.Module):
    def __init__(self, input_dim, num_classes, k_list, K_fc=8):
        super().__init__()
        # 이제 NeuralFeatureRanker를 정상적으로 참조할 수 있습니다.
        self.selector = NeuralFeatureRanker(input_dim, K_fc)
        self.subset_builder = HierarchicalSubsetBuilder(k_list)

        # Shared Backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, num_classes)
        )

    def forward(self, x, tau=1.0):
        B, F = x.shape
        S = self.selector(x)  # S: (B, K_fc, F) -> (B, 8, F)
        M = self.subset_builder(S, tau=tau)  # M: (B, K_fc, K_sb, F) -> (B, 8, 10, F)

        # M의 실제 차원을 명확히 추출
        # B: 배치, K_fc: 8, K_sb: 10, F: 78
        B_m, K_fc, K_sb, F_m = M.shape

        # 1. x를 (B, 1, 1, F)로 변환하여 브로드캐스팅 준비
        # unsqueeze(1) -> (B, 1, F)
        # unsqueeze(2) -> (B, 1, 1, F)
        x_reshaped = x.unsqueeze(1).unsqueeze(2)

        # 2. 브로드캐스팅 곱셈 (B, 1, 1, F) * (B, 8, 10, F)
        # PyTorch가 자동으로 (B, 8, 10, F)로 맞춰서 계산합니다.
        x_masked = x_reshaped * M  # 결과: (B, 8, 10, F)

        # 3. Backbone 처리를 위한 평탄화 (Flattening)
        # reshape 전 contiguous()를 호출하여 메모리 배치를 정렬해 에러를 방지합니다.
        x_flat = x_masked.contiguous().view(-1, F)  # (B * 8 * 10, F)

        # 4. Backbone 추론
        logits_flat = self.backbone(x_flat)

        # 5. 결과 복원 (B, 8, 10, C)
        # C는 클래스 개수 (15)
        num_classes = logits_flat.size(-1)
        Y = logits_flat.view(B, K_fc, K_sb, num_classes)

        # 6. 최종 앙상블 (Mean Ensemble)
        # 각 FC(8개)의 마지막 단계(SB10)인 Y[:, :, -1, :]를 사용
        final_probs = torch.mean(torch.softmax(Y[:, :, -1, :], dim=-1), dim=1)

        return final_probs, Y, M, S

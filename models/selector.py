import torch
import torch.nn as nn
import torch.nn.functional as F


class ResGMLPBlock(nn.Module):
    def __init__(self, d_model, expansion_factor=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * expansion_factor)
        self.fc2 = nn.Linear(d_model * expansion_factor // 2, d_model)
        self.glu = nn.GLU(dim=-1)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.glu(x)
        x = self.fc2(x)
        return x + residual


class NeuralFeatureRanker(nn.Module):
    def __init__(self, input_dim, K_fc=8, d_model=64):
        super().__init__()
        self.input_dim = input_dim
        self.K_fc = K_fc

        # Global Context Encoder
        self.input_proj = nn.Linear(input_dim, d_model)
        self.encoder = nn.Sequential(ResGMLPBlock(d_model), ResGMLPBlock(d_model))

        # 8 Perspective-specific Heads
        # Each head produces (B, F) scores
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, input_dim),
                )
                for _ in range(K_fc)
            ]
        )

    def forward(self, x):
        # x: (B, F)
        z = self.input_proj(x)
        z = self.encoder(z)  # (B, d_model)

        all_logits = []
        for head in self.heads:
            all_logits.append(head(z).unsqueeze(1))  # (B, 1, F)

        return torch.cat(all_logits, dim=1)  # (B, K_fc, F)

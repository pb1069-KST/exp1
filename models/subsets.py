import torch
import torch.nn as nn


class HierarchicalSubsetBuilder(nn.Module):
    def __init__(self, k_list):
        super().__init__()
        self.k_list = torch.tensor(k_list)
        self.K_sb = len(k_list)

    def forward(self, scores, tau=1.0, hard=True):
        """
        scores: (B, K_fc, F)
        M: (B, K_fc, K_sb, F)
        """
        B, K_fc, F = scores.shape
        device = scores.device

        # 1. Add Gumbel Noise for stochastic exploration during training
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-20) + 1e-20)
        perturbed_scores = (scores + gumbel_noise) / tau

        # 2. Get Ranks (Sort in descending order)
        # Use soft-sorting approximation via ST-Estimator
        _, indices = torch.sort(perturbed_scores, dim=-1, descending=True)

        # 3. Create Cumulative Top-K Masks
        masks = []
        for k in self.k_list:
            # Create a binary mask for top-k indices
            topk_indices = indices[:, :, :k]
            m = torch.zeros_like(scores).scatter_(-1, topk_indices, 1.0)
            masks.append(m.unsqueeze(2))  # (B, K_fc, 1, F)

        M_hard = torch.cat(masks, dim=2)  # (B, K_fc, K_sb, F)

        if hard:
            # Straight-Through Estimator:
            # Forward: Hard (0/1), Backward: Soft (Gradient flows through perturbed_scores)
            M_soft = (
                torch.sigmoid(perturbed_scores)
                .unsqueeze(2)
                .expand(-1, -1, self.K_sb, -1)
            )
            return M_hard + (M_soft - M_soft.detach())

        return M_hard

import torch
import torch.nn.functional as F


def delta_confidence_loss(Y, lambda_mono=0.5):
    """
    Y: (B, K_fc, K_sb, C)
    Confidence Incremental Gain을 최대화하고 단조 증가를 강제함.
    """
    probs = torch.softmax(Y, dim=-1)
    conf = torch.max(probs, dim=-1)[0]  # (B, K_fc, K_sb)

    # 1. Step-wise Delta
    delta_conf = conf[:, :, 1:] - conf[:, :, :-1]  # (B, K_fc, K_sb-1)

    # 2. Max Jump Reward (가장 큰 폭의 상승 유도)
    max_jump = torch.max(delta_conf, dim=-1)[0]
    loss_jump = -torch.mean(max_jump)

    # 3. Monotonicity Penalty (신뢰도 하락 방지)
    penalty_mono = torch.mean(F.relu(-delta_conf))

    return loss_jump + lambda_mono * penalty_mono

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_confidence_curves(sample_idx, Y, save_path=None):
    """
    특정 샘플에 대한 8개 시나리오(FC)의 신뢰도 곡선을 그림.
    Y: (B, K_fc, K_sb, C)
    """
    probs = torch.softmax(Y[sample_idx], dim=-1)
    conf_curves = torch.max(probs, dim=-2)[0].detach().cpu().numpy()  # (K_fc, K_sb)

    plt.figure(figsize=(10, 6))
    for i in range(conf_curves.shape[0]):
        plt.plot(conf_curves[i], label=f"Scenario {i+1}", marker="o")

    plt.title(f"Confidence Curves for Sample {sample_idx}")
    plt.xlabel("SB Step (Cumulative Features)")
    plt.ylabel("Confidence")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

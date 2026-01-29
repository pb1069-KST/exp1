# tests/test_shapes.py
import torch
from models.bsfs_net import BSFSNet


def test_bsfs_shape_contract():
    B = 4
    F = 100
    C = 2
    K_fc = 8
    k_list = [5, 10, 20, 30, 40, 55, 70, 85, 100]

    model = BSFSNet(
        input_dim=F,
        num_classes=C,
        k_list=k_list,
        K_fc=K_fc,
    )

    x = torch.randn(B, F)
    final_probs, Y, M, S = model(x)

    # ---- Shape Contract ----
    assert S.shape == (B, K_fc, F), f"S shape mismatch: {S.shape}"
    assert M.shape == (B, K_fc, len(k_list), F), f"M shape mismatch: {M.shape}"
    assert Y.shape == (B, K_fc, len(k_list), C), f"Y shape mismatch: {Y.shape}"
    assert final_probs.shape == (
        B,
        C,
    ), f"final_probs shape mismatch: {final_probs.shape}"

    # ---- Probability Sanity ----
    prob_sum = final_probs.sum(dim=-1)
    assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5)

    # ---- SB Cumulative Inclusion Check ----
    # SB_t ⊆ SB_{t+1}
    for t in range(len(k_list) - 1):
        diff = M[:, :, t, :] - M[:, :, t + 1, :]
        # 이전에 1이었던 feature가 다음 단계에서 0이 되면 안 됨
        assert torch.all(diff <= 0), f"Subset violation at SB step {t}"

    print("✅ BSFS shape contract test passed.")

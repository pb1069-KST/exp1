def check_shapes(S, M, Y, final_probs, B, K_fc, K_sb, F, C):
    assert S.shape == (B, K_fc, F), f"S shape mismatch: {S.shape}"
    assert M.shape == (B, K_fc, K_sb, F), f"M shape mismatch: {M.shape}"
    assert Y.shape == (B, K_fc, K_sb, C), f"Y shape mismatch: {Y.shape}"
    assert final_probs.shape == (
        B,
        C,
    ), f"Final probs shape mismatch: {final_probs.shape}"
    print("✅ All Tensor Shapes are valid!")

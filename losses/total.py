import torch
import torch.nn.functional as F


def get_bsfs_losses(Y, masks, y_true, lambda_delta=1.0, lambda_div=0.1):
    """
    Y: (B, K_fc, K_sb, C)
    masks: (B, K_fc, K_sb, F)
    """
    B, K_fc, K_sb, C = Y.shape

    # --- 1. Classification Loss (Mean Ensemble) ---
    # Use SB_last for final decision
    logits_sb_last = Y[:, :, -1, :]  # (B, K_fc, C)
    probs_fc = torch.softmax(logits_sb_last, dim=-1)
    avg_probs = torch.mean(probs_fc, dim=1)  # (B, C)
    L_cls = F.cross_entropy(torch.log(avg_probs + 1e-10), y_true)

    # --- 2. Delta (CIG) Loss ---
    conf = torch.max(torch.softmax(Y, dim=-1), dim=-1)[0]  # (B, K_fc, K_sb)
    delta_conf = conf[:, :, 1:] - conf[:, :, :-1]  # (B, K_fc, K_sb-1)

    # Maximize the maximum jump in each scenario
    L_delta_jump = -torch.mean(torch.max(delta_conf, dim=-1)[0])
    # Monotonicity Penalty: discourage confidence decrease
    L_mono = torch.mean(F.relu(-delta_conf))
    L_delta = L_delta_jump + 0.5 * L_mono

    # --- 3. Diversity Loss ---
    # Use the final masks to calculate cosine similarity between 8 FCs
    final_masks = masks[:, :, -1, :]  # (B, K_fc, F)
    final_masks = F.normalize(final_masks, p=2, dim=-1)
    # Batch matrix multiplication: (B, K_fc, F) @ (B, F, K_fc) -> (B, K_fc, K_fc)
    sim_matrix = torch.bmm(final_masks, final_masks.transpose(1, 2))

    # Remove diagonal (self-similarity)
    eye = torch.eye(K_fc, device=Y.device).unsqueeze(0)
    L_div = torch.sum(sim_matrix * (1 - eye)) / (B * K_fc * (K_fc - 1))

    total_loss = L_cls + lambda_delta * L_delta + lambda_div * L_div

    return total_loss, {"cls": L_cls, "delta": L_delta, "div": L_div}

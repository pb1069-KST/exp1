import torch


def compute_confidence(Y):
    """
    Y: (B, K_fc, K_sb, C)
    return:
        conf: (B, K_fc, K_sb)
        pred: (B, K_fc, K_sb)
    """
    probs = torch.softmax(Y, dim=-1)
    conf, pred = torch.max(probs, dim=-1)
    return conf, pred


def extract_inflection_points(Y, M):
    """
    Y: (B, K_fc, K_sb, C)
    M: (B, K_fc, K_sb, F)

    return:
        results: list of dict (length B)
    """
    B, K_fc, K_sb, F = M.shape
    conf, _ = compute_confidence(Y)

    results = []

    for b in range(B):
        sample_result = {
            "fc_results": [],
            "global_feature_importance": None,
        }

        # --- FC별 분석 ---
        fc_feature_scores = torch.zeros(F, device=Y.device)

        for k in range(K_fc):
            delta_conf = conf[b, k, 1:] - conf[b, k, :-1]  # (K_sb-1,)
            t_star = torch.argmax(delta_conf).item() + 1  # SB index

            added_mask = M[b, k, t_star] - M[b, k, t_star - 1]
            added_features = torch.nonzero(added_mask > 0).squeeze(-1)

            gain = delta_conf[t_star - 1].item()

            sample_result["fc_results"].append(
                {
                    "fc_index": k,
                    "inflection_step": t_star,
                    "confidence_gain": gain,
                    "added_features": added_features.tolist(),
                }
            )

            # --- Global aggregation (Δconf weighted) ---
            fc_feature_scores[added_features] += gain

        # Normalize global importance
        if torch.sum(fc_feature_scores) > 0:
            fc_feature_scores = fc_feature_scores / torch.sum(fc_feature_scores)

        sample_result["global_feature_importance"] = fc_feature_scores.detach().cpu()

        results.append(sample_result)

    return results


def get_local_explanation(Y, M, feature_names):
    probs = torch.softmax(Y, dim=-1)
    conf_curves = torch.max(probs, dim=-1)[0] # (B, K_fc, K_sb) 모든 신뢰도 추출
    
    delta_conf = conf_curves[:, :, 1:] - conf_curves[:, :, :-1]
    max_delta_idx = torch.argmax(delta_conf, dim=-1)
    
    explanations = []
    for b in range(Y.shape[0]):
        sample_res = []
        for k in range(Y.shape[1]):
            t_star = max_delta_idx[b, k].item()
            # ... 기존 로직 ...
            sample_res.append({
                "scenario": k,
                "inflection_step": t_star + 1,
                "confidence_jump": delta_conf[b, k, t_star].item(),
                "key_features": added_features,
                "full_curve": conf_curves[b, k].tolist() # 전체 곡선 데이터 추가
            })
        explanations.append(sample_res)
    return explanations, conf_curves.detach().cpu().numpy()


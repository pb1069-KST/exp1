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
    """
    Y: (B, K_fc, K_sb, C) - Logits
    M: (B, K_fc, K_sb, F) - Binary Masks
    feature_names: 리스트 [f1, f2, ..., f100]
    """
    B, K_fc, K_sb, C = Y.shape

    # 1. 시나리오별 신뢰도 곡선 계산
    probs = torch.softmax(Y, dim=-1)
    conf_curves = torch.max(probs, dim=-1)[0]  # (B, K_fc, K_sb)

    # 2. 신뢰도 증분(Delta) 계산
    delta_conf = conf_curves[:, :, 1:] - conf_curves[:, :, :-1]  # (B, K_fc, K_sb-1)

    # 3. 최대 증분 지점(Inflection Point) 찾기
    max_delta_idx = torch.argmax(delta_conf, dim=-1)  # (B, K_fc)

    explanations = []
    for b in range(B):
        sample_res = []
        for k in range(K_fc):
            t_star = max_delta_idx[b, k].item()
            # t_star 단계에서 추가된 피처들 (SB_t+1 \ SB_t)
            mask_t = M[b, k, t_star, :]
            mask_next = M[b, k, t_star + 1, :]
            added_indices = torch.where((mask_next - mask_t) > 0.5)[0]

            added_features = [feature_names[i] for i in added_indices.tolist()]
            sample_res.append(
                {
                    "scenario": k,
                    "inflection_step": t_star + 1,
                    "confidence_jump": delta_conf[b, k, t_star].item(),
                    "key_features": added_features,
                    "full_curve": conf_curves[b, k].tolist(),  # 전체 곡선 데이터 추가
                }
            )
        explanations.append(sample_res)
    return explanations, conf_curves.detach().cpu().numpy()

import matplotlib

matplotlib.use("Agg")  # 리눅스 서버 환경 설정
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import json
import os


def print_detailed_table(sample_idx, explanations, conf_curves):
    """8개 시나리오의 단계별 신뢰도를 표 형태로 출력"""
    print(f"\n[상세 분석 리포트 - 샘플 #{sample_idx}]")
    K_fc, K_sb = conf_curves[sample_idx].shape
    header = "단계  | " + " | ".join([f"시나리오{i+1}" for i in range(K_fc)])
    print(header)
    print("-" * len(header))
    for t in range(K_sb):
        row = f"SB{t+1:02d} | " + " | ".join(
            [f"{conf_curves[sample_idx, k, t]:.4f}" for k in range(K_fc)]
        )
        print(row)


def plot_confidence_heatmap(
    sample_idx, conf_curves, save_path="analysis_results/confidence_heatmap.png"
):
    """특정 샘플의 신뢰도 축적 과정을 히트맵으로 저장"""
    plt.figure(figsize=(12, 6))
    data = conf_curves[sample_idx]
    sns.heatmap(
        data,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=[f"SB{i+1}" for i in range(data.shape[1])],
        yticklabels=[f"Scenario {i+1}" for i in range(data.shape[0])],
    )
    plt.title(f"Confidence Accumulation Heatmap (Sample {sample_idx})")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Heatmap saved to {save_path}")


def analyze_global_importance(
    all_explanations,
    feature_names,
    top_n=20,
    save_path="analysis_results/global_feature_importance.png",
):
    """전역 피처 중요도 산출 및 시각화 (Seaborn 경고 해결 버전)"""
    importance_scores = {name: 0.0 for name in feature_names}
    occurrence_counts = {name: 0 for name in feature_names}

    for sample_exps in all_explanations:
        for scenario in sample_exps:
            jump = scenario.get("confidence_jump", 0.0)
            for feat in scenario.get("key_features", []):
                if feat in importance_scores:
                    importance_scores[feat] += jump
                    occurrence_counts[feat] += 1

    df_importance = pd.DataFrame(
        {
            "Feature": list(importance_scores.keys()),
            "Total_Jump_Score": list(importance_scores.values()),
            "Occurrence_Count": list(occurrence_counts.values()),
        }
    )
    df_importance = df_importance.sort_values(
        by="Total_Jump_Score", ascending=False
    ).head(top_n)

    plt.figure(figsize=(12, 8))
    # hue와 legend 인자를 추가하여 Future Warning 해결
    sns.barplot(
        x="Total_Jump_Score",
        y="Feature",
        data=df_importance,
        hue="Feature",
        palette="viridis",
        legend=False,
    )
    plt.title(f"Global Feature Importance (Top {top_n})")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Global Importance Plot saved to {save_path}")
    return df_importance


def save_global_results(all_explanations, save_dir="analysis_results"):
    """전체 인스턴스 분석 결과를 JSON 및 CSV로 저장"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. JSON 저장
    json_path = os.path.join(save_dir, "full_xai_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_explanations, f, indent=4, ensure_ascii=False)

    # 2. CSV 저장
    csv_path = os.path.join(save_dir, "xai_summary_statistics.csv")
    summary_data = []
    for s_idx, sample_exps in enumerate(all_explanations):
        for scenario in sample_exps:
            summary_data.append(
                {
                    "sample_index": s_idx,
                    "scenario_id": scenario["scenario"] + 1,
                    "inflection_step": scenario["inflection_step"],
                    "confidence_jump": scenario["confidence_jump"],
                    "feature_count": len(scenario["key_features"]),
                    "features": "|".join(scenario["key_features"]),
                }
            )
    pd.DataFrame(summary_data).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"📄 데이터 저장 완료: {json_path}, {csv_path}")

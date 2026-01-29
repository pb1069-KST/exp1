import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
from models.bsfs_net import BSFSNet
from core.trainer import BSFSTrainer
from data.dataset import load_and_clean_data, get_dataloader
from data.preprocess import preprocess_pipeline
from data.features import FEATURE_NAMES, NUM_CLASSES
from xai.attribution import get_local_explanation
from xai.reporter import (
    print_detailed_table,
    plot_confidence_heatmap,
    analyze_global_importance,
    save_global_results,
)


def main():
    config = {
        "K_fc": 8,
        "K_sb": 10,
        "k_list": [5, 10, 20, 30, 40, 50, 60, 78],
        "lambda_delta": 1.5,
        "lambda_div": 0.5,
        "tau_init": 2.0,
        "tau_min": 0.5,
        "tau_decay": 0.95,
        "learning_rate": 0.001,
        "batch_size": 128,
        "epochs": 30,
        "data_path": "data/input_sampled.csv",
    }

    yaml_path = "configs/default.yaml"
    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            yaml_config = yaml.safe_load(f)
            config.update(yaml_config)
            print(f"✅ Config loaded from {yaml_path}")

    # 중요: YAML의 k_list 길이에 맞춰 K_sb를 동적으로 업데이트
    config["K_sb"] = len(config["k_list"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, label_encoder = load_and_clean_data(config["data_path"])
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, _ = preprocess_pipeline(X_train_raw, X_val_raw)

    train_loader = get_dataloader(X_train, y_train, batch_size=config["batch_size"])
    val_loader = get_dataloader(
        X_val, y_val, batch_size=config["batch_size"], shuffle=False
    )

    model = BSFSNet(
        len(FEATURE_NAMES), NUM_CLASSES, config["k_list"], config["K_fc"]
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    trainer = BSFSTrainer(model, optimizer, device, config)

    print("🚀 학습 시작...")
    for epoch in range(config["epochs"]):
        trainer.train_epoch(train_loader, epoch)

    # 글로벌 XAI 분석 및 데이터 추출
    print("\n📊 전체 인스턴스 분석 및 결과 추출 중...")
    model.eval()
    all_explanations = []
    with torch.no_grad():
        for x_batch, _ in val_loader:
            x_batch = x_batch.to(device)
            _, Y, M, _ = model(x_batch, tau=0.1)
            # get_local_explanation이 (explanations, conf_curves) 튜플을 리턴하는지 확인
            batch_exps, _ = get_local_explanation(Y, M, FEATURE_NAMES)
            all_explanations.extend(batch_exps)

    # 1. 시각화 및 CSV/JSON 저장
    analyze_global_importance(all_explanations, FEATURE_NAMES)
    save_global_results(all_explanations, save_dir="analysis_results")

    # 2. 개별 샘플 상세 분석 (샘플 0번)
    print("\n🔍 샘플 #0 상세 분석 중...")

    # Numpy Array를 Tensor로 변환하여 AttributeError 해결
    if isinstance(X_val, np.ndarray):
        sample_x = torch.from_numpy(X_val[:1]).float().to(device)
    else:
        sample_x = X_val[:1].to(device)

    with torch.no_grad():
        _, Y_s, M_s, _ = model(sample_x, tau=0.1)
        expl_s, conf_all_s = get_local_explanation(Y_s, M_s, FEATURE_NAMES)

    # 상세 텍스트 표 출력
    print_detailed_table(0, expl_s, conf_all_s)
    # 히트맵 이미지 저장
    plot_confidence_heatmap(
        0, conf_all_s, save_path="analysis_results/sample_0_heatmap.png"
    )

    print("\n✅ 모든 분석이 완료되었습니다. 'analysis_results/' 폴더를 확인하세요.")


if __name__ == "__main__":
    main()

import os
import torch
import torch.optim as optim
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

# 내부 모듈 임포트
from models.bsfs_net import BSFSNet
from core.trainer import BSFSTrainer
from core.evaluator import BSFSPerformanceEvaluator
from data.dataset import load_and_clean_data, get_dataloader
from data.preprocess import preprocess_pipeline
from data.features import FEATURE_NAMES, NUM_CLASSES
from xai.attribution import get_local_explanation
from xai.reporter import plot_confidence_curves


def main():
    # 1. 설정 로드 (파일이 없을 경우를 대비해 기본값 정의)
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

    # YAML 파일이 있다면 덮어쓰기
    if os.path.exists("configs/default.yaml"):
        with open("configs/default.yaml", "r") as f:
            config.update(yaml.safe_load(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # 2. 데이터 로드 및 전처리
    print("📊 Loading and cleaning data...")
    X, y, label_encoder = load_and_clean_data(config["data_path"])

    # Train/Val Split (8:2)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling
    print("⚖️ Scaling features...")
    X_train, X_val, preprocessor = preprocess_pipeline(X_train_raw, X_val_raw)

    # DataLoader
    train_loader = get_dataloader(X_train, y_train, batch_size=config["batch_size"])
    val_loader = get_dataloader(
        X_val, y_val, batch_size=config["batch_size"], shuffle=False
    )

    # 3. 모델 및 최적화 도구 초기화
    print("🏗️ Initializing BSFS-Net...")
    model = BSFSNet(
        input_dim=len(FEATURE_NAMES),
        num_classes=NUM_CLASSES,
        k_list=config["k_list"],
        K_fc=config["K_fc"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    trainer = BSFSTrainer(model, optimizer, device, config)
    evaluator = BSFSPerformanceEvaluator(model, device)

    # 4. 학습 루프
    print(f"🚀 Starting training for {config['epochs']} epochs...")
    for epoch in range(config["epochs"]):
        loss, current_tau = trainer.train_epoch(train_loader, epoch)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            metrics = evaluator.evaluate(val_loader)
            f1 = metrics["f1_macro"]
            print(
                f"Epoch [{epoch+1}/{config['epochs']}] | Loss: {loss:.4f} | Val F1: {f1:.4f} | Tau: {current_tau:.2f}"
            )

    # 5. 최종 평가 및 XAI 리포트 샘플 생성
    print("\n🔍 Generating XAI Report for a sample...")
    model.eval()
    sample_x, sample_y = next(iter(val_loader))
    sample_x, sample_y = sample_x[:1].to(device), sample_y[:1].to(
        device
    )  # 첫 번째 샘플만

    with torch.no_grad():
        final_probs, Y, M, S = model(sample_x, tau=0.1)

    # Local Attribution 추출
    explanations = get_local_explanation(Y, M, FEATURE_NAMES)

    # 결과 출력
    target_class = label_encoder.inverse_transform([sample_y.item()])[0]
    pred_class = label_encoder.inverse_transform([torch.argmax(final_probs).item()])[0]

    print(f"--- XAI Analysis ---")
    print(f"Actual Label: {target_class}")
    print(f"Predicted Label: {pred_class} (Conf: {torch.max(final_probs).item():.4f})")

    # 8개 시나리오 중 가장 영향력 큰(Jump가 큰) 시나리오 리포트
    best_scenario = max(explanations[0], key=lambda x: x["confidence_jump"])
    print(f"Top Scenario: #{best_scenario['scenario']+1}")
    print(f"Inflection Step: {best_scenario['inflection_step']}")
    print(f"Key Features added at jump: {best_scenario['key_features']}")

    # 신뢰도 곡선 시각화 (선택 사항)
    # plot_confidence_curves(0, Y, save_path="confidence_curve_sample.png")

    print("\n✅ All processes completed successfully.")


if __name__ == "__main__":
    main()

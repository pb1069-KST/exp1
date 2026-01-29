import torch
from tqdm import tqdm
from losses.total import get_bsfs_losses


class BSFSTrainer:
    def __init__(self, model, optimizer, device, config):
        """
        BSFS-Net 전용 트레이너

        Args:
            model: BSFSNet 모델
            optimizer: 최적화 도구 (Adam 등)
            device: 'cuda' 또는 'cpu'
            config: 하이퍼파라미터 설정 (tau, lambda 등 포함)
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.config = config

        # 최신 PyTorch 규격에 맞는 GradScaler 및 device_type 설정
        self.device_type = "cuda" if "cuda" in str(device) else "cpu"
        self.scaler = torch.amp.GradScaler(self.device_type)

    def train_epoch(self, dataloader, epoch):
        """
        한 에포크 동안 학습을 진행하며 주요 KPI를 모니터링합니다.
        """
        self.model.train()
        total_loss = 0

        # Gumbel-Softmax Temperature (Tau) 스케줄링
        # 에포크가 진행될수록 낮아져 마스크를 0 또는 1로 수렴시킵니다.
        tau = max(
            self.config["tau_min"],
            self.config["tau_init"] * (self.config["tau_decay"] ** epoch),
        )

        # TQDM Progress Bar 설정 (KPI 모니터링용)
        pbar = tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{self.config['epochs']}", unit="batch"
        )

        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            # Mixed Precision 학습 (속도 및 메모리 효율화)
            with torch.amp.autocast(device_type=self.device_type):
                # 1. Forward
                final_probs, Y, M, S = self.model(x, tau=tau)

                # 2. BSFS 전용 멀티태스크 손실 함수 계산
                loss, loss_dict = get_bsfs_losses(
                    Y,
                    M,
                    y,
                    lambda_delta=self.config["lambda_delta"],
                    lambda_div=self.config["lambda_div"],
                )

            # 3. Backward & Step
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            # 실시간 KPI 업데이트
            # Loss: 전체 손실 / delta: 신뢰도 점프 성능 / div: 시나리오별 독립성
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Δ-Loss": f"{loss_dict['delta']:.4f}",
                    "Div": f"{loss_dict['div']:.4f}",
                    "Tau": f"{tau:.2f}",
                }
            )

        avg_loss = total_loss / len(dataloader)
        return avg_loss, tau

    def validate(self, dataloader):
        """
        검증 데이터셋에 대한 손실 값을 측정합니다. (XAI 분석 제외 단순 평가용)
        """
        self.model.eval()
        total_loss = 0
        tau = self.config["tau_min"]  # 검증 시에는 낮은 온도로 고정

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                with torch.amp.autocast(device_type=self.device_type):
                    final_probs, Y, M, S = self.model(x, tau=tau)
                    loss, _ = get_bsfs_losses(
                        Y, M, y, self.config["lambda_delta"], self.config["lambda_div"]
                    )
                total_loss += loss.item()

        return total_loss / len(dataloader)

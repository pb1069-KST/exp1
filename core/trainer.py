import torch
from torch.cuda.amp import autocast, GradScaler
from losses.total import get_bsfs_losses


class BSFSTrainer:
    def __init__(self, model, optimizer, device, config):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.config = config

        # FutureWarning 해결: 장치 유형('cuda' 또는 'cpu')을 명시적으로 전달
        device_type = "cuda" if "cuda" in str(device) else "cpu"
        self.scaler = torch.amp.GradScaler(device_type)

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        tau = max(
            self.config["tau_min"],
            self.config["tau_init"] * (self.config["tau_decay"] ** epoch),
        )

        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            # 최신 버전의 autocast 문법 사용
            device_type = "cuda" if "cuda" in str(self.device) else "cpu"
            with torch.amp.autocast(device_type=device_type):
                final_probs, Y, M, S = self.model(x, tau=tau)
                loss, loss_dict = get_bsfs_losses(
                    Y,
                    M,
                    y,
                    lambda_delta=self.config["lambda_delta"],
                    lambda_div=self.config["lambda_div"],
                )

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(dataloader), tau

    def evaluate(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                probs, _, _, _ = self.model(x, tau=0.1)  # 추론 시엔 낮은 tau
                preds = torch.argmax(probs, dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        return correct / total

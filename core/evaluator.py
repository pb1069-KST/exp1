import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, classification_report


class BSFSPerformanceEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        all_probs = []
        all_targets = []

        for x, y in dataloader:
            x = x.to(self.device)
            # 추론 시에는 확정적인 결정을 위해 낮은 tau 사용
            probs, _, _, _ = self.model(x, tau=0.1)
            all_probs.append(probs.cpu())
            all_targets.append(y)

        all_probs = torch.cat(all_probs, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        all_preds = np.argmax(all_probs, axis=1)

        metrics = {
            "f1_macro": f1_score(
                all_targets, all_preds, average="macro", zero_division=0
            ),
            "report": classification_report(all_targets, all_preds, output_dict=True),
        }

        # Binary Classification일 경우 AUROC 추가
        if all_probs.shape[1] == 2:
            metrics["auroc"] = roc_auc_score(all_targets, all_probs[:, 1])

        return metrics

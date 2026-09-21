import torch
import numpy as np
from typing import Dict
from .metrics import MetricTracker
from src.utils.logger import get_logger

logger = get_logger("validator")


class Validator:
    """
    Runs evaluation on a validation or test DataLoader and returns metrics.
    Works with any model that returns logits given a batch.
    """

    def __init__(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                 criterion: torch.nn.Module, device: torch.device):
        self.model      = model
        self.dataloader = dataloader
        self.criterion  = criterion
        self.device     = device

    @torch.no_grad()
    def evaluate(self) -> Dict:
        """
        Run full evaluation pass.

        Returns:
            Dict of metrics including loss, accuracy, f1, confusion matrix.
        """
        self.model.eval()
        tracker = MetricTracker()

        for batch in self.dataloader:
            inputs, labels = self._unpack_batch(batch)
            logits = self.model(*inputs)
            loss   = self.criterion(logits, labels)

            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = np.argmax(probs, axis=1)
            y_true = labels.cpu().numpy()

            tracker.update(y_true, preds, probs, loss.item())

        metrics = tracker.compute()
        self._log_metrics(metrics)
        return metrics

    def _unpack_batch(self, batch):
        """
        Unpack a batch into (inputs_tuple, labels).
        Supports single-input and multi-modal (body, face, speech) batches.
        """
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, labels = batch
            if isinstance(inputs, (list, tuple)):
                inputs = tuple(x.to(self.device) for x in inputs)
            else:
                inputs = (inputs.to(self.device),)
            labels = labels.to(self.device)
        else:
            raise ValueError(f"Unexpected batch format: {type(batch)}")
        return inputs, labels

    def _log_metrics(self, metrics: Dict):
        logger.info(
            f"[Val] loss: {metrics.get('loss', 'N/A'):.4f}  "
            f"acc: {metrics['accuracy']:.4f}  "
            f"f1_w: {metrics['f1_weighted']:.4f}  "
            f"f1_mac: {metrics['f1_macro']:.4f}"
        )

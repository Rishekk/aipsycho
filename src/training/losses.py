import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger("losses")


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Down-weights easy examples and focuses on hard ones.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy with label smoothing to prevent overconfidence.
    """

    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        smooth_val = self.smoothing / (self.num_classes - 1)

        # Build smooth target distribution
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, smooth_val)
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(smooth_targets * log_probs).sum(dim=-1)
        return loss.mean()


class MultimodalLoss(nn.Module):
    """
    Combined loss for multimodal training.
    Sums individual agent losses + fusion loss with configurable weights.
    Encourages each agent to be independently discriminative.
    """

    def __init__(self, num_classes: int = 5, use_focal: bool = True,
                 agent_weight: float = 0.3, fusion_weight: float = 1.0):
        super().__init__()
        self.agent_weight  = agent_weight
        self.fusion_weight = fusion_weight

        base_loss = FocalLoss() if use_focal else nn.CrossEntropyLoss()
        self.body_loss    = base_loss
        self.face_loss    = base_loss
        self.speech_loss  = base_loss
        self.fusion_loss  = LabelSmoothingLoss(num_classes)

        logger.info(
            f"MultimodalLoss — agent_weight: {agent_weight}, "
            f"fusion_weight: {fusion_weight}, focal: {use_focal}"
        )

    def forward(self, body_logits: torch.Tensor, face_logits: torch.Tensor,
                speech_logits: torch.Tensor, fusion_logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        l_body    = self.body_loss(body_logits,     targets)
        l_face    = self.face_loss(face_logits,     targets)
        l_speech  = self.speech_loss(speech_logits, targets)
        l_fusion  = self.fusion_loss(fusion_logits, targets)

        return (
            self.agent_weight  * (l_body + l_face + l_speech) +
            self.fusion_weight * l_fusion
        )

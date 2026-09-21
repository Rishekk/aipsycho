import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict
from config.constants import PSYCHOLOGICAL_STATES, IDX_TO_STATE, NUM_CLASSES
from src.utils.logger import get_logger

logger = get_logger("early_fusion")


class EarlyFusionModel(nn.Module):
    """
    Early fusion: concatenates feature embeddings from all three agents
    before passing through a shared classification head.

    Input: body_feat (256,) + face_feat (2048,) + speech_feat (256,)
    """

    def __init__(self, body_dim: int = 256, face_dim: int = 2048,
                 speech_dim: int = 256, hidden_dim: int = 512,
                 num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()

        fused_dim = body_dim + face_dim + speech_dim

        # Project each modality to a common space before concat
        self.body_proj   = nn.Linear(body_dim,   hidden_dim // 3)
        self.face_proj   = nn.Linear(face_dim,   hidden_dim // 3)
        self.speech_proj = nn.Linear(speech_dim, hidden_dim // 3)

        proj_dim = (hidden_dim // 3) * 3

        self.classifier = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        logger.info(
            f"EarlyFusionModel initialized — "
            f"body: {body_dim}, face: {face_dim}, speech: {speech_dim}"
        )

    def forward(self, body_feat: torch.Tensor, face_feat: torch.Tensor,
                speech_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            body_feat:   (batch, body_dim)
            face_feat:   (batch, face_dim)
            speech_feat: (batch, speech_dim)

        Returns:
            Logits of shape (batch, num_classes).
        """
        b = self.body_proj(body_feat)
        f = self.face_proj(face_feat)
        s = self.speech_proj(speech_feat)
        fused = torch.cat([b, f, s], dim=-1)
        return self.classifier(fused)


class EarlyFusionClassifier:
    """High-level wrapper for EarlyFusionModel inference."""

    def __init__(self, model_path: str = None, device: str = None, **model_kwargs):
        self.device = torch.device(device or (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        ))
        self.model = EarlyFusionModel(**model_kwargs).to(self.device)
        if model_path:
            ckpt = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Loaded early fusion weights from {model_path}")
        self.model.eval()

    def predict(self, body_emb: np.ndarray, face_emb: np.ndarray,
                speech_emb: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        bt = torch.tensor(body_emb,   dtype=torch.float32).unsqueeze(0).to(self.device)
        ft = torch.tensor(face_emb,   dtype=torch.float32).unsqueeze(0).to(self.device)
        st = torch.tensor(speech_emb, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(bt, ft, st)
            probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        idx = int(np.argmax(probs))
        return (
            IDX_TO_STATE[idx],
            float(probs[idx]),
            {s: float(probs[i]) for i, s in enumerate(PSYCHOLOGICAL_STATES)},
        )

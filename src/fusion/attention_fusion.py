import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict
from config.constants import PSYCHOLOGICAL_STATES, IDX_TO_STATE, NUM_CLASSES
from src.utils.logger import get_logger

logger = get_logger("attention_fusion")


class AttentionFusionModel(nn.Module):
    """
    Attention-based multimodal fusion.

    Projects each modality into a shared latent space, stacks them as a
    sequence, and applies multi-head self-attention so the model learns
    which modality to trust for each sample.

    Architecture:
        body / face / speech → linear projections → stack (3, hidden_dim)
        → multi-head self-attention → mean pool → classifier
    """

    def __init__(self, body_dim: int = 256, face_dim: int = 2048,
                 speech_dim: int = 256, hidden_dim: int = 256,
                 num_heads: int = 4, num_classes: int = NUM_CLASSES,
                 dropout: float = 0.3):
        super().__init__()

        # Project each modality to hidden_dim
        self.body_proj   = nn.Sequential(nn.Linear(body_dim,   hidden_dim), nn.ReLU())
        self.face_proj   = nn.Sequential(nn.Linear(face_dim,   hidden_dim), nn.ReLU())
        self.speech_proj = nn.Sequential(nn.Linear(speech_dim, hidden_dim), nn.ReLU())

        # Multi-head self-attention over the 3 modality tokens
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        logger.info(
            f"AttentionFusionModel initialized — hidden: {hidden_dim}, "
            f"heads: {num_heads}, classes: {num_classes}"
        )

    def forward(self, body_feat: torch.Tensor, face_feat: torch.Tensor,
                speech_feat: torch.Tensor,
                return_weights: bool = False):
        """
        Args:
            body_feat:      (batch, body_dim)
            face_feat:      (batch, face_dim)
            speech_feat:    (batch, speech_dim)
            return_weights: If True, also return attention weights.

        Returns:
            logits (batch, num_classes), and optionally attn_weights (batch, 3, 3).
        """
        # Project to common space → (batch, hidden_dim)
        b = self.body_proj(body_feat)
        f = self.face_proj(face_feat)
        s = self.speech_proj(speech_feat)

        # Stack as sequence of 3 tokens → (batch, 3, hidden_dim)
        tokens = torch.stack([b, f, s], dim=1)

        # Self-attention
        attn_out, attn_weights = self.attention(tokens, tokens, tokens)

        # Residual + norm
        attn_out = self.norm(attn_out + tokens)

        # Mean pool over modality tokens
        pooled = attn_out.mean(dim=1)   # (batch, hidden_dim)
        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)

        if return_weights:
            return logits, attn_weights
        return logits

    def get_features(self, body_feat: torch.Tensor, face_feat: torch.Tensor,
                     speech_feat: torch.Tensor) -> torch.Tensor:
        """Return fused embedding before classification (for downstream use)."""
        b = self.body_proj(body_feat)
        f = self.face_proj(face_feat)
        s = self.speech_proj(speech_feat)
        tokens = torch.stack([b, f, s], dim=1)
        attn_out, _ = self.attention(tokens, tokens, tokens)
        return self.norm(attn_out + tokens).mean(dim=1)


class AttentionFusionClassifier:
    """High-level wrapper for AttentionFusionModel inference."""

    def __init__(self, model_path: str = None, device: str = None, **model_kwargs):
        self.device = torch.device(device or (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        ))
        self.model = AttentionFusionModel(**model_kwargs).to(self.device)
        if model_path:
            ckpt = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Loaded attention fusion weights from {model_path}")
        self.model.eval()

    def predict(self, body_emb: np.ndarray, face_emb: np.ndarray,
                speech_emb: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        bt = torch.tensor(body_emb,   dtype=torch.float32).unsqueeze(0).to(self.device)
        ft = torch.tensor(face_emb,   dtype=torch.float32).unsqueeze(0).to(self.device)
        st = torch.tensor(speech_emb, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, attn_w = self.model(bt, ft, st, return_weights=True)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        idx = int(np.argmax(probs))
        return (
            IDX_TO_STATE[idx],
            float(probs[idx]),
            {s: float(probs[i]) for i, s in enumerate(PSYCHOLOGICAL_STATES)},
        )

    def get_attention_weights(self, body_emb: np.ndarray, face_emb: np.ndarray,
                              speech_emb: np.ndarray) -> np.ndarray:
        """
        Return attention weights for interpretability.

        Returns:
            Array of shape (3, 3) — attention between [body, face, speech] tokens.
        """
        bt = torch.tensor(body_emb,   dtype=torch.float32).unsqueeze(0).to(self.device)
        ft = torch.tensor(face_emb,   dtype=torch.float32).unsqueeze(0).to(self.device)
        st = torch.tensor(speech_emb, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, attn_w = self.model(bt, ft, st, return_weights=True)

        return attn_w.squeeze(0).cpu().numpy()

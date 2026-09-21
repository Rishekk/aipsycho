import torch
import torch.nn as nn
from src.utils.logger import get_logger

logger = get_logger("temporal_model")


class BodyLSTM(nn.Module):
    """
    LSTM-based temporal model for body language sequence classification.
    Processes sequences of pose feature vectors to capture motion dynamics.
    """

    def __init__(self, input_size: int = 99, hidden_size: int = 256,
                 num_layers: int = 2, num_classes: int = 5,
                 dropout: float = 0.3, bidirectional: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.norm = nn.LayerNorm(hidden_size * self.num_directions)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        logger.info(
            f"BodyLSTM initialized — input: {input_size}, hidden: {hidden_size}, "
            f"layers: {num_layers}, classes: {num_classes}, bidirectional: {bidirectional}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, input_size).

        Returns:
            Logits of shape (batch, num_classes).
        """
        out, _ = self.lstm(x)           # (batch, seq_len, hidden * directions)
        out = out[:, -1, :]             # take last timestep
        out = self.norm(out)
        out = self.dropout(out)
        return self.classifier(out)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return LSTM output embedding before classification (for fusion)."""
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.norm(out)
        return out


class BodyGRU(nn.Module):
    """
    GRU-based alternative to BodyLSTM — lighter and often faster to train.
    Same interface as BodyLSTM.
    """

    def __init__(self, input_size: int = 99, hidden_size: int = 256,
                 num_layers: int = 2, num_classes: int = 5,
                 dropout: float = 0.3, bidirectional: bool = False):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.norm = nn.LayerNorm(hidden_size * self.num_directions)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        logger.info(f"BodyGRU initialized — input: {input_size}, hidden: {hidden_size}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.norm(out)
        out = self.dropout(out)
        return self.classifier(out)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.norm(out)

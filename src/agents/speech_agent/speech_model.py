import torch
import torch.nn as nn
from src.utils.logger import get_logger

logger = get_logger("speech_model")


class SpeechCNNLSTM(nn.Module):
    """
    CNN-LSTM model for speech emotion classification.
    CNN extracts local patterns from MFCC frames, LSTM captures temporal dynamics.

    Input: (batch, channels=3, n_mfcc=40, T) — stacked MFCC + delta + delta2
    """

    def __init__(self, n_mfcc: int = 40, num_classes: int = 5,
                 lstm_hidden: int = 256, lstm_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        # CNN block — extract spectro-temporal patterns
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(0.1),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(0.1),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # compress freq, keep time
        )

        # After CNN: (batch, 128, n_mfcc//8, T//4) → reshape for LSTM
        cnn_freq = n_mfcc // 8
        self.lstm_input_size = 128 * cnn_freq

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.norm = nn.LayerNorm(lstm_hidden)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        logger.info(
            f"SpeechCNNLSTM initialized — n_mfcc: {n_mfcc}, lstm_hidden: {lstm_hidden}, "
            f"classes: {num_classes}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, 3, n_mfcc, T).

        Returns:
            Logits of shape (batch, num_classes).
        """
        batch = x.size(0)

        # CNN
        out = self.cnn(x)                       # (batch, 128, freq', time')
        _, _, freq, time = out.shape
        out = out.permute(0, 3, 1, 2)           # (batch, time', 128, freq')
        out = out.reshape(batch, time, -1)      # (batch, time', lstm_input_size)

        # LSTM
        out, _ = self.lstm(out)                 # (batch, time', lstm_hidden)
        out = out[:, -1, :]                     # last timestep

        out = self.norm(out)
        out = self.dropout(out)
        return self.classifier(out)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return LSTM embedding before classification head (for fusion)."""
        batch = x.size(0)
        out = self.cnn(x)
        _, _, freq, time = out.shape
        out = out.permute(0, 3, 1, 2).reshape(batch, time, -1)
        out, _ = self.lstm(out)
        return self.norm(out[:, -1, :])


class SpeechMLP(nn.Module):
    """
    Lightweight MLP that operates on pre-computed statistical feature vectors.
    Faster alternative when real-time CNN-LSTM is too heavy.
    """

    def __init__(self, input_size: int = 312, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        logger.info(f"SpeechMLP initialized — input: {input_size}, classes: {num_classes}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in list(self.net.children())[:-1]:
            out = layer(out)
        return out

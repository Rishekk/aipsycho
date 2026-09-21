import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from typing import Tuple, Dict
from config.constants import PSYCHOLOGICAL_STATES, IDX_TO_STATE, NUM_CLASSES
from src.utils.logger import get_logger

logger = get_logger("emotion_cnn")


class EmotionCNN(nn.Module):
    """
    ResNet50-based CNN for psychological state classification from facial images.
    Uses transfer learning with a custom classification head.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True,
                 dropout: float = 0.3, freeze_layers: int = 6):
        super().__init__()

        # Load pretrained backbone
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)

        # Freeze early layers
        layers = list(backbone.children())
        for layer in layers[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

        # Replace final FC layer
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        logger.info(f"EmotionCNN initialized — backbone: ResNet50, classes: {num_classes}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature embeddings before classification head (for fusion)."""
        return self.backbone(x)


class FacePreprocessor:
    """Preprocess face images for EmotionCNN input."""

    def __init__(self, input_size: int = 224):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def preprocess(self, face_bgr: np.ndarray) -> torch.Tensor:
        """
        Convert a BGR face crop to a normalized tensor.

        Args:
            face_bgr: Face image in BGR format.

        Returns:
            Tensor of shape (1, 3, H, W).
        """
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(face_rgb)
        return tensor.unsqueeze(0)  # add batch dim


class FaceClassifier:
    """
    High-level wrapper that runs FaceDetector → EmotionCNN and returns predictions.
    """

    def __init__(self, model_path: str = None, device: str = None):
        if device is None:
            self.device = (
                torch.device("cuda") if torch.cuda.is_available()
                else torch.device("mps") if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = torch.device(device)

        self.model = EmotionCNN().to(self.device)
        self.preprocessor = FacePreprocessor()

        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded weights from {model_path}")

        self.model.eval()

    def predict(self, face_bgr: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict psychological state from a face crop.

        Args:
            face_bgr: Cropped face image in BGR format.

        Returns:
            (predicted_state, confidence, {state: probability})
        """
        tensor = self.preprocessor.preprocess(face_bgr).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        return (
            IDX_TO_STATE[pred_idx],
            float(probs[pred_idx]),
            {state: float(probs[i]) for i, state in enumerate(PSYCHOLOGICAL_STATES)},
        )

    def get_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        """Return feature embedding for fusion."""
        tensor = self.preprocessor.preprocess(face_bgr).to(self.device)
        with torch.no_grad():
            embedding = self.model.get_features(tensor)
        return embedding.squeeze(0).cpu().numpy()

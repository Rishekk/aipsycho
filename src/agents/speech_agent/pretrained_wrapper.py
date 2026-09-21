import torch
import numpy as np
from typing import Tuple, Dict, Optional
from .audio_processor import AudioProcessor
from .feature_extractor import SpeechFeatureExtractor
from .speech_model import SpeechCNNLSTM
from config.constants import PSYCHOLOGICAL_STATES, IDX_TO_STATE, NUM_CLASSES
from src.utils.logger import get_logger

logger = get_logger("pretrained_wrapper")


class Wav2VecWrapper:
    """
    Wraps HuggingFace wav2vec 2.0 for feature extraction.
    Used as a powerful pretrained encoder instead of hand-crafted MFCCs.
    Requires: pip install transformers
    """

    def __init__(self, model_name: str = "facebook/wav2vec2-base",
                 device: str = None):
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2Model
        except ImportError:
            raise ImportError("Install transformers: pip install transformers")

        self.device = torch.device(device or (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        ))

        logger.info(f"Loading wav2vec2 model: {model_name}...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info("Wav2Vec2 loaded.")

    def extract_features(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Extract contextual embeddings from raw audio.

        Args:
            audio: 1D float32 audio array.
            sample_rate: Must be 16000 for wav2vec2.

        Returns:
            Mean-pooled embedding of shape (768,).
        """
        inputs = self.processor(
            audio, sampling_rate=sample_rate,
            return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Mean pool over time dimension
        hidden = outputs.last_hidden_state  # (1, T, 768)
        return hidden.mean(dim=1).squeeze(0).cpu().numpy()


class SpeechClassifier:
    """
    Full speech agent pipeline:
      AudioProcessor → SpeechFeatureExtractor → SpeechCNNLSTM → prediction

    Supports both MFCC-based CNN-LSTM and optional wav2vec2 embeddings.
    """

    def __init__(self, model_path: str = None, use_wav2vec: bool = False,
                 device: str = None):
        self.device = torch.device(device or (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        ))

        self.audio_processor = AudioProcessor()
        self.feature_extractor = SpeechFeatureExtractor()

        self.model = SpeechCNNLSTM(
            n_mfcc=40, num_classes=NUM_CLASSES
        ).to(self.device)

        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded speech model weights from {model_path}")

        self.model.eval()

        self.wav2vec: Optional[Wav2VecWrapper] = None
        if use_wav2vec:
            self.wav2vec = Wav2VecWrapper(device=str(self.device))

    def predict_from_file(self, path: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict psychological state from an audio file.

        Args:
            path: Path to audio file.

        Returns:
            (predicted_state, confidence, {state: probability})
        """
        audio = self.audio_processor.load(path)
        audio = self.audio_processor.pad_or_trim(audio)
        return self._predict(audio)

    def predict_from_array(self, audio: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """Predict from a raw audio array."""
        audio = self.audio_processor.normalize(audio)
        audio = self.audio_processor.pad_or_trim(audio)
        return self._predict(audio)

    def predict_from_mic(self, duration: float = 2.0) -> Tuple[str, float, Dict[str, float]]:
        """Record from microphone and predict."""
        audio = self.audio_processor.record(duration=duration)
        audio = self.audio_processor.pad_or_trim(audio)
        return self._predict(audio)

    def get_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        Return speech feature embedding for fusion agent.
        Uses wav2vec2 if available, otherwise CNN-LSTM features.
        """
        audio = self.audio_processor.normalize(audio)
        audio = self.audio_processor.pad_or_trim(audio)

        if self.wav2vec:
            return self.wav2vec.extract_features(audio)

        tensor = self._audio_to_tensor(audio)
        with torch.no_grad():
            embedding = self.model.get_features(tensor)
        return embedding.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict(self, audio: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        tensor = self._audio_to_tensor(audio)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        return (
            IDX_TO_STATE[pred_idx],
            float(probs[pred_idx]),
            {state: float(probs[i]) for i, state in enumerate(PSYCHOLOGICAL_STATES)},
        )

    def _audio_to_tensor(self, audio: np.ndarray) -> torch.Tensor:
        """Convert audio array to (1, 3, n_mfcc, T) tensor for CNN-LSTM."""
        mfcc_mat = self.feature_extractor.get_mfcc_matrix(audio)  # (120, T)
        # Split into 3 channels: MFCC, delta, delta2
        n = 40
        channels = np.stack([mfcc_mat[:n], mfcc_mat[n:2*n], mfcc_mat[2*n:]], axis=0)  # (3, 40, T)
        tensor = torch.tensor(channels, dtype=torch.float32).unsqueeze(0)  # (1, 3, 40, T)
        return tensor.to(self.device)

import torch
import numpy as np
from collections import deque
from typing import Tuple, Dict, Optional
from .pose_estimator import PoseEstimator
from .feature_extractor import BodyFeatureExtractor
from .temporal_model import BodyLSTM
from config.constants import PSYCHOLOGICAL_STATES, IDX_TO_STATE, NUM_CLASSES
from src.utils.logger import get_logger

logger = get_logger("body_classifier")


class BodyClassifier:
    """
    Full body agent pipeline:
      PoseEstimator → BodyFeatureExtractor → BodyLSTM → prediction

    When no model_path is supplied the classifier automatically falls back to
    RuleBasedBodyAgent (heuristic rules on pose landmarks — no training data
    needed).  Pass a checkpoint path to use the trained LSTM instead.

    Maintains a rolling sequence buffer so it can be called frame-by-frame
    during real-time inference.
    """

    def __init__(self, model_path: str = None, sequence_length: int = 30,
                 device: str = None):
        if device is None:
            self.device = (
                torch.device("cuda") if torch.cuda.is_available()
                else torch.device("mps") if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = torch.device(device)

        self.sequence_length = sequence_length

        # ── Choose backend ─────────────────────────────────────────────
        if model_path:
            # Trained LSTM path
            self._use_rules = False
            self.pose_estimator    = PoseEstimator()
            self.feature_extractor = BodyFeatureExtractor()
            self.model = BodyLSTM(
                input_size=99,
                hidden_size=256,
                num_layers=2,
                num_classes=NUM_CLASSES,
            ).to(self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            logger.info(f"BodyClassifier: loaded LSTM weights from {model_path}")
        else:
            # Rule-based fallback — no model file required
            self._use_rules = True
            from .rule_based_body_agent import RuleBasedBodyAgent
            self._rule_agent = RuleBasedBodyAgent(smoothing_window=sequence_length // 2)
            # Also keep these for get_embedding() LSTM path guard
            self.pose_estimator    = self._rule_agent.pose_estimator
            self.feature_extractor = self._rule_agent.feature_extractor
            logger.info("BodyClassifier: no model_path given — using RuleBasedBodyAgent")

        self._buffer: deque = deque(maxlen=sequence_length)

    def process_frame(self, frame: np.ndarray) -> Optional[Tuple[str, float, Dict[str, float]]]:
        """
        Process a single video frame. Returns a prediction once the sequence
        buffer is full (LSTM) or warm-up window is filled (rule-based).

        Args:
            frame: BGR image.

        Returns:
            (predicted_state, confidence, {state: probability}) or None.
        """
        if self._use_rules:
            return self._rule_agent.process_frame(frame)

        landmarks = self.pose_estimator.extract_pose(frame)
        if landmarks is None:
            return None

        feature_vec = self.feature_extractor.get_feature_vector(landmarks)
        self._buffer.append(feature_vec)

        if len(self._buffer) < self.sequence_length:
            return None

        return self._predict_from_buffer()

    def predict_sequence(self, sequence: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Run prediction directly on a pre-built sequence array (LSTM path only).

        Args:
            sequence: Array of shape (seq_len, 99).

        Returns:
            (predicted_state, confidence, {state: probability})
        """
        if self._use_rules:
            # Average rule scores across all frames in the sequence
            scores_list = []
            for frame_vec in sequence:
                landmarks = frame_vec.reshape(33, 3)
                scores_list.append(self._rule_agent._score_landmarks(landmarks))
            avg_scores = np.mean(scores_list, axis=0)
            return self._rule_agent._scores_to_result(avg_scores)

        tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        return self._run_inference(tensor)

    def get_embedding(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Return body feature embedding for fusion agent.
        Rule-based path returns a 5-dim score vector.
        LSTM path returns 256-dim hidden state (buffer must be full).
        """
        if self._use_rules:
            return self._rule_agent.get_embedding(frame)

        landmarks = self.pose_estimator.extract_pose(frame)
        if landmarks is None:
            return None

        feature_vec = self.feature_extractor.get_feature_vector(landmarks)
        self._buffer.append(feature_vec)

        if len(self._buffer) < self.sequence_length:
            return None

        sequence = np.array(self._buffer, dtype=np.float32)
        tensor = torch.tensor(sequence).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.get_features(tensor)
        return embedding.squeeze(0).cpu().numpy()

    def reset_buffer(self):
        self._buffer.clear()
        if self._use_rules:
            self._rule_agent.reset_buffer()

    # ------------------------------------------------------------------
    # Internal helpers (LSTM path only)
    # ------------------------------------------------------------------

    def _predict_from_buffer(self) -> Tuple[str, float, Dict[str, float]]:
        sequence = np.array(self._buffer, dtype=np.float32)
        tensor = torch.tensor(sequence).unsqueeze(0).to(self.device)
        return self._run_inference(tensor)

    def _run_inference(self, tensor: torch.Tensor) -> Tuple[str, float, Dict[str, float]]:
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        return (
            IDX_TO_STATE[pred_idx],
            float(probs[pred_idx]),
            {state: float(probs[i]) for i, state in enumerate(PSYCHOLOGICAL_STATES)},
        )

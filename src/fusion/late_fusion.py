import numpy as np
from typing import Tuple, Dict, List, Optional
from config.constants import PSYCHOLOGICAL_STATES, IDX_TO_STATE, NUM_CLASSES
from src.utils.logger import get_logger

logger = get_logger("late_fusion")


class LateFusionClassifier:
    """
    Late fusion: combines per-agent probability distributions into a
    single prediction using weighted averaging.

    Each agent independently predicts class probabilities;
    this module aggregates them with configurable weights.
    """

    def __init__(self, weights: Optional[List[float]] = None):
        """
        Args:
            weights: [body_weight, face_weight, speech_weight].
                     Defaults to equal weighting if None.
        """
        if weights is None:
            weights = [1.0, 1.0, 1.0]
        total = sum(weights)
        self.weights = [w / total for w in weights]
        logger.info(f"LateFusionClassifier initialized — weights: {self.weights}")

    def predict(self,
                body_probs: Dict[str, float],
                face_probs: Dict[str, float],
                speech_probs: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        """
        Weighted average of three agent probability dicts.

        Args:
            body_probs:   {state: probability} from body agent.
            face_probs:   {state: probability} from face agent.
            speech_probs: {state: probability} from speech agent.

        Returns:
            (predicted_state, confidence, {state: fused_probability})
        """
        fused = {}
        for state in PSYCHOLOGICAL_STATES:
            fused[state] = (
                self.weights[0] * body_probs.get(state, 0.0) +
                self.weights[1] * face_probs.get(state, 0.0) +
                self.weights[2] * speech_probs.get(state, 0.0)
            )

        best_state = max(fused, key=fused.get)
        return best_state, float(fused[best_state]), fused

    def predict_from_arrays(self,
                            body_probs: np.ndarray,
                            face_probs: np.ndarray,
                            speech_probs: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Same as predict() but accepts numpy probability arrays directly.

        Args:
            body_probs, face_probs, speech_probs: Arrays of shape (num_classes,).
        """
        fused_arr = (
            self.weights[0] * body_probs +
            self.weights[1] * face_probs +
            self.weights[2] * speech_probs
        )
        idx = int(np.argmax(fused_arr))
        fused_dict = {s: float(fused_arr[i]) for i, s in enumerate(PSYCHOLOGICAL_STATES)}
        return IDX_TO_STATE[idx], float(fused_arr[idx]), fused_dict

    def update_weights(self, weights: List[float]):
        """Dynamically update fusion weights (e.g., when one modality is unavailable)."""
        total = sum(weights)
        self.weights = [w / total for w in weights]
        logger.info(f"Updated fusion weights: {self.weights}")

    def predict_with_missing(self,
                             body_probs: Optional[np.ndarray],
                             face_probs: Optional[np.ndarray],
                             speech_probs: Optional[np.ndarray]) -> Tuple[str, float, Dict[str, float]]:
        """
        Handle missing modalities gracefully by re-normalizing weights.
        Pass None for any unavailable modality.
        """
        available = [
            (self.weights[0], body_probs),
            (self.weights[1], face_probs),
            (self.weights[2], speech_probs),
        ]
        present = [(w, p) for w, p in available if p is not None]

        if not present:
            uniform = np.ones(NUM_CLASSES) / NUM_CLASSES
            return PSYCHOLOGICAL_STATES[0], 1.0 / NUM_CLASSES, \
                   {s: 1.0 / NUM_CLASSES for s in PSYCHOLOGICAL_STATES}

        total_w = sum(w for w, _ in present)
        fused_arr = sum((w / total_w) * p for w, p in present)
        idx = int(np.argmax(fused_arr))
        fused_dict = {s: float(fused_arr[i]) for i, s in enumerate(PSYCHOLOGICAL_STATES)}
        return IDX_TO_STATE[idx], float(fused_arr[idx]), fused_dict

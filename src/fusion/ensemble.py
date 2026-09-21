import numpy as np
from typing import Tuple, Dict, List, Optional
from config.constants import PSYCHOLOGICAL_STATES, IDX_TO_STATE, NUM_CLASSES
from src.utils.logger import get_logger

logger = get_logger("ensemble")


class FusionEnsemble:
    """
    Combines predictions from multiple fusion strategies
    (early, late, attention) using majority voting or weighted averaging.

    Use this as the final decision layer when running all fusion modes.
    """

    STRATEGIES = ["early", "late", "attention"]

    def __init__(self, strategy_weights: Optional[Dict[str, float]] = None):
        """
        Args:
            strategy_weights: e.g. {"early": 0.2, "late": 0.3, "attention": 0.5}
                              Defaults to equal weighting.
        """
        if strategy_weights is None:
            strategy_weights = {s: 1.0 for s in self.STRATEGIES}
        total = sum(strategy_weights.values())
        self.weights = {k: v / total for k, v in strategy_weights.items()}
        logger.info(f"FusionEnsemble initialized — weights: {self.weights}")

    def weighted_average(self, predictions: Dict[str, Dict[str, float]]) -> Tuple[str, float, Dict[str, float]]:
        """
        Weighted average of probability dicts from multiple fusion strategies.

        Args:
            predictions: {"early": {state: prob}, "late": {state: prob}, ...}

        Returns:
            (best_state, confidence, {state: averaged_prob})
        """
        fused = {s: 0.0 for s in PSYCHOLOGICAL_STATES}

        for strategy, probs in predictions.items():
            w = self.weights.get(strategy, 0.0)
            for state in PSYCHOLOGICAL_STATES:
                fused[state] += w * probs.get(state, 0.0)

        best_state = max(fused, key=fused.get)
        return best_state, float(fused[best_state]), fused

    def majority_vote(self, predictions: Dict[str, str]) -> str:
        """
        Majority vote over predicted class labels.

        Args:
            predictions: {"early": "calm", "late": "stressed", "attention": "calm"}

        Returns:
            Winning class label.
        """
        votes: Dict[str, int] = {}
        for strategy, label in predictions.items():
            votes[label] = votes.get(label, 0) + 1
        return max(votes, key=votes.get)

    def confidence_weighted(self, predictions: Dict[str, Tuple[str, float, Dict[str, float]]]) -> Tuple[str, float, Dict[str, float]]:
        """
        Weight each strategy's output by its own confidence score.

        Args:
            predictions: {"strategy": (label, confidence, prob_dict)}

        Returns:
            (best_state, confidence, {state: prob})
        """
        fused = {s: 0.0 for s in PSYCHOLOGICAL_STATES}
        total_conf = sum(conf for _, conf, _ in predictions.values()) + 1e-9

        for strategy, (label, conf, probs) in predictions.items():
            w = conf / total_conf
            for state in PSYCHOLOGICAL_STATES:
                fused[state] += w * probs.get(state, 0.0)

        best_state = max(fused, key=fused.get)
        return best_state, float(fused[best_state]), fused


class TemporalSmoother:
    """
    Smooths predictions over time using an exponential moving average.
    Reduces flickering in real-time inference.
    """

    def __init__(self, alpha: float = 0.3):
        """
        Args:
            alpha: Smoothing factor (0 = no smoothing, 1 = no memory).
        """
        self.alpha = alpha
        self._smoothed: Optional[np.ndarray] = None
        logger.info(f"TemporalSmoother initialized — alpha: {alpha}")

    def update(self, probs: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        """
        Apply EMA smoothing to incoming probability distribution.

        Args:
            probs: {state: probability} from current frame.

        Returns:
            (smoothed_state, confidence, smoothed_prob_dict)
        """
        arr = np.array([probs.get(s, 0.0) for s in PSYCHOLOGICAL_STATES], dtype=np.float32)

        if self._smoothed is None:
            self._smoothed = arr
        else:
            self._smoothed = self.alpha * arr + (1 - self.alpha) * self._smoothed

        idx = int(np.argmax(self._smoothed))
        smoothed_dict = {s: float(self._smoothed[i]) for i, s in enumerate(PSYCHOLOGICAL_STATES)}
        return IDX_TO_STATE[idx], float(self._smoothed[idx]), smoothed_dict

    def reset(self):
        self._smoothed = None

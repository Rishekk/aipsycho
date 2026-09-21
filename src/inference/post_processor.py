import numpy as np
from collections import deque
from typing import Dict, Tuple, List
from config.constants import PSYCHOLOGICAL_STATES, IDX_TO_STATE
from src.utils.logger import get_logger

logger = get_logger("post_processor")

# State → suggested intervention message
STATE_MESSAGES = {
    "calm":       "You appear calm and composed.",
    "confident":  "You appear confident and engaged.",
    "stressed":   "Signs of stress detected. Consider taking a deep breath.",
    "anxious":    "Signs of anxiety detected. Try grounding techniques.",
    "frustrated": "Signs of frustration detected. Take a short break if possible.",
}

# State → color (R, G, B) for UI rendering
STATE_COLORS = {
    "calm":       (0,   200, 100),
    "confident":  (0,   150, 255),
    "stressed":   (255, 150,   0),
    "anxious":    (255, 200,   0),
    "frustrated": (220,  50,  50),
}


class PostProcessor:
    """
    Post-processes raw model predictions for display:
    - Majority-vote smoothing over a short history window
    - Confidence thresholding (low-confidence → "uncertain")
    - Human-readable message and color lookup
    - Per-session statistics
    """

    def __init__(self, window_size: int = 15, confidence_threshold: float = 0.4):
        self.window_size          = window_size
        self.confidence_threshold = confidence_threshold
        self._history: deque = deque(maxlen=window_size)
        self._session_counts: Dict[str, int] = {s: 0 for s in PSYCHOLOGICAL_STATES}
        self._frame_count = 0
        logger.info(f"PostProcessor initialized — window: {window_size}, threshold: {confidence_threshold}")

    def process(self, state: str, confidence: float,
                probs: Dict[str, float]) -> Dict:
        """
        Process a single prediction and return enriched output.

        Args:
            state:      Raw predicted state string.
            confidence: Confidence score (0–1).
            probs:      Full probability dict.

        Returns:
            Dict with smoothed_state, confidence, message, color, session_stats.
        """
        self._frame_count += 1

        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            display_state = "uncertain"
        else:
            display_state = state
            self._history.append(state)
            self._session_counts[state] = self._session_counts.get(state, 0) + 1

        # Majority vote over history window
        smoothed_state = self._majority_vote() if self._history else display_state

        return {
            "state":         smoothed_state,
            "raw_state":     state,
            "confidence":    round(confidence, 3),
            "probs":         {k: round(v, 3) for k, v in probs.items()},
            "message":       STATE_MESSAGES.get(smoothed_state, ""),
            "color":         STATE_COLORS.get(smoothed_state, (200, 200, 200)),
            "session_stats": self.get_session_stats(),
        }

    def get_session_stats(self) -> Dict[str, float]:
        """Return percentage of time spent in each state this session."""
        total = max(sum(self._session_counts.values()), 1)
        return {s: round(c / total, 3) for s, c in self._session_counts.items()}

    def get_dominant_state(self) -> str:
        """Return the most frequent state of the entire session."""
        return max(self._session_counts, key=self._session_counts.get)

    def reset_session(self):
        self._history.clear()
        self._session_counts = {s: 0 for s in PSYCHOLOGICAL_STATES}
        self._frame_count = 0

    def _majority_vote(self) -> str:
        votes: Dict[str, int] = {}
        for s in self._history:
            votes[s] = votes.get(s, 0) + 1
        return max(votes, key=votes.get)

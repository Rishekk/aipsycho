import numpy as np
from collections import deque
from typing import Optional, Dict, List
from src.utils.logger import get_logger

logger = get_logger("micro_expression")

# Thresholds for AU-inspired micro-expression detection
EAR_BLINK_THRESHOLD = 0.25       # EAR below this = eye closed
BLINK_CONSEC_FRAMES = 2          # frames to confirm a blink
MAR_OPEN_THRESHOLD = 0.5         # MAR above this = mouth open
BROW_RAISE_THRESHOLD = 0.04      # brow raise above this = surprise/stress
SMILE_THRESHOLD = -0.015         # negative = corners raised (smile)


class MicroExpressionDetector:
    """
    Detects micro-expressions and transient facial action units
    from sequences of geometric features extracted by LandmarkExtractor.

    Works on a rolling window of recent frames to spot brief movements.
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self._history: deque = deque(maxlen=window_size)
        self._blink_counter = 0
        self._blink_total = 0
        logger.info(f"MicroExpressionDetector initialized (window={window_size}).")

    def update(self, features: Dict[str, float]):
        """
        Add latest frame's geometric features to the rolling window.

        Args:
            features: Output of LandmarkExtractor.extract_features().
        """
        self._history.append(features)
        self._track_blinks(features)

    def detect(self) -> Dict[str, object]:
        """
        Analyse the current window and return detected micro-expressions.

        Returns:
            Dict with detected events and stress indicators.
        """
        if len(self._history) < 3:
            return {}

        results: Dict[str, object] = {}

        # --- Blink rate ---
        results["blink_total"] = self._blink_total

        # --- EAR stability (tension / squinting) ---
        ears = [f["avg_ear"] for f in self._history]
        results["ear_mean"] = float(np.mean(ears))
        results["ear_std"] = float(np.std(ears))
        results["squinting"] = bool(results["ear_mean"] < EAR_BLINK_THRESHOLD + 0.05
                                    and results["ear_std"] < 0.02)

        # --- Eyebrow raise (surprise / stress) ---
        brow_raises = [
            (f["left_brow_raise"] + f["right_brow_raise"]) / 2
            for f in self._history
        ]
        results["avg_brow_raise"] = float(np.mean(brow_raises))
        results["brow_raised"] = bool(results["avg_brow_raise"] > BROW_RAISE_THRESHOLD)

        # --- Smile detection ---
        smiles = [
            (f["smile_left"] + f["smile_right"]) / 2
            for f in self._history
        ]
        results["smile_score"] = float(np.mean(smiles))
        results["smiling"] = bool(results["smile_score"] < SMILE_THRESHOLD)

        # --- Mouth tension ---
        mars = [f["mar"] for f in self._history]
        results["mar_mean"] = float(np.mean(mars))
        results["mouth_open"] = bool(results["mar_mean"] > MAR_OPEN_THRESHOLD)

        # --- Micro-expression spike: sudden AU change in a short window ---
        results["micro_event"] = self._detect_spike(ears) or self._detect_spike(brow_raises)

        # --- Inferred stress signal from AUs ---
        results["stress_signal"] = self._infer_stress(results)

        return results

    def get_stress_score(self) -> float:
        """
        Return a 0–1 stress score based on current window.
        Useful as an auxiliary input to the fusion agent.
        """
        detected = self.detect()
        if not detected:
            return 0.0
        return float(detected.get("stress_signal", 0.0))

    def reset(self):
        """Clear history and counters."""
        self._history.clear()
        self._blink_counter = 0
        self._blink_total = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _track_blinks(self, features: Dict[str, float]):
        ear = features.get("avg_ear", 1.0)
        if ear < EAR_BLINK_THRESHOLD:
            self._blink_counter += 1
        else:
            if self._blink_counter >= BLINK_CONSEC_FRAMES:
                self._blink_total += 1
            self._blink_counter = 0

    def _detect_spike(self, values: List[float], threshold_factor: float = 2.0) -> bool:
        """
        Return True if there is a sudden spike > threshold_factor * std
        relative to the rolling mean — indicates a brief micro-movement.
        """
        if len(values) < 4:
            return False
        arr = np.array(values)
        mean, std = arr.mean(), arr.std()
        if std < 1e-6:
            return False
        return bool(np.any(np.abs(arr - mean) > threshold_factor * std))

    def _infer_stress(self, results: Dict) -> float:
        """
        Combine AU signals into a simple 0–1 stress score.
        Higher = more stress indicators present.
        """
        score = 0.0
        if results.get("brow_raised"):
            score += 0.3
        if results.get("squinting"):
            score += 0.2
        if results.get("micro_event"):
            score += 0.25
        if not results.get("smiling"):
            score += 0.1
        if results.get("mouth_open"):
            score += 0.15
        return min(score, 1.0)

"""
Rule-based body language agent.

Uses geometric features from MediaPipe pose landmarks to score each
psychological state with hand-crafted heuristics.  No model file or
training data required — it can be used as a drop-in replacement for
BodyClassifier when no trained checkpoint is available.

State heuristics (image-space coordinates, y increases downward):
  calm       — relaxed, upright, symmetric posture, arms loosely at sides
  stressed   — forward head, compressed torso, hunched shoulders
  anxious    — closed posture (crossed/low arms), narrow shoulders
  confident  — open posture, wide shoulders, upright torso
  frustrated — tense / asymmetric posture, elevated or erratic shoulders

All scores are soft (0.0–1.0) and get softmax-normalised to probabilities.
"""
import numpy as np
from typing import Tuple, Dict, Optional
from collections import deque

from .pose_estimator import PoseEstimator
from .feature_extractor import BodyFeatureExtractor
from config.constants import PSYCHOLOGICAL_STATES, IDX_TO_STATE, STATE_TO_IDX
from src.utils.logger import get_logger

logger = get_logger("rule_based_body_agent")

# Landmark indices (same as in feature_extractor.py)
L_SHOULDER, R_SHOULDER = 11, 12
L_WRIST,    R_WRIST    = 15, 16
L_HIP,      R_HIP      = 23, 24
NOSE                   = 0


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    e = np.exp((x - x.max()) / temperature)
    return e / e.sum()


class RuleBasedBodyAgent:
    """
    Heuristic-based psychological state estimator from body pose.

    Processes frames in a rolling window (default 15 frames) and
    averages scores for temporal smoothing.
    """

    def __init__(self, smoothing_window: int = 15):
        self.pose_estimator    = PoseEstimator()
        self.feature_extractor = BodyFeatureExtractor()
        self._score_buffer: deque = deque(maxlen=smoothing_window)
        logger.info("RuleBasedBodyAgent initialised (no model needed).")

    # ------------------------------------------------------------------
    # Public API — matches BodyClassifier interface
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> Optional[Tuple[str, float, Dict[str, float]]]:
        """
        Process one BGR frame.  Returns prediction once the warm-up
        window is filled, otherwise None.
        """
        landmarks = self.pose_estimator.extract_pose(frame)
        if landmarks is None:
            return None

        scores = self._score_landmarks(landmarks)
        self._score_buffer.append(scores)

        # Average over window for temporal smoothing
        avg_scores = np.mean(np.stack(list(self._score_buffer)), axis=0)
        return self._scores_to_result(avg_scores)

    def predict_landmarks(self, landmarks: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """Direct prediction from a (33, 3) landmark array."""
        scores = self._score_landmarks(landmarks)
        return self._scores_to_result(scores)

    def get_embedding(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Return a pseudo-embedding (raw score vector, length 5) for fusion.
        Compatible with AttentionFusionModel when body_dim=5.
        """
        landmarks = self.pose_estimator.extract_pose(frame)
        if landmarks is None:
            return None
        return self._score_landmarks(landmarks)

    def reset_buffer(self):
        self._score_buffer.clear()

    # ------------------------------------------------------------------
    # Core heuristics
    # ------------------------------------------------------------------

    def _score_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Compute a raw score for each of the 5 psychological states.

        Args:
            landmarks: (33, 3) array in normalized MediaPipe image coords
                       (x, y ∈ [0,1], y increases downward, z = depth).

        Returns:
            float32 array of shape (5,) — one score per state.
        """
        f = self.feature_extractor.extract_features(landmarks)
        scores = np.zeros(5, dtype=np.float32)

        # ── Derived signals ─────────────────────────────────────────────
        # shoulder_width: wider → more open/confident
        shoulder_width = f["shoulder_width"]

        # torso_lean_y: in image space shoulder_y < hip_y (shoulders above),
        # so torso_lean_y should be negative (shoulder_mid_y - hip_mid_y < 0).
        # A less-negative value (closer to 0) means more compressed/hunched posture.
        torso_lean_y = f["torso_lean_y"]          # typically negative (−0.3 … −0.05)
        torso_upright = -torso_lean_y              # positive; larger = more upright

        # head forward: in image space, head_y_offset = nose_y - shoulder_mid_y.
        # Positive = nose below shoulder line = head down / forward.
        head_forward = f["head_y_offset"]           # >0 means head down

        # wrist heights: shoulder_y - wrist_y (image space)
        # Negative → wrist is below shoulder (arms down)
        # Positive → wrist is above shoulder (arms raised)
        left_wrist_h  = f["left_wrist_height"]
        right_wrist_h = f["right_wrist_height"]
        avg_wrist_h   = (left_wrist_h + right_wrist_h) / 2

        # cross-body wrist distances (small → arms crossed)
        cross_dist = min(f["left_wrist_to_right_shoulder"],
                         f["right_wrist_to_left_shoulder"])

        # shoulder asymmetry (tilt magnitude)
        shoulder_asym = abs(f["shoulder_tilt"])

        # elbow angles: more extended (close to 180°) = more open
        avg_elbow_angle = (f["left_elbow_angle"] + f["right_elbow_angle"]) / 2

        # ── calm ────────────────────────────────────────────────────────
        # Moderately wide shoulders, upright torso, arms loosely down,
        # head not forward, symmetric.
        calm = 0.0
        calm += _sigmoid(shoulder_width,    centre=0.18, steepness=30)   # moderate-wide
        calm += _sigmoid(torso_upright,     centre=0.12, steepness=40)   # upright
        calm -= _sigmoid(abs(head_forward), centre=0.04, steepness=60)   # not forward
        calm -= _sigmoid(shoulder_asym,     centre=0.02, steepness=80)   # symmetric
        # arms mostly down
        calm += _sigmoid(-avg_wrist_h,      centre=0.05, steepness=30)   # wrist below shoulder
        scores[STATE_TO_IDX["calm"]] = max(calm, 0.0)

        # ── stressed ─────────────────────────────────────────────────────
        # Hunched / compressed torso, head forward, shoulders rounded inward.
        stressed = 0.0
        stressed += _sigmoid(-torso_upright,    centre=-0.10, steepness=40)  # less upright
        stressed += _sigmoid(head_forward,      centre=0.04,  steepness=60)  # head down/forward
        stressed += _sigmoid(-shoulder_width,   centre=-0.20, steepness=30)  # narrower shoulders
        stressed -= _sigmoid(avg_elbow_angle,   centre=150,   steepness=0.05) # bent arms
        scores[STATE_TO_IDX["stressed"]] = max(stressed, 0.0)

        # ── anxious ──────────────────────────────────────────────────────
        # Closed posture: narrow shoulders, possibly arms crossed,
        # wrists low or pulled in, slight forward head.
        anxious = 0.0
        anxious += _sigmoid(-shoulder_width,    centre=-0.22, steepness=35)  # narrow
        anxious += _sigmoid(-cross_dist,        centre=-0.25, steepness=20)  # arms crossed/close
        anxious += _sigmoid(head_forward,       centre=0.03,  steepness=50)  # head slightly forward
        anxious -= _sigmoid(avg_wrist_h,        centre=0.0,   steepness=30)  # wrists not raised
        scores[STATE_TO_IDX["anxious"]] = max(anxious, 0.0)

        # ── confident ────────────────────────────────────────────────────
        # Wide shoulders, upright torso, head up (slightly back), open arms.
        confident = 0.0
        confident += _sigmoid(shoulder_width,   centre=0.22,  steepness=35)  # wide
        confident += _sigmoid(torso_upright,    centre=0.15,  steepness=40)  # very upright
        confident -= _sigmoid(head_forward,     centre=0.02,  steepness=80)  # head NOT forward
        confident += _sigmoid(avg_elbow_angle,  centre=140,   steepness=0.05) # arms more extended
        confident -= _sigmoid(shoulder_asym,    centre=0.02,  steepness=80)  # symmetric
        scores[STATE_TO_IDX["confident"]] = max(confident, 0.0)

        # ── frustrated ───────────────────────────────────────────────────
        # Tense / asymmetric: raised shoulders, high wrists, asymmetry.
        frustrated = 0.0
        frustrated += _sigmoid(shoulder_asym,   centre=0.025, steepness=80)  # asymmetric
        frustrated += _sigmoid(avg_wrist_h,     centre=0.02,  steepness=40)  # wrists raised
        frustrated += _sigmoid(-shoulder_width, centre=-0.15, steepness=25)  # somewhat narrow
        frustrated -= _sigmoid(torso_upright,   centre=0.15,  steepness=40)  # less upright
        scores[STATE_TO_IDX["frustrated"]] = max(frustrated, 0.0)

        return scores

    @staticmethod
    def _scores_to_result(scores: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        probs   = _softmax(scores, temperature=0.5)
        idx     = int(np.argmax(probs))
        state   = IDX_TO_STATE[idx]
        conf    = float(probs[idx])
        prob_dict = {s: float(probs[i]) for i, s in enumerate(PSYCHOLOGICAL_STATES)}
        return state, conf, prob_dict


# ── Helper ────────────────────────────────────────────────────────────────────

def _sigmoid(x: float, centre: float, steepness: float) -> float:
    """Soft step function: 1 / (1 + exp(−steepness*(x − centre)))."""
    return float(1.0 / (1.0 + np.exp(-steepness * (x - centre))))

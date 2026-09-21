import numpy as np
from typing import Dict
from .pose_estimator import PoseEstimator
from src.utils.logger import get_logger

logger = get_logger("body_feature_extractor")

# Landmark index constants
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW,    R_ELBOW    = 13, 14
L_WRIST,    R_WRIST    = 15, 16
L_HIP,      R_HIP      = 23, 24
L_KNEE,     R_KNEE     = 25, 26
L_ANKLE,    R_ANKLE    = 27, 28
NOSE                   = 0


class BodyFeatureExtractor:
    """
    Computes geometric and spatial features from 33 body pose landmarks.
    These features are used directly or as input to the temporal LSTM model.
    """

    def extract_features(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Compute named geometric features from a single frame's landmarks.

        Args:
            landmarks: Array of shape (33, 3).

        Returns:
            Dict of feature name → value.
        """
        features = {}

        # --- Joint angles ---
        features["left_elbow_angle"]  = self._angle(landmarks[L_SHOULDER], landmarks[L_ELBOW],  landmarks[L_WRIST])
        features["right_elbow_angle"] = self._angle(landmarks[R_SHOULDER], landmarks[R_ELBOW],  landmarks[R_WRIST])
        features["left_shoulder_angle"]  = self._angle(landmarks[L_HIP],  landmarks[L_SHOULDER], landmarks[L_ELBOW])
        features["right_shoulder_angle"] = self._angle(landmarks[R_HIP],  landmarks[R_SHOULDER], landmarks[R_ELBOW])
        features["left_knee_angle"]  = self._angle(landmarks[L_HIP],  landmarks[L_KNEE], landmarks[L_ANKLE])
        features["right_knee_angle"] = self._angle(landmarks[R_HIP],  landmarks[R_KNEE], landmarks[R_ANKLE])
        features["left_hip_angle"]   = self._angle(landmarks[L_SHOULDER], landmarks[L_HIP], landmarks[L_KNEE])
        features["right_hip_angle"]  = self._angle(landmarks[R_SHOULDER], landmarks[R_HIP], landmarks[R_KNEE])

        # --- Shoulder symmetry & tilt ---
        shoulder_diff = landmarks[L_SHOULDER] - landmarks[R_SHOULDER]
        features["shoulder_tilt"] = float(shoulder_diff[1])             # y-diff (positive = left higher)
        features["shoulder_width"] = float(np.linalg.norm(shoulder_diff[:2]))

        # --- Hip symmetry ---
        hip_diff = landmarks[L_HIP] - landmarks[R_HIP]
        features["hip_tilt"]  = float(hip_diff[1])
        features["hip_width"] = float(np.linalg.norm(hip_diff[:2]))

        # --- Torso lean (shoulder midpoint vs hip midpoint) ---
        shoulder_mid = (landmarks[L_SHOULDER] + landmarks[R_SHOULDER]) / 2
        hip_mid      = (landmarks[L_HIP]      + landmarks[R_HIP])      / 2
        torso_vec    = shoulder_mid - hip_mid
        features["torso_lean_x"] = float(torso_vec[0])   # lateral lean
        features["torso_lean_y"] = float(torso_vec[1])   # vertical lean
        features["torso_length"] = float(np.linalg.norm(torso_vec[:2]))

        # --- Head position relative to shoulders ---
        features["head_x_offset"] = float(landmarks[NOSE][0] - shoulder_mid[0])
        features["head_y_offset"] = float(landmarks[NOSE][1] - shoulder_mid[1])

        # --- Wrist heights (raised/lowered arms indicator) ---
        features["left_wrist_height"]  = float(landmarks[L_SHOULDER][1] - landmarks[L_WRIST][1])
        features["right_wrist_height"] = float(landmarks[R_SHOULDER][1] - landmarks[R_WRIST][1])

        # --- Cross-body distances (crossed arms) ---
        features["left_wrist_to_right_shoulder"]  = float(np.linalg.norm(landmarks[L_WRIST] - landmarks[R_SHOULDER]))
        features["right_wrist_to_left_shoulder"]  = float(np.linalg.norm(landmarks[R_WRIST] - landmarks[L_SHOULDER]))

        return features

    def get_feature_vector(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Flatten all 33 landmarks into a 1D vector (99 dims).
        Used directly as input to the LSTM model.

        Args:
            landmarks: Array of shape (33, 3).

        Returns:
            1D array of length 99.
        """
        return landmarks.flatten()

    def get_named_vector(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Compute named features and return as a sorted numpy array.

        Args:
            landmarks: Array of shape (33, 3).

        Returns:
            1D array of named feature values.
        """
        feats = self.extract_features(landmarks)
        return np.array(list(feats.values()), dtype=np.float32)

    @staticmethod
    def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        Compute the angle (degrees) at joint b formed by vectors b→a and b→c.
        Uses only x and y coordinates.
        """
        ba = a[:2] - b[:2]
        bc = c[:2] - b[:2]
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

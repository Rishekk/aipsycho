import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from typing import Optional, Dict
from src.utils.logger import get_logger

logger = get_logger("landmark_extractor")

DEFAULT_MODEL = "models/mediapipe/face_landmarker.task"


class LandmarkExtractor:
    """Extracts 478 facial landmarks and geometric features using MediaPipe Tasks API."""

    LEFT_EYE      = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE     = [362, 385, 387, 263, 373, 380]
    MOUTH_OUTER   = [61, 291, 39, 181, 0, 17, 269, 405]
    MOUTH_INNER   = [78, 308, 82, 87, 13, 14, 312, 317]
    LEFT_EYEBROW  = [70, 63, 105, 66, 107]
    RIGHT_EYEBROW = [336, 296, 334, 293, 300]
    NOSE          = [1, 2, 5, 4, 19, 94]
    JAW           = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397]

    def __init__(self, model_path: str = DEFAULT_MODEL,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.face_mesh = mp_vision.FaceLandmarker.create_from_options(options)
        logger.info("LandmarkExtractor initialized.")

    def extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 478 facial landmarks from frame.

        Returns:
            Array of shape (478, 3) or None.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.face_mesh.detect(mp_image)

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    def extract_features(self, landmarks: np.ndarray) -> Dict[str, float]:
        features = {}
        features["left_ear"]        = self._eye_aspect_ratio(landmarks, self.LEFT_EYE)
        features["right_ear"]       = self._eye_aspect_ratio(landmarks, self.RIGHT_EYE)
        features["avg_ear"]         = (features["left_ear"] + features["right_ear"]) / 2
        features["mar"]             = self._mouth_aspect_ratio(landmarks, self.MOUTH_OUTER)
        features["left_brow_raise"] = self._brow_raise(landmarks, self.LEFT_EYEBROW,  self.LEFT_EYE)
        features["right_brow_raise"]= self._brow_raise(landmarks, self.RIGHT_EYEBROW, self.RIGHT_EYE)
        left_corner  = landmarks[61]
        right_corner = landmarks[291]
        mouth_center = landmarks[13]
        features["smile_left"]    = float(mouth_center[1] - left_corner[1])
        features["smile_right"]   = float(mouth_center[1] - right_corner[1])
        features["nose_to_mouth"] = float(np.linalg.norm(landmarks[4] - mouth_center))
        return features

    def get_feature_vector(self, landmarks: np.ndarray) -> np.ndarray:
        key_indices = (
            self.LEFT_EYE + self.RIGHT_EYE +
            self.MOUTH_OUTER + self.MOUTH_INNER +
            self.LEFT_EYEBROW + self.RIGHT_EYEBROW +
            self.NOSE + self.JAW
        )
        return landmarks[key_indices].flatten()

    def draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        for idx in self.LEFT_EYE + self.RIGHT_EYE + self.MOUTH_OUTER:
            x = int(landmarks[idx, 0] * w)
            y = int(landmarks[idx, 1] * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
        return frame

    def close(self):
        pass

    def _eye_aspect_ratio(self, landmarks, indices):
        pts = landmarks[indices]
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        h  = np.linalg.norm(pts[0] - pts[3])
        return float((v1 + v2) / (2.0 * h + 1e-6))

    def _mouth_aspect_ratio(self, landmarks, indices):
        pts = landmarks[indices]
        v1 = np.linalg.norm(pts[2] - pts[6])
        v2 = np.linalg.norm(pts[3] - pts[5])
        h  = np.linalg.norm(pts[0] - pts[1])
        return float((v1 + v2) / (2.0 * h + 1e-6))

    def _brow_raise(self, landmarks, brow_indices, eye_indices):
        return float(np.mean(landmarks[eye_indices, 1]) - np.mean(landmarks[brow_indices, 1]))

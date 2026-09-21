import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger("pose_estimator")

DEFAULT_MODEL = "models/mediapipe/pose_landmarker_lite.task"


class PoseEstimator:
    """Extracts 33 body pose landmarks from video frames using MediaPipe Tasks API."""

    NOSE           = 0
    LEFT_SHOULDER  = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW     = 13
    RIGHT_ELBOW    = 14
    LEFT_WRIST     = 15
    RIGHT_WRIST    = 16
    LEFT_HIP       = 23
    RIGHT_HIP      = 24
    LEFT_KNEE      = 25
    RIGHT_KNEE     = 26
    LEFT_ANKLE     = 27
    RIGHT_ANKLE    = 28

    def __init__(self, model_path: str = DEFAULT_MODEL,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        # Force CPU delegate — GPU/OpenGL is not available in all environments
        # (e.g. macOS without discrete GPU, headless servers).
        base_options = mp_python.BaseOptions(
            model_asset_path=model_path,
            delegate=mp_python.BaseOptions.Delegate.CPU,
        )
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.pose = mp_vision.PoseLandmarker.create_from_options(options)
        logger.info("PoseEstimator initialized (CPU delegate).")

    def extract_pose(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 33 pose landmarks from a frame.

        Returns:
            Array of shape (33, 3) or None.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.pose.detect(mp_image)

        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    def draw_pose(self, frame: np.ndarray) -> np.ndarray:
        """Draw pose skeleton overlay on frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.pose.detect(mp_image)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            h, w = frame.shape[:2]
            # Draw keypoints
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
        return frame

    def close(self):
        pass

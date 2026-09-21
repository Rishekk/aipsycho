import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from typing import Optional, Tuple
from src.utils.logger import get_logger

logger = get_logger("face_detector")

DEFAULT_MODEL = "models/mediapipe/blaze_face_short_range.tflite"


class FaceDetector:
    """Detects and aligns faces from video frames using MediaPipe Tasks API."""

    def __init__(self, model_path: str = DEFAULT_MODEL,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_detection_confidence,
        )
        self.detector = mp_vision.FaceDetector.create_from_options(options)
        logger.info("FaceDetector initialized.")

    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the largest face bounding box in frame.

        Args:
            frame: BGR image as numpy array.

        Returns:
            (x, y, w, h) bounding box or None if no face found.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        if not result.detections:
            return None

        h, w = frame.shape[:2]
        best = max(result.detections, key=lambda d: d.categories[0].score)
        bbox = best.bounding_box

        x  = max(0, bbox.origin_x)
        y  = max(0, bbox.origin_y)
        bw = min(bbox.width,  w - x)
        bh = min(bbox.height, h - y)
        return (x, y, bw, bh)

    def crop_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                  target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        x, y, bw, bh = bbox
        pad_x = int(bw * 0.2)
        pad_y = int(bh * 0.2)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame.shape[1], x + bw + pad_x)
        y2 = min(frame.shape[0], y + bh + pad_y)
        face = frame[y1:y2, x1:x2]
        return cv2.resize(face, target_size)

    def detect_and_crop(self, frame: np.ndarray,
                        target_size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
        bbox = self.detect(frame)
        if bbox is None:
            return None
        return self.crop_face(frame, bbox, target_size)

    def draw_bbox(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame

    def close(self):
        pass  # Tasks API handles cleanup automatically

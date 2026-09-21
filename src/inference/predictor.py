import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from src.agents.face_agent import FaceDetector, LandmarkExtractor, FaceClassifier, MicroExpressionDetector
from src.agents.body_agent import BodyClassifier
from src.agents.speech_agent import SpeechClassifier, AudioProcessor
from src.fusion.late_fusion import LateFusionClassifier
from src.fusion.attention_fusion import AttentionFusionClassifier
from .post_processor import PostProcessor
from src.utils.logger import get_logger

logger = get_logger("predictor")


class BatchPredictor:
    """
    Batch inference on pre-recorded video and audio files.
    Processes frame-by-frame and returns per-frame + aggregate predictions.
    """

    def __init__(self,
                 face_model_path:   Optional[str] = None,
                 body_model_path:   Optional[str] = None,
                 speech_model_path: Optional[str] = None,
                 fusion_strategy:   str = "late",
                 device:            str = None):

        logger.info("Initializing BatchPredictor...")
        self.face_classifier  = FaceClassifier(model_path=face_model_path,   device=device)
        self.face_detector    = FaceDetector()
        self.landmark_extractor = LandmarkExtractor()
        self.body_classifier  = BodyClassifier(model_path=body_model_path,   device=device)
        self.speech_classifier = SpeechClassifier(model_path=speech_model_path, device=device)
        self.post_processor   = PostProcessor()

        if fusion_strategy == "late":
            self.fusion = LateFusionClassifier(weights=[1.0, 1.0, 1.0])
        elif fusion_strategy == "attention":
            self.fusion = AttentionFusionClassifier(device=device)
        else:
            self.fusion = LateFusionClassifier()

        self.fusion_strategy = fusion_strategy
        logger.info(f"BatchPredictor ready — fusion: {fusion_strategy}")

    def predict_video(self, video_path: str) -> Dict:
        """
        Run full pipeline on a video file.

        Args:
            video_path: Path to .mp4 or other video file.

        Returns:
            Dict with per-frame predictions and aggregate stats.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        frame_results: List[Dict] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self._process_frame(frame)
            if result:
                result["frame"] = frame_idx
                frame_results.append(result)

            frame_idx += 1

        cap.release()
        return {
            "frames":          frame_results,
            "total_frames":    frame_idx,
            "session_stats":   self.post_processor.get_session_stats(),
            "dominant_state":  self.post_processor.get_dominant_state(),
        }

    def _process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """Process one frame through face + body agents and fuse."""
        results = {}

        # Face
        face_crop = self.face_detector.detect_and_crop(frame)
        if face_crop is not None:
            f_state, f_conf, f_probs = self.face_classifier.predict(face_crop)
            results["face"] = {"state": f_state, "confidence": f_conf, "probs": f_probs}

        # Body
        body_result = self.body_classifier.process_frame(frame)
        if body_result:
            b_state, b_conf, b_probs = body_result
            results["body"] = {"state": b_state, "confidence": b_conf, "probs": b_probs}

        if not results:
            return None

        # Late fusion (use available modalities)
        body_probs   = np.array(list(results["body"]["probs"].values()))   if "body"   in results else None
        face_probs   = np.array(list(results["face"]["probs"].values()))   if "face"   in results else None

        state, conf, probs = self.fusion.predict_with_missing(body_probs, face_probs, None)
        return self.post_processor.process(state, conf, probs)

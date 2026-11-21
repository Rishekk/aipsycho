import cv2
import numpy as np
from typing import List, Dict, Optional
from src.agents.face_agent import FaceDetector, FaceClassifier
from src.agents.body_agent import BodyClassifier
from src.fusion.late_fusion import LateFusionClassifier
from .modalities import plan_modalities
from .post_processor import PostProcessor
from src.utils.logger import get_logger

logger = get_logger("predictor")


class BatchPredictor:
    """
    Batch inference on pre-recorded video files (video only: face and body agents).

    Like ``RealtimePredictor`` it fails closed: an agent is built and run only if a
    trained checkpoint was supplied for it (body may instead be opted in to the
    experimental rule-based heuristics). Untrained models emit confident-looking noise,
    so they are never run. Construction raises ``FileNotFoundError`` for a checkpoint
    path that does not exist and ``ValueError`` if no agent could run.
    """

    SUPPORTED_FUSION = ("late",)

    def __init__(self,
                 face_model_path:   Optional[str] = None,
                 body_model_path:   Optional[str] = None,
                 fusion_strategy:   str = "late",
                 device:            str = None,
                 allow_rule_based_body: bool = False):

        if fusion_strategy not in self.SUPPORTED_FUSION:
            raise ValueError(
                f"Unsupported fusion_strategy {fusion_strategy!r}: BatchPredictor supports "
                f"{self.SUPPORTED_FUSION}. Attention/early fusion need embeddings from all "
                "three agents and a trained fusion checkpoint, which this class does not take."
            )

        # Speech is not used for video-only analysis, so no speech checkpoint is planned.
        plan = plan_modalities(face_model_path=face_model_path,
                               body_model_path=body_model_path,
                               speech_model_path=None,
                               allow_rule_based_body=allow_rule_based_body)
        self.plan = plan

        logger.info(f"Initializing BatchPredictor (modalities: {plan.active})...")
        self.face_detector   = FaceDetector() if plan.face else None
        self.face_classifier = FaceClassifier(model_path=face_model_path, device=device) \
            if plan.face else None
        # model_path=None selects the rule-based body agent, which the plan only allows
        # when the caller opted in.
        self.body_classifier = BodyClassifier(model_path=body_model_path, device=device) \
            if plan.body else None
        self.post_processor  = PostProcessor()
        self.fusion          = LateFusionClassifier(weights=[1.0, 1.0, 1.0])
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
            "modalities":      self.plan.active,
        }

    def _process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """Process one frame through the enabled face / body agents and fuse."""
        results = {}

        # Face (only if its checkpoint was supplied)
        if self.face_classifier is not None:
            face_crop = self.face_detector.detect_and_crop(frame)
            if face_crop is not None:
                f_state, f_conf, f_probs = self.face_classifier.predict(face_crop)
                results["face"] = {"state": f_state, "confidence": f_conf, "probs": f_probs}

        # Body (only if trained, or rule-based was opted in)
        body_result = self.body_classifier.process_frame(frame) \
            if self.body_classifier is not None else None
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

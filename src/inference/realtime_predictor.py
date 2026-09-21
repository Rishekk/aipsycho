import cv2
import numpy as np
import threading
import queue
import time
from typing import Optional, Dict, Callable
from src.agents.face_agent import FaceDetector, FaceClassifier, LandmarkExtractor, MicroExpressionDetector
from src.agents.body_agent import BodyClassifier
from src.agents.speech_agent import SpeechClassifier, AudioProcessor
from src.fusion.late_fusion import LateFusionClassifier
from src.fusion.ensemble import TemporalSmoother
from .post_processor import PostProcessor
from src.utils.logger import get_logger

logger = get_logger("realtime_predictor")


class RealtimePredictor:
    """
    Multi-threaded real-time psychological state detection.

    Thread 1 (video): Captures frames → runs face + body agents.
    Thread 2 (audio): Captures audio chunks → runs speech agent.
    Thread 3 (fusion): Reads from both queues → fuses → post-processes.

    Results are pushed to a result_queue that the UI can poll.
    """

    def __init__(self,
                 face_model_path:   Optional[str] = None,
                 body_model_path:   Optional[str] = None,
                 speech_model_path: Optional[str] = None,
                 device:            str = None,
                 camera_idx:        int = 0,
                 audio_duration:    float = 2.0):

        logger.info("Initializing RealtimePredictor...")

        # Agents
        self.face_detector  = FaceDetector()
        self.face_classifier = FaceClassifier(model_path=face_model_path, device=device)
        self.landmark_extractor = LandmarkExtractor()
        self.micro_detector = MicroExpressionDetector(window_size=10)
        self.body_classifier = BodyClassifier(model_path=body_model_path, device=device,
                                               sequence_length=30)
        self.speech_classifier = SpeechClassifier(model_path=speech_model_path, device=device)
        self.audio_processor = AudioProcessor(chunk_duration=audio_duration)

        # Fusion + post-processing
        self.fusion        = LateFusionClassifier(weights=[1.0, 1.2, 0.8])  # face slightly upweighted
        self.smoother      = TemporalSmoother(alpha=0.35)
        self.post_processor = PostProcessor(window_size=15, confidence_threshold=0.35)

        # Threading
        self.camera_idx    = camera_idx
        self.audio_duration = audio_duration
        self._video_queue: queue.Queue = queue.Queue(maxsize=5)
        self._audio_queue: queue.Queue = queue.Queue(maxsize=3)
        self.result_queue:  queue.Queue = queue.Queue(maxsize=30)

        self._running = False
        self._threads = []

        logger.info("RealtimePredictor ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Start all processing threads."""
        self._running = True
        self._threads = [
            threading.Thread(target=self._video_loop,  daemon=True, name="VideoThread"),
            threading.Thread(target=self._audio_loop,  daemon=True, name="AudioThread"),
            threading.Thread(target=self._fusion_loop, daemon=True, name="FusionThread"),
        ]
        for t in self._threads:
            t.start()
        logger.info("All threads started.")

    def stop(self):
        """Signal all threads to stop."""
        self._running = False
        for t in self._threads:
            t.join(timeout=3.0)
        logger.info("RealtimePredictor stopped.")

    def get_result(self, timeout: float = 0.05) -> Optional[Dict]:
        """Non-blocking poll for latest result. Returns None if nothing available."""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def reset_session(self):
        self.post_processor.reset_session()
        self.body_classifier.reset_buffer()
        self.micro_detector.reset()
        self.smoother.reset()

    # ------------------------------------------------------------------
    # Thread loops
    # ------------------------------------------------------------------

    def _video_loop(self):
        cap = cv2.VideoCapture(self.camera_idx)
        if not cap.isOpened():
            logger.error(f"Cannot open camera {self.camera_idx}")
            return

        logger.info("Video thread running.")
        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            video_data = self._process_video_frame(frame)
            if video_data:
                try:
                    self._video_queue.put_nowait((frame.copy(), video_data))
                except queue.Full:
                    pass  # drop stale frame

        cap.release()
        logger.info("Video thread stopped.")

    def _audio_loop(self):
        logger.info("Audio thread running.")
        while self._running:
            try:
                audio = self.audio_processor.record(duration=self.audio_duration)
                state, conf, probs = self.speech_classifier.predict_from_array(audio)
                try:
                    self._audio_queue.put_nowait({"state": state, "confidence": conf, "probs": probs})
                except queue.Full:
                    pass
            except Exception as e:
                logger.warning(f"Audio capture error: {e}")
                time.sleep(0.5)
        logger.info("Audio thread stopped.")

    def _fusion_loop(self):
        logger.info("Fusion thread running.")
        latest_audio: Optional[Dict] = None

        while self._running:
            try:
                frame, video_data = self._video_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Update latest audio (non-blocking)
            try:
                latest_audio = self._audio_queue.get_nowait()
            except queue.Empty:
                pass

            # Fuse modalities
            body_probs   = np.array(list(video_data["body"]["probs"].values()))   if video_data.get("body")   else None
            face_probs   = np.array(list(video_data["face"]["probs"].values()))   if video_data.get("face")   else None
            speech_probs = np.array(list(latest_audio["probs"].values()))         if latest_audio             else None

            state, conf, probs = self.fusion.predict_with_missing(body_probs, face_probs, speech_probs)

            # Temporal smoothing
            state, conf, probs = self.smoother.update(probs)

            # Post-process
            result = self.post_processor.process(state, conf, probs)
            result["frame_rgb"]   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result["micro_event"] = video_data.get("micro_event", False)
            result["landmarks"]   = video_data.get("landmarks")

            try:
                self.result_queue.put_nowait(result)
            except queue.Full:
                try:
                    self.result_queue.get_nowait()   # discard oldest
                    self.result_queue.put_nowait(result)
                except queue.Empty:
                    pass

        logger.info("Fusion thread stopped.")

    # ------------------------------------------------------------------
    # Frame processing helpers
    # ------------------------------------------------------------------

    def _process_video_frame(self, frame: np.ndarray) -> Dict:
        data = {}

        # --- Face ---
        face_crop = self.face_detector.detect_and_crop(frame)
        if face_crop is not None:
            f_state, f_conf, f_probs = self.face_classifier.predict(face_crop)
            data["face"] = {"state": f_state, "confidence": f_conf, "probs": f_probs}

        # --- Landmarks + micro-expressions ---
        landmarks = self.landmark_extractor.extract_landmarks(frame)
        if landmarks is not None:
            geo_feats = self.landmark_extractor.extract_features(landmarks)
            self.micro_detector.update(geo_feats)
            data["micro_event"] = self.micro_detector.detect().get("micro_event", False)
            data["landmarks"]   = landmarks

        # --- Body ---
        body_result = self.body_classifier.process_frame(frame)
        if body_result:
            b_state, b_conf, b_probs = body_result
            data["body"] = {"state": b_state, "confidence": b_conf, "probs": b_probs}

        return data

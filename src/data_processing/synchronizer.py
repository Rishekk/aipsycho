"""
Multimodal synchronizer: pairs video frames with audio segments
by timestamp so both modalities cover the same time window.
Used when building the multimodal (fusion) dataset from raw recordings.
"""
import cv2
import numpy as np
import librosa
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Iterator
from src.utils.logger import get_logger

logger = get_logger("synchronizer")


@dataclass
class SyncedSample:
    """One time-aligned multimodal sample."""
    frame_sequence: np.ndarray        # (T, H, W, 3) BGR frames
    audio_chunk:    np.ndarray        # (sr * duration,) float32
    timestamp_sec:  float
    label:          Optional[int] = None


class AudioVideoSynchronizer:
    """
    Extracts temporally aligned (video frame sequence, audio chunk) pairs
    from a video file that has an embedded audio track.

    Usage:
        sync = AudioVideoSynchronizer(seq_len=30, audio_duration=2.0)
        for sample in sync.extract("path/to/video.mp4", label=2):
            # sample.frame_sequence → (30, H, W, 3)
            # sample.audio_chunk    → float32 audio array
    """

    def __init__(self, seq_len: int = 30, audio_duration: float = 2.0,
                 sr: int = 16000, step_sec: float = 1.0):
        self.seq_len       = seq_len
        self.audio_duration = audio_duration
        self.sr            = sr
        self.step_sec      = step_sec   # slide window by this many seconds

    def extract(self, video_path: str,
                label: Optional[int] = None) -> Iterator[SyncedSample]:
        """
        Slide a window over the video and yield SyncedSample objects.

        Args:
            video_path: Path to video file.
            label:      Integer class label for this clip.

        Yields:
            SyncedSample at each window position.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)

        # Load full audio track
        try:
            audio_full, _ = librosa.load(video_path, sr=self.sr, mono=True)
        except Exception as e:
            logger.warning(f"Could not load audio from {video_path}: {e}")
            audio_full = np.zeros(int(self.sr * 60), dtype=np.float32)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_sec    = total_frames / fps

        # How many video frames cover the audio window?
        frames_per_window = max(self.seq_len, int(self.audio_duration * fps))
        step_frames = max(1, int(self.step_sec * fps))

        all_frames = self._read_all_frames(cap)
        cap.release()

        window_start = 0
        while window_start + frames_per_window <= len(all_frames):
            t_sec = window_start / fps

            # Sample seq_len frames evenly from the window
            indices = np.linspace(window_start,
                                  window_start + frames_per_window - 1,
                                  self.seq_len, dtype=int)
            frame_seq = np.stack([all_frames[i] for i in indices])  # (T, H, W, 3)

            # Corresponding audio slice
            audio_start = int(t_sec * self.sr)
            audio_end   = audio_start + int(self.audio_duration * self.sr)
            audio_chunk = audio_full[audio_start:audio_end]
            # Pad if needed
            if len(audio_chunk) < int(self.audio_duration * self.sr):
                audio_chunk = np.pad(audio_chunk,
                    (0, int(self.audio_duration * self.sr) - len(audio_chunk)))
            audio_chunk = self._normalize(audio_chunk)

            yield SyncedSample(
                frame_sequence=frame_seq,
                audio_chunk=audio_chunk,
                timestamp_sec=t_sec,
                label=label,
            )
            window_start += step_frames

    def _read_all_frames(self, cap: cv2.VideoCapture) -> List[np.ndarray]:
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        return frames

    @staticmethod
    def _normalize(audio: np.ndarray) -> np.ndarray:
        mx = np.max(np.abs(audio))
        return (audio / mx).astype(np.float32) if mx > 0 else audio.astype(np.float32)


class MultimodalDatasetBuilder:
    """
    Processes a directory of labelled video clips and builds a multimodal dataset
    of (frame_sequence, audio_chunk, label) triples.

    Expected directory layout:
        root/
            calm/        clip1.mp4  clip2.mp4 ...
            stressed/    ...
            anxious/     ...
            confident/   ...
            frustrated/  ...

    Saves extracted samples as .npz files in output_dir.
    """

    def __init__(self, seq_len: int = 30, audio_duration: float = 2.0,
                 sr: int = 16000, step_sec: float = 1.0):
        self.sync = AudioVideoSynchronizer(
            seq_len=seq_len, audio_duration=audio_duration,
            sr=sr, step_sec=step_sec,
        )

    def build(self, root: str, output_dir: str):
        from config.constants import STATE_TO_IDX
        os.makedirs(output_dir, exist_ok=True)
        count = 0

        for state_dir in sorted(os.listdir(root)):
            state_path = os.path.join(root, state_dir)
            if not os.path.isdir(state_path):
                continue
            label = STATE_TO_IDX.get(state_dir.lower())
            if label is None:
                logger.warning(f"Unknown state directory: {state_dir} — skipping")
                continue

            for fname in os.listdir(state_path):
                if not fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    continue
                video_path = os.path.join(state_path, fname)
                try:
                    for sample in self.sync.extract(video_path, label=label):
                        out_path = os.path.join(output_dir, f"sample_{count:06d}.npz")
                        np.savez_compressed(
                            out_path,
                            frames=sample.frame_sequence,
                            audio=sample.audio_chunk,
                            label=np.array(sample.label),
                            # Source clip, so training can split by clip (windows cut
                            # from one video must not land on both sides of a split).
                            source=np.array(os.path.relpath(video_path, root)),
                        )
                        count += 1
                except Exception as e:
                    logger.error(f"Failed to process {video_path}: {e}")

        logger.info(f"Built {count} multimodal samples → {output_dir}")
        return count

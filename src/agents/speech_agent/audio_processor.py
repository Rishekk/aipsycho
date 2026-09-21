import numpy as np
import librosa
import sounddevice as sd
from typing import Optional, Tuple
from src.utils.logger import get_logger

logger = get_logger("audio_processor")

DEFAULT_SR = 16000
DEFAULT_CHUNK_DURATION = 2.0   # seconds
DEFAULT_OVERLAP = 0.5          # seconds


class AudioProcessor:
    """
    Handles audio loading, recording, normalization, and chunking.
    Supports both file-based and real-time microphone input.
    """

    def __init__(self, sample_rate: int = DEFAULT_SR,
                 chunk_duration: float = DEFAULT_CHUNK_DURATION,
                 overlap: float = DEFAULT_OVERLAP):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.overlap_samples = int(overlap * sample_rate)
        self.step_samples = self.chunk_samples - self.overlap_samples
        logger.info(f"AudioProcessor initialized — sr={sample_rate}, chunk={chunk_duration}s")

    def load(self, path: str) -> np.ndarray:
        """
        Load audio file and resample to target sample rate.

        Args:
            path: Path to audio file (.wav, .mp3, etc.)

        Returns:
            Mono audio signal as float32 array.
        """
        audio, _ = librosa.load(path, sr=self.sample_rate, mono=True)
        return self.normalize(audio)

    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio

    def remove_silence(self, audio: np.ndarray, top_db: int = 30) -> np.ndarray:
        """Trim leading and trailing silence."""
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed

    def chunk_audio(self, audio: np.ndarray) -> list:
        """
        Split audio into overlapping chunks of fixed length.

        Args:
            audio: 1D audio signal.

        Returns:
            List of audio chunks, each of length chunk_samples.
        """
        chunks = []
        start = 0
        while start + self.chunk_samples <= len(audio):
            chunks.append(audio[start:start + self.chunk_samples])
            start += self.step_samples

        # Pad last chunk if remaining audio is significant
        remaining = len(audio) - start
        if remaining > self.chunk_samples // 2:
            chunk = np.zeros(self.chunk_samples, dtype=np.float32)
            chunk[:remaining] = audio[start:]
            chunks.append(chunk)

        return chunks

    def record(self, duration: float = None) -> np.ndarray:
        """
        Record audio from the default microphone.

        Args:
            duration: Recording duration in seconds. Uses chunk_duration if None.

        Returns:
            Recorded audio as float32 array.
        """
        duration = duration or self.chunk_duration
        logger.info(f"Recording {duration}s of audio...")
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        return self.normalize(audio.flatten())

    def preprocess(self, audio: np.ndarray) -> np.ndarray:
        """Full preprocessing: normalize + remove silence."""
        audio = self.normalize(audio)
        audio = self.remove_silence(audio)
        return audio

    def pad_or_trim(self, audio: np.ndarray) -> np.ndarray:
        """Pad or trim audio to exactly chunk_samples length."""
        if len(audio) >= self.chunk_samples:
            return audio[:self.chunk_samples]
        padded = np.zeros(self.chunk_samples, dtype=np.float32)
        padded[:len(audio)] = audio
        return padded

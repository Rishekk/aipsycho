import numpy as np
import librosa
from typing import Dict
from src.utils.logger import get_logger

logger = get_logger("speech_feature_extractor")

DEFAULT_SR       = 16000
DEFAULT_N_MFCC   = 40
DEFAULT_N_FFT    = 512
DEFAULT_HOP      = 256
DEFAULT_N_MELS   = 128


class SpeechFeatureExtractor:
    """
    Extracts a comprehensive set of acoustic features from an audio chunk:
    MFCCs, pitch, energy, ZCR, spectral features, and chroma.
    Returns both raw feature matrices and a flattened feature vector for ML input.
    """

    def __init__(self, sample_rate: int = DEFAULT_SR, n_mfcc: int = DEFAULT_N_MFCC,
                 n_fft: int = DEFAULT_N_FFT, hop_length: int = DEFAULT_HOP,
                 n_mels: int = DEFAULT_N_MELS):
        self.sr         = sample_rate
        self.n_mfcc     = n_mfcc
        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.n_mels     = n_mels
        logger.info("SpeechFeatureExtractor initialized.")

    def extract_all(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all acoustic features from an audio chunk.

        Args:
            audio: 1D float32 audio array (mono).

        Returns:
            Dict mapping feature name → numpy array.
        """
        features = {}

        # MFCCs (40 coefficients, mean + std per coeff → 80 values)
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=self.n_mfcc,
                                     n_fft=self.n_fft, hop_length=self.hop_length)
        features["mfcc"] = mfcc                             # (40, T)
        features["mfcc_delta"] = librosa.feature.delta(mfcc)
        features["mfcc_delta2"] = librosa.feature.delta(mfcc, order=2)

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=self.n_mels,
                                              n_fft=self.n_fft, hop_length=self.hop_length)
        features["mel_spec"] = librosa.power_to_db(mel, ref=np.max)   # (128, T)

        # Pitch (fundamental frequency)
        f0, _, _ = librosa.pyin(audio, fmin=librosa.note_to_hz("C2"),
                                 fmax=librosa.note_to_hz("C7"),
                                 sr=self.sr, hop_length=self.hop_length)
        f0 = np.nan_to_num(f0)
        features["pitch"] = f0                              # (T,)

        # RMS energy
        features["energy"] = librosa.feature.rms(
            y=audio, frame_length=self.n_fft, hop_length=self.hop_length
        )[0]                                                # (T,)

        # Zero crossing rate
        features["zcr"] = librosa.feature.zero_crossing_rate(
            audio, frame_length=self.n_fft, hop_length=self.hop_length
        )[0]                                                # (T,)

        # Spectral centroid
        features["spectral_centroid"] = librosa.feature.spectral_centroid(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]

        # Spectral bandwidth
        features["spectral_bandwidth"] = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]

        # Spectral rolloff
        features["spectral_rolloff"] = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]

        # Chroma (pitch class profiles)
        features["chroma"] = librosa.feature.chroma_stft(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )                                                   # (12, T)

        return features

    def get_feature_vector(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute statistical summaries (mean + std) of all features into a
        compact 1D vector suitable for ML model input.

        Args:
            audio: 1D audio array.

        Returns:
            1D float32 numpy array.
        """
        raw = self.extract_all(audio)
        parts = []

        for key in ["mfcc", "mfcc_delta", "mfcc_delta2", "chroma"]:
            mat = raw[key]
            parts.append(np.mean(mat, axis=1))
            parts.append(np.std(mat, axis=1))

        for key in ["pitch", "energy", "zcr",
                    "spectral_centroid", "spectral_bandwidth", "spectral_rolloff"]:
            vec = raw[key]
            parts.append(np.array([np.mean(vec), np.std(vec),
                                    np.min(vec), np.max(vec)]))

        return np.concatenate(parts).astype(np.float32)

    def get_mfcc_matrix(self, audio: np.ndarray) -> np.ndarray:
        """
        Return stacked MFCC + delta + delta2 matrix (120, T).
        Useful as 2D input to CNN models.

        Args:
            audio: 1D audio array.

        Returns:
            Array of shape (120, T).
        """
        raw = self.extract_all(audio)
        return np.vstack([raw["mfcc"], raw["mfcc_delta"], raw["mfcc_delta2"]])

    def get_scalar_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Return single scalar summary per feature for interpretability."""
        raw = self.extract_all(audio)
        return {
            "mean_pitch":        float(np.mean(raw["pitch"][raw["pitch"] > 0]) if np.any(raw["pitch"] > 0) else 0),
            "pitch_std":         float(np.std(raw["pitch"])),
            "mean_energy":       float(np.mean(raw["energy"])),
            "energy_std":        float(np.std(raw["energy"])),
            "mean_zcr":          float(np.mean(raw["zcr"])),
            "mean_spec_centroid":float(np.mean(raw["spectral_centroid"])),
            "mean_spec_bw":      float(np.mean(raw["spectral_bandwidth"])),
            "speaking_rate":     float(np.sum(raw["zcr"] > 0.1) / len(raw["zcr"])),
        }

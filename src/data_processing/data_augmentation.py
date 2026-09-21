import cv2
import numpy as np
import torch
import random
import librosa
from torchvision import transforms
from typing import Tuple


# ── Face augmentation ─────────────────────────────────────────────────────────

def get_face_train_transforms(size: int = 224) -> transforms.Compose:
    """Strong augmentation pipeline for face images during training."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size + 16, size + 16)),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])


def get_face_val_transforms(size: int = 224) -> transforms.Compose:
    """Deterministic transforms for validation/test."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# ── Audio augmentation ────────────────────────────────────────────────────────

class AudioAugmentor:
    """Stochastic augmentations for raw audio waveforms."""

    def __init__(self, sr: int = 16000):
        self.sr = sr

    def augment(self, audio: np.ndarray) -> np.ndarray:
        audio = self._add_noise(audio)
        audio = self._time_shift(audio)
        audio = self._pitch_shift(audio)
        audio = self._time_stretch(audio)
        return audio

    def _add_noise(self, audio: np.ndarray, prob: float = 0.4) -> np.ndarray:
        if random.random() < prob:
            noise_amp = random.uniform(0.001, 0.015) * np.max(np.abs(audio))
            audio = audio + noise_amp * np.random.randn(len(audio))
        return audio.astype(np.float32)

    def _time_shift(self, audio: np.ndarray, prob: float = 0.4,
                    max_shift: float = 0.15) -> np.ndarray:
        if random.random() < prob:
            shift = int(random.uniform(-max_shift, max_shift) * len(audio))
            audio = np.roll(audio, shift)
        return audio

    def _pitch_shift(self, audio: np.ndarray, prob: float = 0.3,
                     steps_range: Tuple[float, float] = (-2.0, 2.0)) -> np.ndarray:
        if random.random() < prob:
            steps = random.uniform(*steps_range)
            audio = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=steps)
        return audio.astype(np.float32)

    def _time_stretch(self, audio: np.ndarray, prob: float = 0.3,
                      rate_range: Tuple[float, float] = (0.85, 1.15)) -> np.ndarray:
        if random.random() < prob:
            rate  = random.uniform(*rate_range)
            audio = librosa.effects.time_stretch(audio, rate=rate)
            target = int(2.0 * self.sr)
            if len(audio) < target:
                audio = np.pad(audio, (0, target - len(audio)))
            else:
                audio = audio[:target]
        return audio.astype(np.float32)


# ── Pose augmentation ─────────────────────────────────────────────────────────

class PoseAugmentor:
    """Stochastic augmentations for pose landmark sequences."""

    def augment(self, sequence: np.ndarray) -> np.ndarray:
        """sequence: (T, 99)"""
        sequence = self._jitter(sequence)
        sequence = self._scale(sequence)
        sequence = self._flip(sequence)
        sequence = self._dropout(sequence)
        return sequence

    def _jitter(self, seq: np.ndarray, prob: float = 0.5,
                std: float = 0.005) -> np.ndarray:
        if random.random() < prob:
            seq = seq + np.random.normal(0, std, seq.shape).astype(np.float32)
        return seq

    def _scale(self, seq: np.ndarray, prob: float = 0.4,
               scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        if random.random() < prob:
            seq = seq * random.uniform(*scale_range)
        return seq.astype(np.float32)

    def _flip(self, seq: np.ndarray, prob: float = 0.5) -> np.ndarray:
        """Horizontal mirror (negate x coordinates)."""
        if random.random() < prob:
            seq = seq.copy()
            seq[:, 0::3] = -seq[:, 0::3]  # x values at indices 0,3,6,...
        return seq

    def _dropout(self, seq: np.ndarray, prob: float = 0.2,
                 drop_rate: float = 0.05) -> np.ndarray:
        """Randomly zero out a small fraction of landmark values."""
        if random.random() < prob:
            mask = np.random.rand(*seq.shape) > drop_rate
            seq  = seq * mask.astype(np.float32)
        return seq

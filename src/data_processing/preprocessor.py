import cv2
import numpy as np
import librosa
import torch
from typing import Tuple, Optional
from torchvision import transforms


# ── Image ─────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def preprocess_face(image_bgr: np.ndarray, size: int = 224) -> torch.Tensor:
    """BGR face crop → normalised (3, H, W) tensor."""
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    from PIL import Image as PILImage
    return t(PILImage.fromarray(img))


def preprocess_face_batch(images: list, size: int = 224) -> torch.Tensor:
    """List of BGR images → (N, 3, H, W) tensor."""
    return torch.stack([preprocess_face(img, size) for img in images])


# ── Audio ─────────────────────────────────────────────────────────────────────

def preprocess_audio(path: str, sr: int = 16000,
                     duration: float = 2.0) -> np.ndarray:
    """
    Load, resample, normalize and pad/trim an audio file to fixed length.

    Returns: float32 array of shape (sr * duration,)
    """
    audio, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    target = int(sr * duration)
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]
    # normalize
    mx = np.max(np.abs(audio))
    if mx > 0:
        audio = audio / mx
    return audio.astype(np.float32)


def audio_to_mfcc_tensor(audio: np.ndarray, sr: int = 16000,
                          n_mfcc: int = 40, n_fft: int = 512,
                          hop_length: int = 256) -> torch.Tensor:
    """
    Audio array → stacked MFCC + delta + delta2 tensor of shape (3, n_mfcc, T).
    """
    mfcc  = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc,
                                  n_fft=n_fft, hop_length=hop_length)
    d1    = librosa.feature.delta(mfcc)
    d2    = librosa.feature.delta(mfcc, order=2)
    stacked = np.stack([mfcc, d1, d2], axis=0).astype(np.float32)
    return torch.tensor(stacked)


# ── Pose ──────────────────────────────────────────────────────────────────────
# Defined once in pose_norm.py (NumPy-only) so training and inference share it.
from .pose_norm import normalize_pose, normalize_pose_sequence  # noqa: E402,F401

"""
Pose normalisation shared by training and inference.

The body LSTM must see the same input distribution when it is trained and when it
runs live. Normalisation is therefore defined once, here, per frame:

    1. centre the 33 landmarks on the hip midpoint
    2. scale so the torso (hip midpoint -> shoulder midpoint) has length 1

Per-frame normalisation does not depend on neighbouring frames, so a live pipeline
that sees one frame at a time computes exactly what the training pipeline computed
for a whole recorded sequence.

This module is NumPy-only so it can be imported (and tested) without PyTorch,
OpenCV or MediaPipe.
"""
import numpy as np

NUM_LANDMARKS = 33
POSE_DIMS = 3
FEATURE_DIM = NUM_LANDMARKS * POSE_DIMS  # 99

_LEFT_SHOULDER, _RIGHT_SHOULDER = 11, 12
_LEFT_HIP, _RIGHT_HIP = 23, 24


def normalize_pose(landmarks: np.ndarray) -> np.ndarray:
    """
    Centre on the hip midpoint and scale so the torso length is 1.

    Args:
        landmarks: (33, 3) array of x, y, z.

    Returns:
        (33, 3) float32 array. A degenerate pose with zero torso length is
        centred but not scaled (no division by zero).
    """
    lm = np.asarray(landmarks, dtype=np.float32).copy()
    if lm.shape != (NUM_LANDMARKS, POSE_DIMS):
        raise ValueError(f"expected landmarks of shape (33, 3), got {lm.shape}")

    lm -= (lm[_LEFT_HIP] + lm[_RIGHT_HIP]) / 2
    torso_len = np.linalg.norm((lm[_LEFT_SHOULDER] + lm[_RIGHT_SHOULDER]) / 2)
    if torso_len > 1e-6:
        lm /= torso_len
    return lm.astype(np.float32)


def normalize_pose_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Apply :func:`normalize_pose` to every frame of a sequence.

    Args:
        sequence: (T, 99) flattened landmarks, or (T, 33, 3).

    Returns:
        (T, 99) float32 array.
    """
    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim == 2 and seq.shape[1] == FEATURE_DIM:
        frames = seq.reshape(-1, NUM_LANDMARKS, POSE_DIMS)
    elif seq.ndim == 3 and seq.shape[1:] == (NUM_LANDMARKS, POSE_DIMS):
        frames = seq
    else:
        raise ValueError(
            f"expected a sequence of shape (T, 99) or (T, 33, 3), got {seq.shape}"
        )
    if len(frames) == 0:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    return np.stack([normalize_pose(f).reshape(FEATURE_DIM) for f in frames])

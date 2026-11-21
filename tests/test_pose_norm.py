"""Pose normalisation: the single definition shared by training and live inference."""
import numpy as np
import pytest

from src.data_processing.pose_norm import normalize_pose, normalize_pose_sequence


def _random_pose(rng):
    return rng.uniform(0.0, 1.0, size=(33, 3)).astype(np.float32)


def test_frame_is_hip_centred_with_unit_torso():
    lm = normalize_pose(_random_pose(np.random.default_rng(0)))
    assert np.allclose((lm[23] + lm[24]) / 2, 0.0, atol=1e-6)
    assert np.isclose(np.linalg.norm((lm[11] + lm[12]) / 2), 1.0, atol=1e-5)


def test_invariant_to_translation_and_uniform_scale():
    """Where the person stands and how far from the camera must not matter."""
    rng = np.random.default_rng(1)
    pose = _random_pose(rng)
    moved = pose * 2.5 + np.array([0.3, -0.2, 0.1], dtype=np.float32)
    assert np.allclose(normalize_pose(pose), normalize_pose(moved), atol=1e-5)


def test_idempotent():
    """Already-normalised data (e.g. from extract_body_poses.py) is left unchanged."""
    once = normalize_pose(_random_pose(np.random.default_rng(2)))
    assert np.allclose(normalize_pose(once), once, atol=1e-5)


def test_degenerate_pose_does_not_produce_nan():
    out = normalize_pose(np.zeros((33, 3), dtype=np.float32))
    assert np.all(np.isfinite(out))
    assert np.allclose(out, 0.0)


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        normalize_pose(np.zeros((10, 3)))


def test_training_path_matches_serving_path():
    """
    The regression test for the train/serve skew.

    Training normalises a stored sequence in one call; serving normalises one frame at a
    time as frames arrive. Both must produce identical model inputs.
    """
    rng = np.random.default_rng(3)
    frames = [_random_pose(rng) for _ in range(30)]

    train_input = normalize_pose_sequence(np.stack([f.reshape(99) for f in frames]))
    serve_input = np.stack([normalize_pose(f).reshape(99) for f in frames])

    assert train_input.shape == (30, 99)
    assert np.allclose(train_input, serve_input, atol=1e-6)


def test_each_frame_is_independent_of_other_frames():
    """Old behaviour used frame 0's hips/torso for the whole sequence; it must not."""
    rng = np.random.default_rng(4)
    frames = np.stack([_random_pose(rng).reshape(99) for _ in range(5)])
    base = normalize_pose_sequence(frames)

    changed = frames.copy()
    changed[0] = _random_pose(rng).reshape(99)  # replace frame 0 only
    after = normalize_pose_sequence(changed)

    assert np.allclose(base[1:], after[1:], atol=1e-6)
    assert not np.allclose(base[0], after[0])


def test_sequence_accepts_both_layouts():
    rng = np.random.default_rng(5)
    seq3d = np.stack([_random_pose(rng) for _ in range(4)])           # (T, 33, 3)
    assert np.allclose(normalize_pose_sequence(seq3d),
                       normalize_pose_sequence(seq3d.reshape(4, 99)))


def test_sequence_rejects_bad_shape_and_handles_empty():
    with pytest.raises(ValueError):
        normalize_pose_sequence(np.zeros((30, 98)))
    assert normalize_pose_sequence(np.zeros((0, 99))).shape == (0, 99)

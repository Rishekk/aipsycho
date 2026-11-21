"""
Dataset split helpers (pure Python, no torch/numpy imports so they are cheap to test).

Why this module exists
----------------------
Emotion/affect models are easy to over-estimate when the same *person* appears in
both the training and evaluation data: the model learns who the speaker/face is,
not the affect. All splits here are therefore made at the *subject* level where
subject identity is available (RAVDESS actors), and every dataset gets a separate
validation split (used for checkpoint selection / early stopping) and test split
(reported once, never used for model selection).
"""
import os
import random
from typing import Dict, List, Optional, Sequence, Set, TypeVar

T = TypeVar("T")

# ── RAVDESS: 24 actors, odd ids = male, even ids = female ────────────────────
# Held-out actors are gender balanced (2 male + 2 female per held-out split).
RAVDESS_TRAIN_ACTORS = tuple(range(1, 17))    # 1..16
RAVDESS_VAL_ACTORS   = tuple(range(17, 21))   # 17..20
RAVDESS_TEST_ACTORS  = tuple(range(21, 25))   # 21..24

_RAVDESS_SPLIT_ACTORS: Dict[str, Set[int]] = {
    "train": set(RAVDESS_TRAIN_ACTORS),
    "val":   set(RAVDESS_VAL_ACTORS),
    "test":  set(RAVDESS_TEST_ACTORS),
}


def ravdess_actor_id(path: str) -> Optional[int]:
    """
    Actor id from a RAVDESS filename.

    RAVDESS names are  modality-channel-emotion-intensity-statement-repetition-actor.wav,
    e.g. ``03-01-06-01-02-01-12.wav`` -> actor 12. Returns None if the name does not
    follow the convention.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("-")
    if len(parts) != 7:
        return None
    try:
        return int(parts[-1])
    except ValueError:
        return None


def split_ravdess_files(paths: Sequence[str], split: str) -> List[str]:
    """
    Speaker-independent split: every actor lives in exactly one of train/val/test.

    Files whose actor id cannot be parsed are dropped (never silently assigned to
    a split), so a malformed name can not leak across splits.
    """
    if split not in _RAVDESS_SPLIT_ACTORS:
        raise ValueError(f"split must be one of train/val/test, got {split!r}")
    wanted = _RAVDESS_SPLIT_ACTORS[split]
    out = []
    for p in sorted(paths):
        actor = ravdess_actor_id(p)
        if actor is not None and actor in wanted:
            out.append(p)
    return out


def actors_in(paths: Sequence[str]) -> Set[int]:
    """Set of parsed actor ids in a list of RAVDESS paths."""
    return {a for a in (ravdess_actor_id(p) for p in paths) if a is not None}


# ── Generic deterministic hold-out (for datasets with no subject ids) ────────

def holdout_split(items: Sequence[T], split: str,
                  val_fraction: float = 0.1, seed: int = 42) -> List[T]:
    """
    Deterministically carve a validation subset out of a training set.

    Use only when the dataset ships no official validation split and carries no
    subject identity (e.g. FER-2013 folder layout). ``split`` is "train" or "val";
    train and val are disjoint and together cover all of ``items``.
    """
    if split not in ("train", "val"):
        raise ValueError(f"holdout_split handles 'train'/'val' only, got {split!r}")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    ordered = sorted(items, key=str)
    rng = random.Random(seed)
    idx = list(range(len(ordered)))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(ordered) * val_fraction))) if ordered else 0
    val_idx = set(idx[:n_val])
    if split == "val":
        return [ordered[i] for i in sorted(val_idx)]
    return [ordered[i] for i in range(len(ordered)) if i not in val_idx]

"""
Dataset loaders for all three modalities.

Supported datasets:
  Face  : FER-2013, AffectNet, CK+, RAF-DB
  Speech: RAVDESS, IEMOCAP (audio-only)
  Body  : Custom CSV of pose sequences

Each dataset returns (input, label) where label is mapped to
the 5-class psychological state schema defined in config/constants.py
"""
import os
import csv
import glob
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from typing import List, Tuple, Optional, Dict

from .preprocessor import preprocess_audio, audio_to_mfcc_tensor, normalize_pose
from .data_augmentation import (get_face_train_transforms, get_face_val_transforms,
                                 AudioAugmentor, PoseAugmentor)
from config.constants import STATE_TO_IDX, NUM_CLASSES
from src.utils.logger import get_logger

logger = get_logger("dataset_loader")

# ── Emotion → psychological state mapping ─────────────────────────────────────

# FER-2013 / AffectNet 7-class → 5 psych states
EMOTION_TO_STATE: Dict[str, str] = {
    "angry":     "frustrated",
    "disgust":   "frustrated",
    "fear":      "anxious",
    "happy":     "confident",
    "sad":       "stressed",
    "surprise":  "anxious",
    "neutral":   "calm",
    # AffectNet extras
    "contempt":  "frustrated",
    "excitement":"confident",
    # RAVDESS / IEMOCAP labels
    "angry":     "frustrated",
    "calm":      "calm",
    "fearful":   "anxious",
    "happy":     "confident",
    "sad":       "stressed",
    "surprised": "anxious",
    "disgusted": "frustrated",
    "neutral":   "calm",
    # CK+ / RAF-DB
    "anger":     "frustrated",
    "contempt":  "frustrated",
    "happiness": "confident",
    "sadness":   "stressed",
    "surprise":  "anxious",
    "fear":      "anxious",
}


def map_to_psych_state(emotion: str) -> Optional[int]:
    """Map raw emotion label string to psychological state index, or None if unknown."""
    state = EMOTION_TO_STATE.get(emotion.lower().strip())
    if state is None:
        return None
    return STATE_TO_IDX[state]


# ── FER-2013 ──────────────────────────────────────────────────────────────────

class FER2013Dataset(Dataset):
    """
    FER-2013 dataset.

    Expected directory layout:
        data/raw/face/fer2013/
            train/
                angry/  disgust/  fear/  happy/  neutral/  sad/  surprise/
            test/
                ...

    Or the original CSV format:
        data/raw/face/fer2013/fer2013.csv  (columns: emotion, pixels, Usage)
    """

    def __init__(self, root: str, split: str = "train", transform=None):
        self.samples: List[Tuple[np.ndarray, int]] = []
        self.transform = transform or (
            get_face_train_transforms() if split == "train" else get_face_val_transforms()
        )

        csv_path = os.path.join(root, "fer2013.csv")
        if os.path.exists(csv_path):
            self._load_csv(csv_path, split)
        else:
            self._load_dir(os.path.join(root, split))

        logger.info(f"FER2013 [{split}]: {len(self.samples)} samples")

    def _load_csv(self, csv_path: str, split: str):
        usage_map = {"train": "Training", "val": "PublicTest", "test": "PrivateTest"}
        usage = usage_map.get(split, "Training")
        df = pd.read_csv(csv_path)
        df = df[df["Usage"] == usage]
        emotion_names = ["angry","disgust","fear","happy","sad","surprise","neutral"]
        for _, row in df.iterrows():
            label_idx = map_to_psych_state(emotion_names[int(row["emotion"])])
            if label_idx is None:
                continue
            pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
            img = np.stack([pixels, pixels, pixels], axis=-1)  # grayscale → 3ch
            self.samples.append((img, label_idx))

    def _load_dir(self, root: str):
        for emotion_dir in sorted(os.listdir(root)):
            label_idx = map_to_psych_state(emotion_dir)
            if label_idx is None:
                continue
            dir_path = os.path.join(root, emotion_dir)
            for img_path in glob.glob(os.path.join(dir_path, "*.jpg")) + \
                            glob.glob(os.path.join(dir_path, "*.png")):
                img = np.array(Image.open(img_path).convert("RGB"))
                self.samples.append((img, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        return self.transform(img), torch.tensor(label, dtype=torch.long)


# ── RAF-DB ────────────────────────────────────────────────────────────────────

class RAFDBDataset(Dataset):
    """
    RAF-DB dataset.

    Expected layout:
        data/raw/face/rafdb/
            train/   (subdirs per emotion: 1_angry, 2_disgust, ...)
            test/
        OR flat with label file:
            EmoLabel/list_patition_label.txt
            Image/aligned/  (*.jpg files)
    """

    RAF_LABELS = {1:"surprise", 2:"fear", 3:"disgust",
                  4:"happy", 5:"sad", 6:"angry", 7:"neutral"}

    def __init__(self, root: str, split: str = "train", transform=None):
        self.samples: List[Tuple[str, int]] = []
        self.transform = transform or (
            get_face_train_transforms() if split == "train" else get_face_val_transforms()
        )

        label_file = os.path.join(root, "EmoLabel", "list_patition_label.txt")
        img_dir    = os.path.join(root, "Image", "aligned")

        if os.path.exists(label_file) and os.path.isdir(img_dir):
            prefix = "train" if split == "train" else "test"
            with open(label_file) as f:
                for line in f:
                    fname, label_id = line.strip().split()
                    if not fname.startswith(prefix):
                        continue
                    emotion  = self.RAF_LABELS.get(int(label_id))
                    psych_id = map_to_psych_state(emotion) if emotion else None
                    if psych_id is None:
                        continue
                    self.samples.append((os.path.join(img_dir, fname), psych_id))
        else:
            # directory layout fallback
            split_dir = os.path.join(root, split)
            if os.path.isdir(split_dir):
                for emotion_dir in os.listdir(split_dir):
                    psych_id = map_to_psych_state(emotion_dir.split("_")[-1])
                    if psych_id is None:
                        continue
                    for p in glob.glob(os.path.join(split_dir, emotion_dir, "*.jpg")):
                        self.samples.append((p, psych_id))

        logger.info(f"RAF-DB [{split}]: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert("RGB"))
        return self.transform(img), torch.tensor(label, dtype=torch.long)


# ── RAVDESS (speech) ──────────────────────────────────────────────────────────

class RAVDESSDataset(Dataset):
    """
    RAVDESS speech dataset.

    Expected layout:
        data/raw/speech/ravdess/
            Actor_01/  Actor_02/  ...
                03-01-01-01-01-01-01.wav   (RAVDESS filename convention)

    Emotion code from filename: 01=neutral,02=calm,03=happy,04=sad,
                                05=angry,06=fearful,07=disgust,08=surprised
    """

    RAVDESS_EMOTIONS = {
        "01": "neutral", "02": "calm",    "03": "happy",
        "04": "sad",     "05": "angry",   "06": "fearful",
        "07": "disgust", "08": "surprised",
    }

    def __init__(self, root: str, split: str = "train",
                 sr: int = 16000, duration: float = 2.0,
                 augment: bool = False):
        self.sr       = sr
        self.duration = duration
        self.augmentor = AudioAugmentor(sr=sr) if augment else None
        self.samples: List[Tuple[str, int]] = []

        all_files = sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True))
        random.seed(42)
        random.shuffle(all_files)

        split_idx = int(len(all_files) * 0.8)
        files = all_files[:split_idx] if split == "train" else all_files[split_idx:]

        for path in files:
            fname     = os.path.basename(path)
            parts     = fname.replace(".wav", "").split("-")
            if len(parts) < 3:
                continue
            emotion   = self.RAVDESS_EMOTIONS.get(parts[2])
            psych_id  = map_to_psych_state(emotion) if emotion else None
            if psych_id is None:
                continue
            self.samples.append((path, psych_id))

        logger.info(f"RAVDESS [{split}]: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        audio = preprocess_audio(path, sr=self.sr, duration=self.duration)
        if self.augmentor:
            audio = self.augmentor.augment(audio)
        tensor = audio_to_mfcc_tensor(audio, sr=self.sr)
        return tensor, torch.tensor(label, dtype=torch.long)


# ── Pose sequence dataset (custom CSV) ────────────────────────────────────────

class PoseSequenceDataset(Dataset):
    """
    Generic pose sequence dataset loaded from a CSV file.

    CSV format:
        label, f0_x0, f0_y0, f0_z0, ..., f29_x32, f29_y32, f29_z32
        (label=psych state string, followed by 30 frames × 33 landmarks × 3 = 2970 values)

    Or a directory of .npy files named  <state>_<id>.npy  of shape (seq_len, 99).
    """

    def __init__(self, root: str, split: str = "train",
                 sequence_length: int = 30, augment: bool = False):
        self.seq_len  = sequence_length
        self.augmentor = PoseAugmentor() if augment else None
        self.samples: List[Tuple[np.ndarray, int]] = []

        csv_path = os.path.join(root, f"{split}.csv")
        npy_dir  = os.path.join(root, split)

        if os.path.exists(csv_path):
            self._load_csv(csv_path)
        elif os.path.isdir(npy_dir):
            self._load_npy_dir(npy_dir)
        else:
            logger.warning(f"PoseSequenceDataset: no data found at {root}")

        logger.info(f"PoseSequence [{split}]: {len(self.samples)} samples")

    def _load_csv(self, path: str):
        df = pd.read_csv(path, header=None)
        for _, row in df.iterrows():
            label   = map_to_psych_state(str(row.iloc[0]))
            if label is None:
                continue
            values  = row.iloc[1:].values.astype(np.float32)
            seq_len = self.seq_len
            feat_dim = 99
            expected = seq_len * feat_dim
            if len(values) < expected:
                values = np.pad(values, (0, expected - len(values)))
            sequence = values[:expected].reshape(seq_len, feat_dim)
            self.samples.append((sequence, label))

    def _load_npy_dir(self, npy_dir: str):
        for npy_path in glob.glob(os.path.join(npy_dir, "*.npy")):
            fname    = os.path.basename(npy_path)
            state    = fname.split("_")[0]
            label    = map_to_psych_state(state)
            if label is None:
                continue
            sequence = np.load(npy_path).astype(np.float32)
            if sequence.shape != (self.seq_len, 99):
                continue
            self.samples.append((sequence, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        seq = normalize_pose(seq.reshape(-1, 3)).reshape(self.seq_len, 99) \
              if seq.ndim == 2 else seq
        if self.augmentor:
            seq = self.augmentor.augment(seq)
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# ── Balanced sampler helper ────────────────────────────────────────────────────

def make_weighted_sampler(dataset: Dataset) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler to handle class imbalance.
    Reads labels directly from dataset.samples to avoid loading images.
    """
    if hasattr(dataset, "samples"):
        labels = [s[1] for s in dataset.samples]
    else:
        labels = [int(dataset[i][1]) for i in range(len(dataset))]
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    weights      = 1.0 / (class_counts + 1e-6)
    sample_weights = torch.tensor([weights[l] for l in labels], dtype=torch.float32)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                 replacement=True)


# ── Convenience DataLoader factories ──────────────────────────────────────────

def _pin_memory() -> bool:
    """pin_memory only works on CUDA, not MPS or CPU."""
    import torch
    return torch.cuda.is_available()


def get_face_loaders(root: str, dataset: str = "fer2013",
                     batch_size: int = 32, num_workers: int = 4):
    """Returns (train_loader, val_loader) for a face dataset."""
    DS = FER2013Dataset if dataset == "fer2013" else RAFDBDataset
    train_ds = DS(root, split="train")
    val_ds   = DS(root, split="test")
    sampler  = make_weighted_sampler(train_ds)
    pin = _pin_memory()
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers,
                              pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=pin)
    return train_loader, val_loader


def get_speech_loaders(root: str, batch_size: int = 32, num_workers: int = 4):
    """Returns (train_loader, val_loader) for RAVDESS."""
    train_ds = RAVDESSDataset(root, split="train", augment=True)
    val_ds   = RAVDESSDataset(root, split="val",   augment=False)
    sampler  = make_weighted_sampler(train_ds)
    pin = _pin_memory()
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers,
                              pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=pin)
    return train_loader, val_loader


def get_pose_loaders(root: str, batch_size: int = 32, num_workers: int = 4):
    """Returns (train_loader, val_loader) for pose sequences."""
    train_ds = PoseSequenceDataset(root, split="train", augment=True)
    val_ds   = PoseSequenceDataset(root, split="val",   augment=False)
    sampler  = make_weighted_sampler(train_ds)
    pin = _pin_memory()
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers,
                              pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=pin)
    return train_loader, val_loader

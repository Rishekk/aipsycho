"""
Extract body pose sequences from video files and save as .npy files
for training the BodyLSTM model.

Supports two input formats:
  1. RAVDESS video files  — automatically parses emotion label from filename
     (download from https://zenodo.org/record/1188976, take the
      "Video only" or "Video + Audio" zip files for speech/song)
  2. Generic labelled folders — subdirectory name is the state label
     e.g.  data/raw/body_videos/calm/clip1.mp4
                                stressed/clip2.mp4

Usage — RAVDESS video:
    python scripts/extract_body_poses.py \
        --source  data/raw/body_videos/ravdess_video \
        --out-dir data/raw/body \
        --format  ravdess

Usage — generic labelled folders:
    python scripts/extract_body_poses.py \
        --source  data/raw/body_videos \
        --out-dir data/raw/body \
        --format  folders

After running, train the body agent:
    python scripts/train_body_agent.py --data-root data/raw/body
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import glob
import random
import shutil
import numpy as np
import cv2
from tqdm import tqdm

from src.agents.body_agent.pose_estimator import PoseEstimator
from src.data_processing.pose_norm import normalize_pose
from src.data_processing.dataset_loader import EMOTION_TO_STATE
from config.constants import STATE_TO_IDX, PSYCHOLOGICAL_STATES
from src.utils.logger import get_logger

logger = get_logger("extract_body_poses")

# ── RAVDESS emotion code → emotion name ───────────────────────────────────────
RAVDESS_EMOTIONS = {
    "01": "neutral",   "02": "calm",    "03": "happy",
    "04": "sad",       "05": "angry",   "06": "fearful",
    "07": "disgust",   "08": "surprised",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _label_from_ravdess(filepath: str):
    """Parse the psychological state label from a RAVDESS filename.
    Returns state string or None if unmappable.
    Filename format: MM-VV-EE-II-SS-RR-AA.mp4
                     (modality, vocal_channel, emotion, intensity, statement, repetition, actor)
    """
    fname  = os.path.basename(filepath)
    parts  = os.path.splitext(fname)[0].split("-")
    if len(parts) < 3:
        return None
    emotion_code = parts[2]
    emotion_name = RAVDESS_EMOTIONS.get(emotion_code)
    if emotion_name is None:
        return None
    return EMOTION_TO_STATE.get(emotion_name)


def _label_from_folder(filepath: str):
    """Use the parent directory name as the state label."""
    folder = os.path.basename(os.path.dirname(filepath))
    return EMOTION_TO_STATE.get(folder.lower()) or (
        folder.lower() if folder.lower() in PSYCHOLOGICAL_STATES else None
    )


def _extract_poses_from_video(
    video_path: str,
    pose_estimator: PoseEstimator,
    seq_len: int = 30,
    stride: int = 15,
    skip_frames: int = 1,
) -> list:
    """
    Extract overlapping windows of pose sequences from a video file.

    Args:
        video_path:  Path to the video file.
        pose_estimator: Initialised PoseEstimator.
        seq_len:     Number of frames per sequence window.
        stride:      Step between consecutive windows.
        skip_frames: Process every Nth frame (1 = every frame).

    Returns:
        List of (seq_len, 99) float32 arrays.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return []

    all_landmarks = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip_frames == 0:
            lm = pose_estimator.extract_pose(frame)
            if lm is not None:
                all_landmarks.append(normalize_pose(lm).flatten())  # (99,)
        frame_idx += 1
    cap.release()

    if len(all_landmarks) < seq_len:
        # Pad with the last valid frame if the clip is too short
        pad = seq_len - len(all_landmarks)
        if all_landmarks:
            all_landmarks += [all_landmarks[-1]] * pad
        else:
            return []

    sequences = []
    for start in range(0, len(all_landmarks) - seq_len + 1, stride):
        seq = np.stack(all_landmarks[start: start + seq_len], axis=0)  # (seq_len, 99)
        sequences.append(seq.astype(np.float32))

    return sequences


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source",  required=True,
                   help="Root directory containing video files (or RAVDESS actor dirs).")
    p.add_argument("--out-dir", default="data/raw/body",
                   help="Output directory for .npy pose sequences.")
    p.add_argument("--format",  default="folders",
                   choices=["ravdess", "folders"],
                   help="'ravdess' — parse label from filename; "
                        "'folders' — subdirectory name is the label.")
    p.add_argument("--seq-len",     type=int, default=30,  help="Frames per sequence.")
    p.add_argument("--stride",      type=int, default=15,  help="Stride between windows.")
    p.add_argument("--skip-frames", type=int, default=1,   help="Process every N-th frame.")
    p.add_argument("--val-split",   type=float, default=0.2,
                   help="Fraction of sequences reserved for validation.")
    p.add_argument("--extensions",  default="mp4,avi,mov,mkv",
                   help="Comma-separated list of video extensions.")
    return p.parse_args()


def main():
    args = parse_args()

    extensions = [f".{e.strip().lstrip('.')}" for e in args.extensions.split(",")]
    label_fn = _label_from_ravdess if args.format == "ravdess" else _label_from_folder

    # Gather all video files
    video_files = []
    for ext in extensions:
        video_files += glob.glob(
            os.path.join(args.source, "**", f"*{ext}"), recursive=True
        )
    video_files = sorted(set(video_files))
    logger.info(f"Found {len(video_files)} video file(s) under {args.source}")

    if not video_files:
        logger.error("No video files found. Check --source and --extensions.")
        sys.exit(1)

    # Prepare output dirs
    train_dir = os.path.join(args.out_dir, "train")
    val_dir   = os.path.join(args.out_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir,   exist_ok=True)

    pose_estimator = PoseEstimator()
    counters = {s: 0 for s in PSYCHOLOGICAL_STATES}
    skipped  = 0
    seq_id   = 0

    for vpath in tqdm(video_files, desc="Processing videos"):
        state = label_fn(vpath)
        if state is None:
            logger.debug(f"Skipping (no label): {vpath}")
            skipped += 1
            continue

        sequences = _extract_poses_from_video(
            vpath, pose_estimator,
            seq_len=args.seq_len,
            stride=args.stride,
            skip_frames=args.skip_frames,
        )

        if not sequences:
            logger.debug(f"No pose detected: {vpath}")
            skipped += 1
            continue

        for seq in sequences:
            # Random train/val split
            split = "val" if random.random() < args.val_split else "train"
            out_dir = val_dir if split == "val" else train_dir
            fname   = f"{state}_{seq_id:06d}.npy"
            np.save(os.path.join(out_dir, fname), seq)
            counters[state] += 1
            seq_id += 1

    pose_estimator.close()

    total = sum(counters.values())
    logger.info(f"\nDone! Extracted {total} pose sequences ({skipped} videos skipped).")
    logger.info(f"Output: {args.out_dir}/train/  and  {args.out_dir}/val/")
    logger.info("State breakdown:")
    for state, count in counters.items():
        logger.info(f"  {state:12s}: {count}")

    logger.info(
        f"\nNext step: train the body agent with\n"
        f"  python scripts/train_body_agent.py --data-root {args.out_dir}"
    )


if __name__ == "__main__":
    main()

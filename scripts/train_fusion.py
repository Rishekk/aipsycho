"""
Train the Fusion model end-to-end.

Requires pre-trained agent checkpoints. Extracts embeddings from each agent,
then trains the fusion model on top.

Usage:
    python scripts/train_fusion.py \\
        --face-model   models/face_agent/face_agent_best.pth \\
        --body-model   models/body_agent/body_lstm_best.pth  \\
        --speech-model models/speech_agent/speech_cnn_lstm_best.pth \\
        --data-root    data/processed/multimodal \\
        --strategy     attention
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.agents.face_agent.emotion_cnn import EmotionCNN, FacePreprocessor
from src.agents.body_agent.temporal_model import BodyLSTM
from src.agents.speech_agent.speech_model import SpeechCNNLSTM
from src.fusion.attention_fusion import AttentionFusionModel
from src.fusion.early_fusion import EarlyFusionModel
from src.training.trainer import Trainer
from src.training.losses import LabelSmoothingLoss
from src.utils.model_utils import get_device, load_checkpoint
from src.utils.logger import get_logger

logger = get_logger("train_fusion")


# ── Embedding dataset ─────────────────────────────────────────────────────────

class EmbeddingDataset(Dataset):
    """
    Pre-computes and caches agent embeddings from .npz multimodal samples.

    Each .npz file contains:
        frames : (T, H, W, 3)  BGR video frames
        audio  : (sr*duration,) float32 audio
        label  : int

    This dataset extracts embeddings offline (much faster than re-running agents
    every epoch).
    """

    def __init__(self, npz_dir: str,
                 face_model: nn.Module, body_model: nn.Module, speech_model: nn.Module,
                 device: torch.device):
        self.samples = []  # [(body_emb, face_emb, speech_emb, label)]

        face_prep = FacePreprocessor()
        files = sorted(f for f in os.listdir(npz_dir) if f.endswith(".npz"))
        logger.info(f"Extracting embeddings from {len(files)} samples...")

        face_model.eval()
        body_model.eval()
        speech_model.eval()

        from src.data_processing.preprocessor import audio_to_mfcc_tensor, normalize_pose

        from src.agents.body_agent.pose_estimator import PoseEstimator
        pose_estimator = PoseEstimator()

        for fname in files:
            data  = np.load(os.path.join(npz_dir, fname))
            frames = data["frames"]   # (T, H, W, 3)
            audio  = data["audio"]    # float32 array
            label  = int(data["label"])

            with torch.no_grad():
                # Face: use middle frame
                mid_frame = frames[len(frames) // 2]
                face_t = face_prep.preprocess(mid_frame).to(device)
                face_emb = face_model.get_features(face_t).squeeze(0).cpu()

                # Body: extract pose landmarks from each frame, build sequence
                seq = np.zeros((30, 99), dtype=np.float32)
                for i, frame in enumerate(frames[:30]):
                    landmarks = pose_estimator.extract_pose(frame)
                    if landmarks is not None:
                        seq[i] = normalize_pose(landmarks).flatten()
                seq_t = torch.tensor(seq).unsqueeze(0).to(device)
                body_emb = body_model.get_features(seq_t).squeeze(0).cpu()

                # Speech
                audio_t = audio_to_mfcc_tensor(audio).unsqueeze(0).to(device)
                speech_emb = speech_model.get_features(audio_t).squeeze(0).cpu()

            self.samples.append((body_emb, face_emb, speech_emb, label))

        pose_estimator.close()

        logger.info(f"Embedding dataset ready: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        body_emb, face_emb, speech_emb, label = self.samples[idx]
        return (body_emb, face_emb, speech_emb), torch.tensor(label, dtype=torch.long)


# ── Fusion wrapper (makes it compatible with Trainer) ─────────────────────────

class FusionWrapper(nn.Module):
    """Wraps AttentionFusionModel / EarlyFusionModel for use with Trainer."""

    def __init__(self, fusion_model: nn.Module):
        super().__init__()
        self.fusion = fusion_model

    def forward(self, body_emb, face_emb, speech_emb):
        return self.fusion(body_emb, face_emb, speech_emb)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",    required=True,  help="Dir with .npz multimodal files")
    p.add_argument("--face-model",   default=None)
    p.add_argument("--body-model",   default=None)
    p.add_argument("--speech-model", default=None)
    p.add_argument("--strategy",     default="attention", choices=["attention", "early"])
    p.add_argument("--hidden-dim",   type=int,   default=256)
    p.add_argument("--num-heads",    type=int,   default=4)
    p.add_argument("--dropout",      type=float, default=0.3)
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch-size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--scheduler",    default="cosine", choices=["cosine","plateau","none"])
    p.add_argument("--workers",      type=int,   default=0)
    p.add_argument("--checkpoint-dir", default="models/fusion")
    p.add_argument("--patience",     type=int,   default=10)
    p.add_argument("--val-split",    type=float, default=0.2)
    return p.parse_args()


def main():
    args   = parse_args()
    device = get_device()
    logger.info(f"Device: {device} | Strategy: {args.strategy}")

    # ── Load agent backbones ───────────────────────────────────────────────────
    face_model   = EmotionCNN(num_classes=5, pretrained=False).to(device)
    body_model   = BodyLSTM(input_size=99, hidden_size=256, num_layers=2,
                             num_classes=5).to(device)
    speech_model = SpeechCNNLSTM(n_mfcc=40, num_classes=5).to(device)

    if args.face_model:
        load_checkpoint(face_model, None, args.face_model, str(device))
        logger.info(f"Loaded face model: {args.face_model}")
    if args.body_model:
        load_checkpoint(body_model, None, args.body_model, str(device))
        logger.info(f"Loaded body model: {args.body_model}")
    if args.speech_model:
        load_checkpoint(speech_model, None, args.speech_model, str(device))
        logger.info(f"Loaded speech model: {args.speech_model}")

    # ── Build embedding dataset ────────────────────────────────────────────────
    full_ds = EmbeddingDataset(args.data_root, face_model, body_model,
                                speech_model, device)

    n_val    = max(1, int(len(full_ds) * args.val_split))
    n_train  = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])

    # Sample embeddings to get dims
    (body_emb, face_emb, speech_emb), _ = full_ds[0]
    body_dim, face_dim, speech_dim = body_emb.shape[0], face_emb.shape[0], speech_emb.shape[0]
    logger.info(f"Embedding dims — body: {body_dim}, face: {face_dim}, speech: {speech_dim}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                               shuffle=False, num_workers=args.workers)

    # ── Fusion model ──────────────────────────────────────────────────────────
    if args.strategy == "attention":
        fusion_core = AttentionFusionModel(
            body_dim=body_dim, face_dim=face_dim, speech_dim=speech_dim,
            hidden_dim=args.hidden_dim, num_heads=args.num_heads,
            num_classes=5, dropout=args.dropout,
        )
    else:
        fusion_core = EarlyFusionModel(
            body_dim=body_dim, face_dim=face_dim, speech_dim=speech_dim,
            hidden_dim=args.hidden_dim * 2, num_classes=5, dropout=args.dropout,
        )

    model     = FusionWrapper(fusion_core)
    criterion = LabelSmoothingLoss(num_classes=5, smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    logger.info(f"Fusion params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Train ─────────────────────────────────────────────────────────────────
    scheduler_type = None if args.scheduler == "none" else args.scheduler
    trainer = Trainer(
        model               = model,
        train_loader        = train_loader,
        val_loader          = val_loader,
        criterion           = criterion,
        optimizer           = optimizer,
        device              = device,
        checkpoint_dir      = args.checkpoint_dir,
        model_name          = f"fusion_{args.strategy}",
        epochs              = args.epochs,
        scheduler_type      = scheduler_type,
        early_stop_patience = args.patience,
        save_every          = 5,
    )

    history = trainer.train()

    best_f1  = max(history["val_f1"])
    best_acc = max(history["val_acc"])
    logger.info(f"\nBest Val F1: {best_f1:.4f} | Best Val Acc: {best_acc:.4f}")
    logger.info(f"Checkpoint: {args.checkpoint_dir}/fusion_{args.strategy}_best.pth")


if __name__ == "__main__":
    main()

"""
Train the Speech Agent (SpeechCNNLSTM) on RAVDESS or similar.

Usage:
    python scripts/train_speech_agent.py --data-root data/raw/speech/ravdess
    python scripts/train_speech_agent.py --data-root data/raw/speech/ravdess --model mlp
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
import torch.nn as nn

from src.agents.speech_agent.speech_model import SpeechCNNLSTM, SpeechMLP
from src.data_processing.dataset_loader import get_speech_loaders
from src.training.trainer import Trainer
from src.training.losses import FocalLoss, LabelSmoothingLoss
from src.utils.model_utils import get_device
from src.utils.logger import get_logger

logger = get_logger("train_speech")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",    required=True)
    p.add_argument("--model",        default="cnn_lstm", choices=["cnn_lstm", "mlp"])
    p.add_argument("--n-mfcc",       type=int,   default=40)
    p.add_argument("--lstm-hidden",  type=int,   default=256)
    p.add_argument("--lstm-layers",  type=int,   default=2)
    p.add_argument("--dropout",      type=float, default=0.3)
    p.add_argument("--epochs",       type=int,   default=60)
    p.add_argument("--batch-size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--scheduler",    default="cosine", choices=["cosine","plateau","none"])
    p.add_argument("--workers",      type=int,   default=4)
    p.add_argument("--checkpoint-dir", default="models/speech_agent")
    p.add_argument("--patience",     type=int,   default=12)
    p.add_argument("--loss",         default="focal", choices=["focal","label_smooth","ce"])
    return p.parse_args()


def main():
    args   = parse_args()
    device = get_device()
    logger.info(f"Device: {device} | Args: {vars(args)}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = get_speech_loaders(
        root        = args.data_root,
        batch_size  = args.batch_size,
        num_workers = args.workers,
    )
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.model == "cnn_lstm":
        model = SpeechCNNLSTM(
            n_mfcc      = args.n_mfcc,
            num_classes = 5,
            lstm_hidden = args.lstm_hidden,
            lstm_layers = args.lstm_layers,
            dropout     = args.dropout,
        )
    else:
        # MLP on pre-computed feature vectors
        # Feature vector size: (40*3 * 2) + (6 * 4) = 240 + 24 = 264
        # Adjust input_size to match SpeechFeatureExtractor.get_feature_vector() output
        model = SpeechMLP(input_size=312, num_classes=5, dropout=args.dropout)

    logger.info(f"Model: {args.model} | "
                f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Loss + Optimiser ──────────────────────────────────────────────────────
    if args.loss == "focal":
        criterion = FocalLoss(gamma=2.0)
    elif args.loss == "label_smooth":
        criterion = LabelSmoothingLoss(num_classes=5, smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

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
        model_name          = f"speech_{args.model}",
        epochs              = args.epochs,
        scheduler_type      = scheduler_type,
        early_stop_patience = args.patience,
        save_every          = 5,
    )

    history = trainer.train()

    best_f1  = max(history["val_f1"])
    best_acc = max(history["val_acc"])
    logger.info(f"\nBest Val F1: {best_f1:.4f} | Best Val Acc: {best_acc:.4f}")
    logger.info(f"Checkpoint: {args.checkpoint_dir}/speech_{args.model}_best.pth")


if __name__ == "__main__":
    main()

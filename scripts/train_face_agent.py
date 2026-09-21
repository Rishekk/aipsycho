"""
Train the Face Agent (EmotionCNN) on FER-2013 or RAF-DB.

Usage:
    python scripts/train_face_agent.py --dataset fer2013 --data-root data/raw/face/fer2013
    python scripts/train_face_agent.py --dataset rafdb   --data-root data/raw/face/rafdb
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
import torch.nn as nn

from src.agents.face_agent.emotion_cnn import EmotionCNN
from src.data_processing.dataset_loader import (
    FER2013Dataset, RAFDBDataset, get_face_loaders, make_weighted_sampler,
)
from src.training.trainer import Trainer
from src.training.losses import FocalLoss, LabelSmoothingLoss
from src.utils.model_utils import get_device
from src.utils.logger import get_logger

logger = get_logger("train_face")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",     default="fer2013", choices=["fer2013", "rafdb"])
    p.add_argument("--data-root",   required=True,     help="Path to dataset root")
    p.add_argument("--epochs",      type=int,  default=50)
    p.add_argument("--batch-size",  type=int,  default=32)
    p.add_argument("--lr",          type=float,default=1e-4)
    p.add_argument("--dropout",     type=float,default=0.3)
    p.add_argument("--freeze-layers",type=int, default=6)
    p.add_argument("--scheduler",   default="cosine", choices=["cosine","plateau","none"])
    p.add_argument("--workers",     type=int,  default=4)
    p.add_argument("--checkpoint-dir", default="models/face_agent")
    p.add_argument("--patience",    type=int,  default=10)
    p.add_argument("--loss",        default="focal", choices=["focal","label_smooth","ce"])
    return p.parse_args()


def main():
    args   = parse_args()
    device = get_device()
    logger.info(f"Device: {device}")
    logger.info(f"Args: {vars(args)}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = get_face_loaders(
        root       = args.data_root,
        dataset    = args.dataset,
        batch_size = args.batch_size,
        num_workers= args.workers,
    )
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EmotionCNN(
        num_classes   = 5,
        pretrained    = True,
        dropout       = args.dropout,
        freeze_layers = args.freeze_layers,
    )
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters — total: {total_params:,} | trainable: {trainable_params:,}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    if args.loss == "focal":
        criterion = FocalLoss(gamma=2.0)
    elif args.loss == "label_smooth":
        criterion = LabelSmoothingLoss(num_classes=5, smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss()

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # Use lower LR for frozen backbone, higher for the new head
    backbone_params = [p for n, p in model.named_parameters()
                       if "classifier" not in n and p.requires_grad]
    head_params     = [p for n, p in model.named_parameters()
                       if "classifier" in n]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=1e-4)

    # ── Train ─────────────────────────────────────────────────────────────────
    scheduler_type = None if args.scheduler == "none" else args.scheduler
    trainer = Trainer(
        model            = model,
        train_loader     = train_loader,
        val_loader       = val_loader,
        criterion        = criterion,
        optimizer        = optimizer,
        device           = device,
        checkpoint_dir   = args.checkpoint_dir,
        model_name       = "face_agent",
        epochs           = args.epochs,
        scheduler_type   = scheduler_type,
        early_stop_patience = args.patience,
        save_every       = 5,
    )

    history = trainer.train()

    # ── Summary ───────────────────────────────────────────────────────────────
    best_f1  = max(history["val_f1"])
    best_acc = max(history["val_acc"])
    logger.info(f"\n{'='*50}")
    logger.info(f"Training complete.")
    logger.info(f"Best Val F1  : {best_f1:.4f}")
    logger.info(f"Best Val Acc : {best_acc:.4f}")
    logger.info(f"Checkpoint   : {args.checkpoint_dir}/face_agent_best.pth")


if __name__ == "__main__":
    main()

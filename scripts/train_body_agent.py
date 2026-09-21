"""
Train the Body Agent (BodyLSTM / BodyGRU) on pose sequence data.

Usage:
    python scripts/train_body_agent.py --data-root data/raw/body
    python scripts/train_body_agent.py --data-root data/raw/body --model gru --bidirectional
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
import torch.nn as nn

from src.agents.body_agent.temporal_model import BodyLSTM, BodyGRU
from src.data_processing.dataset_loader import get_pose_loaders
from src.training.trainer import Trainer
from src.training.losses import FocalLoss
from src.utils.model_utils import get_device
from src.utils.logger import get_logger

logger = get_logger("train_body")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",    required=True)
    p.add_argument("--model",        default="lstm", choices=["lstm", "gru"])
    p.add_argument("--seq-len",      type=int,   default=30)
    p.add_argument("--hidden-size",  type=int,   default=256)
    p.add_argument("--num-layers",   type=int,   default=2)
    p.add_argument("--bidirectional",action="store_true")
    p.add_argument("--dropout",      type=float, default=0.3)
    p.add_argument("--epochs",       type=int,   default=60)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--scheduler",    default="cosine", choices=["cosine","plateau","none"])
    p.add_argument("--workers",      type=int,   default=4)
    p.add_argument("--checkpoint-dir", default="models/body_agent")
    p.add_argument("--patience",     type=int,   default=12)
    return p.parse_args()


def main():
    args   = parse_args()
    device = get_device()
    logger.info(f"Device: {device} | Args: {vars(args)}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = get_pose_loaders(
        root        = args.data_root,
        batch_size  = args.batch_size,
        num_workers = args.workers,
    )
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    ModelCls = BodyLSTM if args.model == "lstm" else BodyGRU
    model = ModelCls(
        input_size    = 99,
        hidden_size   = args.hidden_size,
        num_layers    = args.num_layers,
        num_classes   = 5,
        dropout       = args.dropout,
        bidirectional = args.bidirectional,
    )
    logger.info(f"Model: {args.model.upper()} | "
                f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Loss + Optimiser ──────────────────────────────────────────────────────
    criterion = FocalLoss(gamma=2.0)
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
        model_name          = f"body_{args.model}",
        epochs              = args.epochs,
        scheduler_type      = scheduler_type,
        early_stop_patience = args.patience,
        save_every          = 5,
    )

    history = trainer.train()

    best_f1  = max(history["val_f1"])
    best_acc = max(history["val_acc"])
    logger.info(f"\nBest Val F1: {best_f1:.4f} | Best Val Acc: {best_acc:.4f}")
    logger.info(f"Checkpoint: {args.checkpoint_dir}/body_{args.model}_best.pth")


if __name__ == "__main__":
    main()

"""
Evaluate a trained model on its test set and print a full report.
Supports face, body, speech, and fusion models.

Usage:
    # Face
    python scripts/evaluate_model.py face \\
        --checkpoint models/face_agent/face_agent_best.pth \\
        --data-root  data/raw/face/fer2013 --dataset fer2013

    # Speech
    python scripts/evaluate_model.py speech \\
        --checkpoint models/speech_agent/speech_cnn_lstm_best.pth \\
        --data-root  data/raw/speech/ravdess

    # Body
    python scripts/evaluate_model.py body \\
        --checkpoint models/body_agent/body_lstm_best.pth \\
        --data-root  data/raw/body
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.training.metrics import evaluate, per_class_accuracy
from src.utils.model_utils import get_device, load_checkpoint
from src.utils.logger import get_logger

logger = get_logger("evaluate")


# ── Per-modality loaders ───────────────────────────────────────────────────────

def load_face(args, device):
    from src.agents.face_agent.emotion_cnn import EmotionCNN
    from src.data_processing.dataset_loader import FER2013Dataset, RAFDBDataset
    from src.data_processing.data_augmentation import get_face_val_transforms

    model = EmotionCNN(num_classes=5, pretrained=False).to(device)
    load_checkpoint(model, None, args.checkpoint, str(device))
    model.eval()

    DS = FER2013Dataset if args.dataset == "fer2013" else RAFDBDataset
    ds = DS(args.data_root, split="test",
            transform=get_face_val_transforms())
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers)
    return model, loader


def load_speech(args, device):
    from src.agents.speech_agent.speech_model import SpeechCNNLSTM
    from src.data_processing.dataset_loader import RAVDESSDataset

    model = SpeechCNNLSTM(n_mfcc=40, num_classes=5).to(device)
    load_checkpoint(model, None, args.checkpoint, str(device))
    model.eval()

    ds = RAVDESSDataset(args.data_root, split="val", augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers)
    return model, loader


def load_body(args, device):
    from src.agents.body_agent.temporal_model import BodyLSTM
    from src.data_processing.dataset_loader import PoseSequenceDataset

    model = BodyLSTM(input_size=99, hidden_size=256,
                     num_layers=2, num_classes=5).to(device)
    load_checkpoint(model, None, args.checkpoint, str(device))
    model.eval()

    ds = PoseSequenceDataset(args.data_root, split="val", augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers)
    return model, loader


# ── Inference loop ─────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device):
    y_true, y_pred, y_proba = [], [], []

    for batch in loader:
        inputs, labels = batch
        if isinstance(inputs, (list, tuple)):
            inputs = tuple(x.to(device) for x in inputs)
            logits = model(*inputs)
        else:
            logits = model(inputs.to(device))

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())
        y_proba.extend(probs.tolist())

    return y_true, y_pred, np.array(y_proba)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("modality", choices=["face", "speech", "body"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root",  required=True)
    p.add_argument("--dataset",    default="fer2013", choices=["fer2013","rafdb"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--save-report", default=None,
                   help="Path to save JSON report (optional)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = get_device()
    logger.info(f"Evaluating {args.modality} | checkpoint: {args.checkpoint}")

    loaders = {"face": load_face, "speech": load_speech, "body": load_body}
    model, loader = loaders[args.modality](args, device)

    logger.info(f"Running inference on {len(loader.dataset)} samples...")
    y_true, y_pred, y_proba = run_inference(model, loader, device)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = evaluate(y_true, y_pred, y_proba)
    per_cls = per_class_accuracy(y_true, y_pred)

    print("\n" + "="*60)
    print(f"  Evaluation Report — {args.modality.upper()}")
    print("="*60)
    print(f"  Accuracy         : {metrics['accuracy']:.4f}")
    print(f"  F1 (weighted)    : {metrics['f1_weighted']:.4f}")
    print(f"  F1 (macro)       : {metrics['f1_macro']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro)   : {metrics['recall_macro']:.4f}")
    if metrics.get("auc_macro"):
        print(f"  AUC (macro)      : {metrics['auc_macro']:.4f}")
    print("\n  Per-class Accuracy:")
    for state, acc in per_cls.items():
        bar = "█" * int(acc * 20)
        print(f"    {state:<12} {acc:.3f}  {bar}")
    print("\n  Classification Report:")
    print(metrics["report"])
    print("  Confusion Matrix:")
    cm = np.array(metrics["confusion_matrix"])
    from config.constants import PSYCHOLOGICAL_STATES
    header = "  " + "  ".join(f"{s[:4]:>6}" for s in PSYCHOLOGICAL_STATES)
    print(header)
    for i, row in enumerate(cm):
        label = PSYCHOLOGICAL_STATES[i][:4]
        print(f"  {label:>4} " + "  ".join(f"{v:>6}" for v in row))
    print("="*60)

    # ── Save report ───────────────────────────────────────────────────────────
    if args.save_report:
        report = {
            "modality":    args.modality,
            "checkpoint":  args.checkpoint,
            "accuracy":    metrics["accuracy"],
            "f1_weighted": metrics["f1_weighted"],
            "f1_macro":    metrics["f1_macro"],
            "auc_macro":   metrics.get("auc_macro"),
            "per_class":   per_cls,
            "confusion_matrix": metrics["confusion_matrix"],
        }
        os.makedirs(os.path.dirname(args.save_report), exist_ok=True)
        with open(args.save_report, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved → {args.save_report}")


if __name__ == "__main__":
    main()

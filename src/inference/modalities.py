"""
Decide which modalities may run, based on which trained checkpoints exist.

Why this exists
---------------
Face and speech classifiers constructed without a checkpoint still run: they use an
ImageNet backbone / random weights with a randomly initialised head, so they emit
confident-looking but meaningless probabilities. Fusing those with real output, or
showing them on their own, is misleading. A modality therefore runs only when a
trained checkpoint is supplied. The body agent additionally has a hand-written
rule-based fallback; it is unvalidated, so it must be opted into explicitly.
"""
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ModalityPlan:
    face: bool
    body: bool
    speech: bool
    body_is_rule_based: bool = False

    @property
    def active(self) -> List[str]:
        return [name for name, on in
                (("face", self.face), ("body", self.body), ("speech", self.speech)) if on]


def _checked(path: Optional[str], label: str) -> Optional[str]:
    """Return a usable path or None; raise if a path was given but does not exist."""
    if not path:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} checkpoint not found: {path}")
    return path


def plan_modalities(face_model_path: Optional[str] = None,
                    body_model_path: Optional[str] = None,
                    speech_model_path: Optional[str] = None,
                    allow_rule_based_body: bool = False) -> ModalityPlan:
    """
    Build the plan of modalities to run.

    Raises:
        FileNotFoundError: a checkpoint path was given but the file does not exist.
        ValueError: nothing would run (no checkpoint and no opted-in body heuristics).
    """
    face   = _checked(face_model_path,   "Face")   is not None
    speech = _checked(speech_model_path, "Speech") is not None
    body_trained = _checked(body_model_path, "Body") is not None

    body = body_trained or allow_rule_based_body
    plan = ModalityPlan(face=face, body=body, speech=speech,
                        body_is_rule_based=body and not body_trained)

    if not plan.active:
        raise ValueError(
            "No modality can run: supply at least one trained checkpoint "
            "(face / body / speech), or opt in to the experimental rule-based body "
            "heuristics. Untrained models are not run because their output is noise."
        )
    return plan

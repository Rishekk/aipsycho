"""
BatchPredictor must fail closed, like RealtimePredictor: no agent is built or run
unless a trained checkpoint (or the explicit rule-based opt-in) allows it.

The real agents need torch / OpenCV / MediaPipe, so they are replaced by stubs here;
what is under test is the gating and wiring, not the models.
"""
import importlib
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

PROBS = {"calm": 0.6, "stressed": 0.1, "anxious": 0.1, "confident": 0.1, "frustrated": 0.1}


@pytest.fixture
def env(monkeypatch):
    face = types.ModuleType("src.agents.face_agent")
    face.FaceDetector = MagicMock(name="FaceDetector")
    face.FaceClassifier = MagicMock(name="FaceClassifier")
    body = types.ModuleType("src.agents.body_agent")
    body.BodyClassifier = MagicMock(name="BodyClassifier")
    fusion = types.ModuleType("src.fusion.late_fusion")
    fusion.LateFusionClassifier = MagicMock(name="LateFusionClassifier")
    fusion.LateFusionClassifier.return_value.predict_with_missing.return_value = (
        "calm", 0.6, PROBS)

    for name, mod in (("cv2", MagicMock()),
                      ("src.agents.face_agent", face),
                      ("src.agents.body_agent", body),
                      ("src.fusion.late_fusion", fusion)):
        monkeypatch.setitem(sys.modules, name, mod)
    sys.modules.pop("src.inference.predictor", None)
    module = importlib.import_module("src.inference.predictor")
    yield types.SimpleNamespace(cls=module.BatchPredictor, face=face, body=body,
                                fusion=fusion)
    sys.modules.pop("src.inference.predictor", None)


@pytest.fixture
def ckpt(tmp_path):
    path = tmp_path / "model.pth"
    path.write_bytes(b"stub")
    return str(path)


def test_refuses_to_run_with_no_checkpoint(env):
    with pytest.raises(ValueError):
        env.cls()
    env.face.FaceClassifier.assert_not_called()
    env.body.BodyClassifier.assert_not_called()


def test_missing_checkpoint_path_is_an_error(env, tmp_path):
    with pytest.raises(FileNotFoundError):
        env.cls(face_model_path=str(tmp_path / "nope.pth"))


def test_face_only_builds_only_the_face_agent(env, ckpt):
    p = env.cls(face_model_path=ckpt)
    assert p.plan.active == ["face"]
    env.face.FaceClassifier.assert_called_once()
    env.body.BodyClassifier.assert_not_called()
    assert p.body_classifier is None


def test_body_checkpoint_only_builds_only_the_body_agent(env, ckpt):
    p = env.cls(body_model_path=ckpt)
    assert p.plan.active == ["body"] and not p.plan.body_is_rule_based
    env.body.BodyClassifier.assert_called_once()
    assert env.body.BodyClassifier.call_args.kwargs["model_path"] == ckpt
    env.face.FaceClassifier.assert_not_called()
    assert p.face_classifier is None and p.face_detector is None


def test_rule_based_body_needs_explicit_opt_in(env):
    with pytest.raises(ValueError):
        env.cls(allow_rule_based_body=False)

    p = env.cls(allow_rule_based_body=True)
    assert p.plan.body_is_rule_based
    assert env.body.BodyClassifier.call_args.kwargs["model_path"] is None


@pytest.mark.parametrize("strategy", ["attention", "early", "nonsense"])
def test_unsupported_fusion_is_rejected_before_any_agent_is_built(env, ckpt, strategy):
    with pytest.raises(ValueError):
        env.cls(face_model_path=ckpt, fusion_strategy=strategy)
    env.face.FaceClassifier.assert_not_called()


def test_disabled_modalities_are_passed_as_missing_to_fusion(env, ckpt):
    p = env.cls(body_model_path=ckpt)
    p.body_classifier.process_frame.return_value = ("calm", 0.6, PROBS)

    out = p._process_frame(np.zeros((4, 4, 3), dtype=np.uint8))

    body_probs, face_probs, speech_probs = (
        env.fusion.LateFusionClassifier.return_value.predict_with_missing.call_args.args)
    assert body_probs is not None and face_probs is None and speech_probs is None
    assert out is not None


def test_frame_with_no_output_is_skipped(env, ckpt):
    p = env.cls(body_model_path=ckpt)
    p.body_classifier.process_frame.return_value = None  # e.g. buffer still warming up
    assert p._process_frame(np.zeros((4, 4, 3), dtype=np.uint8)) is None

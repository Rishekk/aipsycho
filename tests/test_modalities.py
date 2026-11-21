import pytest

from src.inference.modalities import plan_modalities


def _ckpt(tmp_path, name):
    p = tmp_path / name
    p.write_bytes(b"x")
    return str(p)


def test_nothing_supplied_refuses_to_run():
    with pytest.raises(ValueError):
        plan_modalities()


def test_untrained_face_and_speech_never_run_without_checkpoints():
    plan = plan_modalities(allow_rule_based_body=True)
    assert not plan.face and not plan.speech
    assert plan.active == ["body"] and plan.body_is_rule_based


def test_rule_based_body_is_opt_in():
    with pytest.raises(ValueError):
        plan_modalities(allow_rule_based_body=False)


def test_checkpoints_enable_their_modalities(tmp_path):
    plan = plan_modalities(face_model_path=_ckpt(tmp_path, "f.pth"),
                           speech_model_path=_ckpt(tmp_path, "s.pth"))
    assert plan.active == ["face", "speech"]
    assert not plan.body


def test_trained_body_checkpoint_is_not_rule_based(tmp_path):
    plan = plan_modalities(body_model_path=_ckpt(tmp_path, "b.pth"), allow_rule_based_body=True)
    assert plan.body and not plan.body_is_rule_based


def test_missing_checkpoint_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        plan_modalities(face_model_path=str(tmp_path / "does_not_exist.pth"))

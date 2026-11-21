from config.constants import PSYCHOLOGICAL_STATES
from src.inference.post_processor import (
    DISCLAIMER, STATE_COLORS, STATE_MESSAGES, PostProcessor,
)


def _probs(top, p=0.8):
    rest = (1 - p) / (len(PSYCHOLOGICAL_STATES) - 1)
    return {s: (p if s == top else rest) for s in PSYCHOLOGICAL_STATES}


def test_confident_prediction_is_reported():
    pp = PostProcessor(window_size=5, confidence_threshold=0.4)
    out = pp.process("calm", 0.8, _probs("calm"))
    assert out["state"] == "calm"


def test_low_confidence_frame_is_uncertain_even_after_confident_history():
    # Regression: low-confidence frames used to show the previous majority state.
    pp = PostProcessor(window_size=5, confidence_threshold=0.4)
    for _ in range(5):
        pp.process("calm", 0.9, _probs("calm", 0.9))
    out = pp.process("anxious", 0.25, _probs("anxious", 0.25))
    assert out["state"] == "uncertain"
    assert out["message"] == STATE_MESSAGES["uncertain"]


def test_uncertain_frames_do_not_enter_session_stats():
    pp = PostProcessor(window_size=5, confidence_threshold=0.4)
    pp.process("calm", 0.9, _probs("calm", 0.9))
    pp.process("anxious", 0.2, _probs("anxious", 0.2))
    stats = pp.get_session_stats()
    assert stats["calm"] == 1.0 and stats["anxious"] == 0.0


def test_majority_vote_smooths_a_single_outlier():
    pp = PostProcessor(window_size=5, confidence_threshold=0.4)
    for _ in range(4):
        pp.process("calm", 0.9, _probs("calm", 0.9))
    out = pp.process("frustrated", 0.9, _probs("frustrated", 0.9))
    assert out["state"] == "calm"
    assert out["raw_state"] == "frustrated"


def test_reset_session_clears_history():
    pp = PostProcessor(window_size=5, confidence_threshold=0.4)
    pp.process("calm", 0.9, _probs("calm", 0.9))
    pp.reset_session()
    assert sum(pp.get_session_stats().values()) == 0.0


def test_messages_are_descriptive_not_advice():
    advice_words = ("consider", "try ", "take a", "should", "breath", "grounding")
    for state, msg in STATE_MESSAGES.items():
        assert not any(w in msg.lower() for w in advice_words), (state, msg)
    for s in PSYCHOLOGICAL_STATES:
        assert s in STATE_MESSAGES and s in STATE_COLORS


def test_disclaimer_states_the_limitation():
    assert "not a measure of stress" in DISCLAIMER.lower() or "not a measure" in DISCLAIMER.lower()

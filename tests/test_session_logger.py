import csv

from config.constants import PSYCHOLOGICAL_STATES
from app.session_logger import SessionLogger


def test_logger_writes_one_row_per_call_with_expected_columns(tmp_path):
    log = SessionLogger("t1", log_dir=str(tmp_path))
    probs = {s: 0.2 for s in PSYCHOLOGICAL_STATES}
    log.log(1, 1, {"state": "calm", "confidence": 0.2, "probs": probs, "micro_event": True})
    log.log(2, 1, {"state": "calm", "confidence": 0.2, "probs": probs})
    log.close()

    with open(log.filepath, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["person_id"] == "P1" and rows[0]["micro_event"] == "1"
    assert rows[1]["micro_event"] == "0"
    assert all(f"prob_{s}" in rows[0] for s in PSYCHOLOGICAL_STATES)
    assert not log.is_open

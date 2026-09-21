"""
Per-session CSV logger for psychological state detection.

Each session writes one CSV file to results/logs/.
Every row corresponds to one frame × one detected person, giving evaluators
a complete record of who was detected, when, and what state was inferred.

CSV columns:
    timestamp       ISO-8601 wall-clock time of the frame
    session_id      Short identifier for the recording session
    frame_no        Monotonically increasing frame counter within the session
    person_id       Stable tracker ID (P1, P2, …)
    state           Predicted psychological state label
    confidence      Confidence of the predicted state (0–1)
    prob_calm       Per-class probability
    prob_stressed
    prob_anxious
    prob_confident
    prob_frustrated
    micro_event     1 if a micro-expression spike was detected, else 0
"""
import csv
import os
from datetime import datetime
from typing import Dict

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.constants import PSYCHOLOGICAL_STATES


class SessionLogger:
    """Write per-person per-frame detection results to a CSV file."""

    FIELDNAMES = (
        ["timestamp", "session_id", "frame_no", "person_id",
         "state", "confidence"]
        + [f"prob_{s}" for s in PSYCHOLOGICAL_STATES]
        + ["micro_event"]
    )

    def __init__(self, session_id: str, log_dir: str = "results/logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.session_id = session_id
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._filepath = os.path.join(log_dir, f"session_{ts}_{session_id}.csv")
        self._file     = open(self._filepath, "w", newline="", encoding="utf-8")
        self._writer   = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self._file.flush()

    # ------------------------------------------------------------------
    def log(self, frame_no: int, person_id: int, result: Dict) -> None:
        """
        Append one row for the given person's result on this frame.

        Parameters
        ----------
        frame_no  : Global frame counter for the session.
        person_id : Integer ID assigned by PersonTracker.
        result    : Result dict produced by _demo_result_for_person or the
                    RealtimePredictor pipeline.
        """
        probs = result.get("probs", {})
        row = {
            "timestamp":  datetime.now().isoformat(timespec="milliseconds"),
            "session_id": self.session_id,
            "frame_no":   frame_no,
            "person_id":  f"P{person_id}",
            "state":      result.get("state", "unknown"),
            "confidence": round(result.get("confidence", 0.0), 4),
            **{f"prob_{s}": round(probs.get(s, 0.0), 4) for s in PSYCHOLOGICAL_STATES},
            "micro_event": int(bool(result.get("micro_event", False))),
        }
        self._writer.writerow(row)
        self._file.flush()

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Flush and close the underlying file."""
        if not self._file.closed:
            self._file.close()

    @property
    def filepath(self) -> str:
        return self._filepath

    @property
    def is_open(self) -> bool:
        return not self._file.closed

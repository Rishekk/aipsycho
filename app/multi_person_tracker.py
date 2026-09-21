"""
IoU-based multi-person tracker.

Assigns stable integer IDs to face bounding boxes across frames so each
individual in the frame keeps a consistent identity throughout a session.
Each tracked person also maintains its own independent demo-mode state drift.
"""
import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.constants import PSYCHOLOGICAL_STATES


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Intersection-over-Union of two (x, y, w, h) bounding boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix  = max(ax, bx)
    iy  = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    inter = max(0, ix2 - ix) * max(0, iy2 - iy)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class TrackedPerson:
    """Holds per-person tracking state and independent demo prediction state."""

    def __init__(self, pid: int, bbox: Tuple[int, int, int, int]):
        self.pid   = pid
        self.bbox  = bbox
        self.lost  = 0                          # consecutive frames without a detection
        # Independent demo-mode state drift — each person starts at a random state
        self._state_idx = random.randint(0, len(PSYCHOLOGICAL_STATES) - 1)

    def next_demo_state(self) -> str:
        """Return the next psychological state for this person, drifting slowly."""
        if random.random() < 0.05:
            self._state_idx = (self._state_idx + 1) % len(PSYCHOLOGICAL_STATES)
        return PSYCHOLOGICAL_STATES[self._state_idx]


class PersonTracker:
    """
    Greedy IoU-based tracker.

    On each call to `update(bboxes)`:
      - Existing tracks are matched to new detections by highest IoU.
      - Unmatched detections spawn new tracks with fresh IDs.
      - Unmatched tracks age out after `max_lost_frames` consecutive misses.

    Returns a list of (person_id, bbox) for every currently active detection.
    """

    def __init__(self, iou_threshold: float = 0.25, max_lost_frames: int = 15):
        self._next_id    = 1
        self._persons: Dict[int, TrackedPerson] = {}
        self.iou_threshold   = iou_threshold
        self.max_lost_frames = max_lost_frames

    # ------------------------------------------------------------------
    def update(
        self,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[int, Tuple[int, int, int, int]]]:
        """Match detections → tracks; return [(pid, bbox)] for live tracks."""

        tracked_ids = list(self._persons.keys())

        if not bboxes:
            # Age every track; evict those that have been missing too long
            dead = [pid for pid, p in self._persons.items()
                    if p.lost + 1 > self.max_lost_frames]
            for pid in dead:
                del self._persons[pid]
            for p in self._persons.values():
                p.lost += 1
            return []

        matched_tracks = set()
        matched_dets   = set()

        # Greedy matching: for each detection find the best overlapping track
        for det_idx, bbox in enumerate(bboxes):
            best_iou = self.iou_threshold
            best_tid = None
            for tid in tracked_ids:
                if tid in matched_tracks:
                    continue
                iou = _iou(self._persons[tid].bbox, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                self._persons[best_tid].bbox = bbox
                self._persons[best_tid].lost = 0
                matched_tracks.add(best_tid)
                matched_dets.add(det_idx)

        # Spawn new tracks for unmatched detections
        for det_idx, bbox in enumerate(bboxes):
            if det_idx not in matched_dets:
                pid = self._next_id
                self._next_id += 1
                self._persons[pid] = TrackedPerson(pid, bbox)
                matched_tracks.add(pid)

        # Age unmatched existing tracks; evict stale ones
        dead = []
        for tid in tracked_ids:
            if tid not in matched_tracks:
                self._persons[tid].lost += 1
                if self._persons[tid].lost > self.max_lost_frames:
                    dead.append(tid)
        for tid in dead:
            del self._persons[tid]

        return [
            (p.pid, p.bbox)
            for p in self._persons.values()
            if p.lost == 0
        ]

    def get_person(self, pid: int) -> Optional[TrackedPerson]:
        return self._persons.get(pid)

    def reset(self):
        self._next_id = 1
        self._persons.clear()

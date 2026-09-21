"""
Record labeled body pose sequences from webcam for body agent training.

Sit/stand in front of your webcam and act out each psychological state.
The script records 30-frame sequences and saves them as .npy files.

Usage:
    python scripts/record_body_data.py --state calm     --samples 100
    python scripts/record_body_data.py --state stressed --samples 100
    python scripts/record_body_data.py --state anxious  --samples 100
    python scripts/record_body_data.py --state confident --samples 100
    python scripts/record_body_data.py --state frustrated --samples 100

Controls:
    SPACE  — start/stop recording a sequence
    Q      — quit

Output: data/raw/body/train/<state>_NNNN.npy  (shape: 30, 99)
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import cv2
import numpy as np
from collections import deque

from src.agents.body_agent.pose_estimator import PoseEstimator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--state",    required=True,
                   choices=["calm", "stressed", "anxious", "confident", "frustrated"])
    p.add_argument("--samples",  type=int, default=100, help="Number of sequences to record")
    p.add_argument("--seq-len",  type=int, default=30,  help="Frames per sequence")
    p.add_argument("--camera",   type=int, default=0)
    p.add_argument("--output",   default="data/raw/body/train")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Count existing samples for this state
    existing = len([f for f in os.listdir(args.output)
                    if f.startswith(args.state + "_") and f.endswith(".npy")])
    print(f"\nRecording {args.state.upper()} sequences")
    print(f"Already saved: {existing} | Target: {args.samples}")
    print(f"Output dir   : {args.output}")
    print("---")
    print("Press SPACE to start recording a sequence.")
    print("Hold the pose steadily until the sequence is captured.")
    print("Press Q to quit.\n")

    pose = PoseEstimator()
    cap  = cv2.VideoCapture(args.camera)

    buffer     = deque(maxlen=args.seq_len)
    recording  = False
    saved      = existing

    while saved < (existing + args.samples):
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = pose.extract_pose(frame)
        display   = frame.copy()

        # Draw skeleton if landmarks found
        if landmarks is not None:
            pose.draw_pose(display, frame)
            if recording:
                feat = landmarks.flatten().astype(np.float32)
                buffer.append(feat)
                # Progress bar
                pct = len(buffer) / args.seq_len
                bar_w = int(300 * pct)
                cv2.rectangle(display, (10, 60), (10 + bar_w, 80), (0, 255, 0), -1)
                cv2.rectangle(display, (10, 60), (310, 80), (255, 255, 255), 2)
                cv2.putText(display, f"Recording... {len(buffer)}/{args.seq_len}",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(display, "No pose detected", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Auto-save when buffer is full
        if recording and len(buffer) == args.seq_len:
            seq = np.array(buffer, dtype=np.float32)   # (30, 99)
            fname = f"{args.state}_{saved:04d}.npy"
            np.save(os.path.join(args.output, fname), seq)
            saved += 1
            buffer.clear()
            recording = False
            print(f"  Saved {fname}  ({saved - existing}/{args.samples})")

        # HUD
        status = "RECORDING" if recording else "READY — press SPACE"
        cv2.putText(display, f"State: {args.state.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(display, f"Saved: {saved - existing}/{args.samples}", (10, display.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(display, status, (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if recording else (200, 200, 200), 2)
        cv2.imshow("Body Data Recorder", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and not recording and landmarks is not None:
            recording = True
            buffer.clear()
        elif key == ord("q"):
            break

    cap.release()
    pose.close()
    cv2.destroyAllWindows()
    print(f"\nDone. Total saved: {saved} sequences for '{args.state}'")


if __name__ == "__main__":
    main()

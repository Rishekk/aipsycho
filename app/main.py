"""
Entry point for headless real-time affect-cue estimation (no Streamlit).
Runs webcam + mic and overlays predictions on the video window.
Press Q to quit.

Only modalities with a trained checkpoint run. Run it with e.g.
    python -m app.main --consent --face-model models/face_agent/face_agent_best.pth
"""
import cv2
import argparse
import sys
from src.inference.realtime_predictor import RealtimePredictor
from src.inference.post_processor import STATE_COLORS, DISCLAIMER
from src.utils.logger import get_logger

logger = get_logger("main")


def parse_args():
    p = argparse.ArgumentParser(description="Real-time affect-cue estimation (research prototype)")
    p.add_argument("--face-model",   default=None, help="trained face checkpoint (.pth)")
    p.add_argument("--body-model",   default=None, help="trained body checkpoint (.pth)")
    p.add_argument("--speech-model", default=None, help="trained speech checkpoint (.pth)")
    p.add_argument("--allow-rule-based-body", action="store_true",
                   help="use the unvalidated rule-based body heuristics when no body checkpoint is given")
    p.add_argument("--camera",       type=int, default=0)
    p.add_argument("--consent", action="store_true",
                   help="confirm that everyone in view / earshot has agreed to be analysed")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.consent:
        sys.exit("Refusing to start: pass --consent to confirm that everyone in view / "
                 "earshot has agreed to be analysed.\n" + DISCLAIMER)

    predictor = RealtimePredictor(
        face_model_path   = args.face_model,
        body_model_path   = args.body_model,
        speech_model_path = args.speech_model,
        camera_idx        = args.camera,
        allow_rule_based_body = args.allow_rule_based_body,
    )
    predictor.start()
    logger.info("Running. Press Q in the window to quit.")

    while True:
        result = predictor.get_result(timeout=0.05)
        if result is None:
            continue

        frame = result.get("frame_rgb")
        if frame is None:
            continue

        import cv2 as _cv2
        frame_bgr = _cv2.cvtColor(frame, _cv2.COLOR_RGB2BGR)

        state  = result["state"]
        conf   = result["confidence"]
        color  = result.get("color", (150, 150, 150))
        bgr    = (color[2], color[1], color[0])

        _cv2.putText(frame_bgr, f"{state.upper()}  {conf*100:.0f}%",
                     (10, 40), _cv2.FONT_HERSHEY_SIMPLEX, 1.2, bgr, 2)
        _cv2.imshow("Affect-cue estimation (research prototype)", frame_bgr)

        if _cv2.waitKey(1) & 0xFF == ord("q"):
            break

    predictor.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

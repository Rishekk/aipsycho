"""
Entry point for headless real-time detection (no Streamlit).
Runs webcam + mic and prints predictions to terminal.
Press Q to quit.
"""
import cv2
import argparse
from src.inference.realtime_predictor import RealtimePredictor
from src.inference.post_processor import STATE_COLORS
from src.utils.logger import get_logger

logger = get_logger("main")


def parse_args():
    p = argparse.ArgumentParser(description="Real-time psychological state detection")
    p.add_argument("--face-model",   default=None)
    p.add_argument("--body-model",   default=None)
    p.add_argument("--speech-model", default=None)
    p.add_argument("--camera",       type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    predictor = RealtimePredictor(
        face_model_path   = args.face_model,
        body_model_path   = args.body_model,
        speech_model_path = args.speech_model,
        camera_idx        = args.camera,
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
        _cv2.imshow("Psychological State Detection", frame_bgr)

        if _cv2.waitKey(1) & 0xFF == ord("q"):
            break

    predictor.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

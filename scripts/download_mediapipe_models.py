"""
Download required MediaPipe Tasks model files.
Run once before starting the app: python scripts/download_mediapipe_models.py
"""
import os
import urllib.request

MODELS_DIR = "models/mediapipe"
MODELS = {
    "blaze_face_short_range.tflite": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
}

os.makedirs(MODELS_DIR, exist_ok=True)

for filename, url in MODELS.items():
    dest = os.path.join(MODELS_DIR, filename)
    if os.path.exists(dest):
        print(f"  [skip] {filename} already exists")
        continue
    print(f"  Downloading {filename}...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved → {dest}")

print("\nAll models ready.")

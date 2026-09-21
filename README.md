# AI-Powered Psychological State Detection

A real-time multimodal system for detecting psychological states (calm, stressed, anxious, confident, frustrated) from facial expressions, body language, and speech.

**Team:** Nitin Kumar Jha, Rishek Raj, Risheek Yadav
**Supervisor:** Dr. Dhirendra Kumar
**Institution:** Delhi Technological University

---

## Architecture

```
Webcam (Video) ──► Face Agent  (ResNet50 + MediaPipe) ──┐
Webcam (Video) ──► Body Agent  (MediaPipe + LSTM)      ──┼──► Fusion ──► Psychological State
Microphone     ──► Speech Agent (CNN-LSTM / wav2vec2)  ──┘
```

The system uses three specialized agents, each independently analyzing one modality, then combines their outputs through attention-based fusion.

---

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate          # Linux/macOS
venv\Scripts\activate             # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download MediaPipe Models

```bash
python scripts/download_mediapipe_models.py
```

---

## Training

### Face Agent (FER-2013 or RAF-DB)

```bash
python scripts/train_face_agent.py \
    --dataset fer2013 \
    --data-root data/raw/face/fer2013 \
    --epochs 50 --lr 1e-4
```

### Body Agent (Pose Sequence Data)

```bash
python scripts/train_body_agent.py \
    --data-root data/raw/body \
    --model lstm --epochs 60
```

### Speech Agent (RAVDESS)

```bash
python scripts/train_speech_agent.py \
    --data-root data/raw/speech/ravdess \
    --model cnn_lstm --epochs 60
```

### Fusion Model

First, build a multimodal dataset from labelled video clips:

```
data/raw/multimodal/
    calm/        clip1.mp4  clip2.mp4 ...
    stressed/    ...
    anxious/     ...
    confident/   ...
    frustrated/  ...
```

Then train:

```bash
python scripts/train_fusion.py \
    --data-root data/processed/multimodal \
    --face-model   models/face_agent/face_agent_best.pth \
    --body-model   models/body_agent/body_lstm_best.pth \
    --speech-model models/speech_agent/speech_cnn_lstm_best.pth \
    --strategy attention
```

---

## Evaluation

```bash
# Face agent
python scripts/evaluate_model.py face \
    --checkpoint models/face_agent/face_agent_best.pth \
    --data-root  data/raw/face/fer2013 --dataset fer2013

# Speech agent
python scripts/evaluate_model.py speech \
    --checkpoint models/speech_agent/speech_cnn_lstm_best.pth \
    --data-root  data/raw/speech/ravdess

# Body agent
python scripts/evaluate_model.py body \
    --checkpoint models/body_agent/body_lstm_best.pth \
    --data-root  data/raw/body
```

---

## Running the Application

### Streamlit Web UI

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501`. Enable **Demo Mode** to run without trained models (generates synthetic predictions from a live webcam feed).

To run with real models, enter checkpoint paths in the sidebar and disable Demo Mode.

### Headless (Terminal)

```bash
python app/main.py \
    --face-model   models/face_agent/face_agent_best.pth \
    --body-model   models/body_agent/body_lstm_best.pth \
    --speech-model models/speech_agent/speech_cnn_lstm_best.pth
```

---

## Project Structure

```
src/
  agents/
    face_agent/     ResNet50 CNN + MediaPipe landmarks
    body_agent/     MediaPipe pose + LSTM temporal model
    speech_agent/   CNN-LSTM / wav2vec2 on MFCCs
  fusion/
    attention_fusion.py   Multi-head self-attention over modality tokens
    early_fusion.py       Feature concatenation + MLP
    late_fusion.py        Weighted probability averaging
    ensemble.py           Ensemble + temporal smoothing
  data_processing/  Dataset loaders, augmentation, synchronizer
  training/         Trainer, validator, losses, metrics
  inference/        Real-time and batch predictors, post-processor

scripts/
  train_face_agent.py     FER-2013 / RAF-DB training
  train_body_agent.py     Pose sequence training
  train_speech_agent.py   RAVDESS speech training
  train_fusion.py         Multimodal fusion training
  evaluate_model.py       Evaluation & metrics report

app/
  streamlit_app.py    Web UI
  main.py             Headless CLI
```

---

## Psychological State Classes

| Index | State       | Mapped From (emotions)           |
|-------|-------------|----------------------------------|
| 0     | Calm        | neutral, calm                    |
| 1     | Stressed    | sad, sadness                     |
| 2     | Anxious     | fear, fearful, surprise          |
| 3     | Confident   | happy, happiness, excitement     |
| 4     | Frustrated  | angry, anger, disgust, contempt  |

---

## Datasets

| Modality | Dataset   | Download |
|----------|-----------|---------|
| Face     | FER-2013  | [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) |
| Face     | RAF-DB    | [Website](http://www.whdeng.cn/raf/model1.html) |
| Speech   | RAVDESS   | [Zenodo](https://zenodo.org/record/1188976) |
| Body     | Custom    | Record pose sequences with provided scripts |

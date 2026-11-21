# AI-Powered Psychological State Detection

A real-time multimodal research prototype that estimates **affect cues** (calm, stressed, anxious, confident, frustrated) from facial expressions, body language, and speech.

> **Status: research prototype.** The models are trained on acted/posed *emotion* datasets
> (FER-2013, RAF-DB, RAVDESS) and the five classes are a relabelling of those emotions
> (see [What the classes mean](#what-the-classes-mean)). The output is **not** a measure of
> stress, anxiety or any psychological state, and not a diagnosis. Do not use it to evaluate
> people (e.g. employees, students, job candidates); in the EU that use is prohibited by the
> AI Act. How to find out whether the approach holds up on real data:
> [docs/VALIDATION_PLAN.md](docs/VALIDATION_PLAN.md).

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

The system uses three specialized agents, each independently analyzing one modality.
The **live pipeline uses late fusion with equal weights** (no validation data supports any
other weighting). Early and attention fusion are implemented and trainable, but they need a
labelled multimodal dataset that this repository does not include, and they have not been
evaluated.

A modality runs only when a **trained checkpoint** is supplied. Untrained models emit noise,
so there is no "demo mode" any more. The rule-based body heuristics are unvalidated and must
be switched on explicitly (`--allow-rule-based-body` / the sidebar checkbox).

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

Poses are normalised **per frame** (hip-centred, torso length 1) by one function
(`src/data_processing/pose_norm.py`) used both when training and by the live `BodyClassifier`.
A body checkpoint trained before this change used a different (first-frame) normalisation and
must be retrained.

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

The multimodal dataset must be built with `MultimodalDatasetBuilder`, which records each
sample's source clip. Training then splits **by clip** (all windows cut from one video stay on
one side), because overlapping windows from the same clip are near-duplicates. This is not a
subject-level split: if the same person appears in several clips they can land on both sides.
Datasets built before this change have no source field and are rejected with a clear error.

---

## Evaluation

Splits are **subject-independent** where subject identity exists, and test data is never used
for model selection:

| Dataset  | train | val (checkpoint selection) | test (report once) |
|----------|-------|----------------------------|--------------------|
| RAVDESS  | actors 1-16 | actors 17-20 | actors 21-24 (2 male + 2 female held-out in val and test) |
| FER-2013 | Training | PublicTest (CSV) or 10% hold-out of train (folder layout) | PrivateTest / `test/` |
| RAF-DB   | train minus 10% | 10% hold-out of train | test |
| Fusion (clips) | clips, by source | whole clips moved to val (`--val-split`) | none yet |

> The numbers in `results/metrics/*.json` were produced **before** this fix: the speech split
> was a random file-level shuffle (every RAVDESS actor appeared in both train and test) and
> the face checkpoint was selected on the same split it was reported on. Treat them as
> optimistic and regenerate them.

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

Open `http://localhost:8501`, enter at least one trained checkpoint path in the sidebar
(or tick the experimental rule-based body option), confirm that everyone in view and earshot
has agreed to be analysed, and press Start. Video and audio are analysed locally and are not
saved; the CSV session log contains only per-frame probabilities.

### Headless (Terminal)

```bash
python -m app.main --consent \
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

## What the classes mean

The model outputs one of five labels, but each label is just a regrouping of the emotion
categories in the training data (`EMOTION_TO_STATE` in `src/data_processing/dataset_loader.py`).
A sad face is labelled "stressed" and a surprised face "anxious" by construction, not because
the data measures stress or anxiety.

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

---

## Tests

```bash
pip install pytest
python -m pytest
```

The suite covers the subject-independent and clip-level splits, modality gating (live and
batch), the shared pose normalisation, post-processing, the tracker and the session logger.
It runs without torch installed; the torch, OpenCV and MediaPipe parts are stubbed or not
covered, so the live pipeline still needs a manual run on a machine with a camera.

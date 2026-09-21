# AI-Powered Psychological State Detection - Project Roadmap

## Project Overview
**Team Members:** Nitin Kumar Jha, Rishek Raj, Risheek Yadav  
**Supervisor:** Dr. Dhirendra Kumar  
**Duration:** B.Tech Project-I (Semester-based)

---

## Phase 1: Foundation & Setup (Weeks 1-2)

### 1.1 Environment Setup
- [ ] Set up Python 3.9+ development environment
- [ ] Install core dependencies (PyTorch/TensorFlow, OpenCV, MediaPipe)
- [ ] Set up version control (Git repository)
- [ ] Configure virtual environment and requirements.txt
- [ ] Set up project directory structure

### 1.2 Literature Review & Dataset Research
- [ ] Deep dive into referenced papers
- [ ] Identify suitable datasets:
  - **Body Language:** EmoPain, EmotioNet, BoLD (Body Language Dataset)
  - **Facial:** AffectNet, FER-2013, CK+, RAF-DB
  - **Speech:** IEMOCAP, RAVDESS, MSP-IMPROV
  - **Multimodal:** GEMEP, CMU-MOSEI
- [ ] Download and organize datasets
- [ ] Create data preprocessing scripts

### 1.3 Hardware & Tools Assessment
- [ ] Test webcam quality and frame rate
- [ ] Test microphone quality and sampling rate
- [ ] Set up GPU access (CUDA) if available
- [ ] Benchmark system performance

---

## Phase 2: Individual Agent Development (Weeks 3-6)

### 2.1 Body Agent Development
**Goal:** Analyze posture, gestures, and movement dynamics

#### Tasks:
- [ ] Implement pose estimation using MediaPipe or OpenPose
- [ ] Extract keypoint features (33 body landmarks)
- [ ] Design feature engineering pipeline:
  - Joint angles
  - Spatial relationships
  - Temporal dynamics (motion vectors)
- [ ] Train LSTM/GRU for temporal modeling
- [ ] Evaluate on body language datasets
- [ ] Save model checkpoints

**Key Technologies:**
- MediaPipe Pose / OpenPose
- PyTorch/TensorFlow for temporal models
- NumPy for feature extraction

---

### 2.2 Face Agent Development
**Goal:** Detect facial expressions and micro-movements

#### Tasks:
- [ ] Implement facial landmark detection (68-point or 478-point)
- [ ] Extract facial features:
  - Action Units (AUs) using OpenFace
  - Geometric features (distances, angles)
  - Texture features
- [ ] Fine-tune CNN model (ResNet/EfficientNet) on facial emotion datasets
- [ ] Add attention mechanism for micro-expression detection
- [ ] Evaluate on FER datasets
- [ ] Save model checkpoints

**Key Technologies:**
- MediaPipe Face Mesh / Dlib
- Pre-trained CNNs (ResNet50, EfficientNet-B0)
- PyTorch/TensorFlow

---

### 2.3 Speech Agent Development
**Goal:** Extract prosodic and spectral features from voice

#### Tasks:
- [ ] Implement audio preprocessing pipeline
- [ ] Extract acoustic features:
  - Mel-Frequency Cepstral Coefficients (MFCCs)
  - Pitch, energy, speaking rate
  - Formants and jitter
  - Zero-crossing rate
- [ ] Train audio classification model (CNN-LSTM or Transformer)
- [ ] Use pretrained models (wav2vec 2.0 / HuBERT) for transfer learning
- [ ] Evaluate on speech emotion datasets
- [ ] Save model checkpoints

**Key Technologies:**
- Librosa for audio processing
- OpenSMILE for feature extraction
- wav2vec 2.0 / HuBERT from HuggingFace
- PyTorch Audio

---

## Phase 3: Fusion Agent & Integration (Weeks 7-9)

### 3.1 Fusion Architecture Design
**Goal:** Combine outputs from all three agents

#### Fusion Strategies to Implement:
1. **Early Fusion:** Concatenate features before classification
2. **Late Fusion:** Weighted average of individual predictions
3. **Hybrid Fusion:** Attention-based feature fusion + decision fusion

#### Tasks:
- [ ] Design fusion network architecture
- [ ] Implement attention mechanism for modality weighting
- [ ] Train end-to-end multimodal model
- [ ] Experiment with different fusion techniques
- [ ] Cross-validate on multimodal datasets
- [ ] Save best fusion model

**Key Technologies:**
- Multi-headed attention layers
- Feature concatenation and fusion layers
- Ensemble learning techniques

---

### 3.2 Psychological State Mapping
- [ ] Define target psychological states:
  - **Primary:** Calm, Stressed, Anxious, Confident, Frustrated
  - **Secondary:** Depressed, Excited, Neutral
- [ ] Create mapping from emotions to psychological states
- [ ] Implement post-processing logic for state inference

---

## Phase 4: Real-Time System Development (Weeks 10-11)

### 4.1 Real-Time Pipeline
- [ ] Implement multi-threaded processing:
  - Thread 1: Video capture and body/face analysis
  - Thread 2: Audio capture and speech analysis
  - Thread 3: Fusion and prediction
- [ ] Optimize inference speed (target: 15-30 FPS)
- [ ] Add frame buffering and synchronization
- [ ] Implement queue-based architecture

### 4.2 User Interface Development
- [ ] Create GUI using Tkinter/PyQt/Streamlit
- [ ] Display real-time video feed
- [ ] Show detected psychological state with confidence scores
- [ ] Visualize pose keypoints and facial landmarks
- [ ] Add audio waveform visualization
- [ ] Implement session recording and logging

---

## Phase 5: Evaluation & Testing (Weeks 12-13)

### 5.1 Performance Evaluation
- [ ] Test on benchmark datasets
- [ ] Calculate metrics:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix
  - ROC-AUC curves
- [ ] Perform ablation studies (remove one modality at a time)
- [ ] Compare with baseline single-modality models

### 5.2 Real-World Testing
- [ ] Test with live subjects (team members, volunteers)
- [ ] Test under different lighting conditions
- [ ] Test with different backgrounds
- [ ] Test with various audio noise levels
- [ ] Collect feedback on accuracy and usability

---

## Phase 6: Documentation & Deployment (Weeks 14-15)

### 6.1 Documentation
- [ ] Write comprehensive README.md
- [ ] Document API and module interfaces
- [ ] Create user guide for the application
- [ ] Prepare technical documentation
- [ ] Write project report (following DTU guidelines)

### 6.2 Research & Presentation
- [ ] Prepare research paper draft
- [ ] Create presentation slides
- [ ] Practice demo scenarios
- [ ] Prepare for B.Tech Project-I presentation

### 6.3 Deployment (Optional)
- [ ] Create Docker container for easy deployment
- [ ] Deploy web demo using Flask/FastAPI + Streamlit
- [ ] Create GitHub repository with documentation
- [ ] Add demo videos and screenshots

---

## Technical Architecture Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT LAYER                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Webcam     │  │   Webcam     │  │  Microphone  │         │
│  │   (Video)    │  │   (Video)    │  │   (Audio)    │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
└─────────┼──────────────────┼──────────────────┼─────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PREPROCESSING LAYER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │Frame Extract │  │Face Detection│  │Audio Segment │         │
│  │& Resize      │  │& Alignment   │  │& Normalize   │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
└─────────┼──────────────────┼──────────────────┼─────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FEATURE EXTRACTION                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Body Agent   │  │  Face Agent  │  │ Speech Agent │         │
│  │ (Pose Est.)  │  │  (Landmarks) │  │  (MFCCs)     │         │
│  │  MediaPipe   │  │  CNN Model   │  │  Librosa     │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
└─────────┼──────────────────┼──────────────────┼─────────────────┘
          │                  │                  │
          │  Feature Vector  │  Feature Vector  │  Feature Vector
          │  (Body Pose)     │  (Face Emotion)  │  (Voice Tone)
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FUSION LAYER                                 │
│              ┌─────────────────────────┐                        │
│              │   Attention Mechanism   │                        │
│              │   Multimodal Fusion     │                        │
│              │   (Late/Hybrid Fusion)  │                        │
│              └───────────┬─────────────┘                        │
└──────────────────────────┼──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PREDICTION LAYER                                │
│              ┌─────────────────────────┐                        │
│              │   Classification Head   │                        │
│              │   Psychological State   │                        │
│              │   (Stress/Calm/Anxious) │                        │
│              └───────────┬─────────────┘                        │
└──────────────────────────┼──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT LAYER                                  │
│  ┌──────────────────────────────────────────────────┐          │
│  │ Psychological State + Confidence Score           │          │
│  │ Visualization (Pose, Face, Audio, State)         │          │
│  │ Logging & Analytics                              │          │
│  └──────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Milestones & Deliverables

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 2 | Setup Complete | Environment, datasets ready |
| 4 | Body Agent | Trained pose-based model |
| 6 | All Agents Ready | Three individual models |
| 9 | Fusion Complete | Integrated multimodal system |
| 11 | Real-time System | Working prototype |
| 13 | Evaluation Done | Performance metrics, report |
| 15 | Project Complete | Final presentation, documentation |

---

## Risk Management

| Risk | Mitigation Strategy |
|------|---------------------|
| Dataset availability issues | Use multiple open-source alternatives |
| Low accuracy in one modality | Focus on fusion and ensemble methods |
| Real-time performance lag | Optimize with multi-threading, model quantization |
| Hardware limitations | Use cloud GPU (Colab, Kaggle) for training |
| Integration complexity | Modular design, test each agent independently |

---

## Resources & Tools

### Development Tools
- **IDE:** VS Code, Cursor, PyCharm
- **Version Control:** Git, GitHub
- **Collaboration:** Google Colab, Jupyter Notebooks

### Libraries & Frameworks
- **Deep Learning:** PyTorch, TensorFlow, Keras
- **Computer Vision:** OpenCV, MediaPipe, Dlib
- **Audio:** Librosa, PyAudio, OpenSMILE
- **Data:** NumPy, Pandas, Scikit-learn
- **Visualization:** Matplotlib, Seaborn, Plotly
- **UI:** Streamlit, Tkinter, PyQt5

### Pre-trained Models
- Face: ResNet50, EfficientNet, VGGFace
- Speech: wav2vec 2.0, HuBERT, DeepSpeech
- Body: MediaPipe Pose, OpenPose

---

## Success Criteria

✅ **Minimum Viable Product (MVP):**
- Three working agents (Body, Face, Speech)
- Basic fusion mechanism
- Real-time detection at 10+ FPS
- Accuracy > 60% on multimodal test set

✅ **Target Goals:**
- Attention-based fusion
- Real-time detection at 20+ FPS
- Accuracy > 75% on benchmark datasets
- Published demo and documentation

✅ **Stretch Goals:**
- Research paper submission
- Web deployment
- Mobile app prototype
- Interpretability dashboard (explain predictions)

---

## Next Steps

1. **Immediate Actions (This Week):**
   - Set up development environment
   - Download initial datasets
   - Read key papers in detail
   - Divide work among team members

2. **Team Roles (Suggested):**
   - **Member 1 (Nitin):** Body Agent + Integration
   - **Member 2 (Rishek Raj):** Face Agent + UI
   - **Member 3 (Risheek Yadav):** Speech Agent + Fusion

3. **Weekly Sync:**
   - Schedule weekly meetings with supervisor
   - Document progress and blockers
   - Review code and models together

---

**Good luck with your project! 🚀**

# Project File Structure

```
psychological-state-detection/
│
├── README.md                          # Project overview and setup instructions
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore file
├── setup.py                          # Package setup file
├── LICENSE                           # Project license
│
├── config/                           # Configuration files
│   ├── __init__.py
│   ├── config.yaml                   # Main configuration (paths, hyperparameters)
│   ├── model_config.yaml            # Model-specific configurations
│   └── constants.py                  # Global constants
│
├── data/                             # Data directory (add to .gitignore)
│   ├── raw/                         # Raw datasets
│   │   ├── body/                    # Body language datasets
│   │   ├── face/                    # Facial expression datasets
│   │   ├── speech/                  # Speech emotion datasets
│   │   └── multimodal/              # Multimodal datasets
│   │
│   ├── processed/                   # Preprocessed data
│   │   ├── body_features/
│   │   ├── face_features/
│   │   └── speech_features/
│   │
│   └── annotations/                 # Labels and annotations
│       └── psychological_states.csv
│
├── models/                          # Trained model checkpoints
│   ├── body_agent/
│   │   ├── pose_model.pth
│   │   └── body_lstm.pth
│   │
│   ├── face_agent/
│   │   ├── face_cnn.pth
│   │   └── emotion_classifier.pth
│   │
│   ├── speech_agent/
│   │   ├── audio_model.pth
│   │   └── wav2vec_finetuned.pth
│   │
│   └── fusion/
│       ├── fusion_model.pth
│       └── attention_fusion.pth
│
├── src/                             # Source code
│   ├── __init__.py
│   │
│   ├── agents/                      # Individual agent modules
│   │   ├── __init__.py
│   │   │
│   │   ├── body_agent/             # Body language analysis
│   │   │   ├── __init__.py
│   │   │   ├── pose_estimator.py   # MediaPipe/OpenPose integration
│   │   │   ├── feature_extractor.py # Extract pose features
│   │   │   ├── temporal_model.py   # LSTM/GRU for temporal analysis
│   │   │   └── body_classifier.py  # Body state classifier
│   │   │
│   │   ├── face_agent/             # Facial expression analysis
│   │   │   ├── __init__.py
│   │   │   ├── face_detector.py    # Face detection and alignment
│   │   │   ├── landmark_extractor.py # Facial landmarks
│   │   │   ├── emotion_cnn.py      # CNN for facial emotions
│   │   │   └── micro_expression.py # Micro-expression detection
│   │   │
│   │   └── speech_agent/           # Speech emotion analysis
│   │       ├── __init__.py
│   │       ├── audio_processor.py  # Audio preprocessing
│   │       ├── feature_extractor.py # MFCC, pitch, energy extraction
│   │       ├── speech_model.py     # CNN-LSTM for speech
│   │       └── pretrained_wrapper.py # wav2vec/HuBERT wrapper
│   │
│   ├── fusion/                      # Fusion mechanisms
│   │   ├── __init__.py
│   │   ├── early_fusion.py         # Feature concatenation
│   │   ├── late_fusion.py          # Decision-level fusion
│   │   ├── attention_fusion.py     # Attention-based fusion
│   │   └── ensemble.py             # Ensemble methods
│   │
│   ├── data_processing/            # Data preprocessing
│   │   ├── __init__.py
│   │   ├── dataset_loader.py       # Load various datasets
│   │   ├── data_augmentation.py    # Augmentation techniques
│   │   ├── preprocessor.py         # General preprocessing
│   │   └── synchronizer.py         # Sync audio-video modalities
│   │
│   ├── models/                      # Model architectures
│   │   ├── __init__.py
│   │   ├── base_model.py           # Base model class
│   │   ├── cnn_models.py           # CNN architectures
│   │   ├── rnn_models.py           # LSTM/GRU models
│   │   ├── transformer_models.py   # Transformer architectures
│   │   └── multimodal_model.py     # End-to-end multimodal model
│   │
│   ├── training/                    # Training pipeline
│   │   ├── __init__.py
│   │   ├── trainer.py              # Main training loop
│   │   ├── validator.py            # Validation logic
│   │   ├── losses.py               # Custom loss functions
│   │   └── metrics.py              # Evaluation metrics
│   │
│   ├── inference/                   # Inference and prediction
│   │   ├── __init__.py
│   │   ├── predictor.py            # Batch prediction
│   │   ├── realtime_predictor.py   # Real-time inference
│   │   └── post_processor.py       # Post-processing predictions
│   │
│   └── utils/                       # Utility functions
│       ├── __init__.py
│       ├── logger.py               # Logging utilities
│       ├── visualization.py        # Visualization functions
│       ├── file_utils.py           # File I/O utilities
│       ├── model_utils.py          # Model save/load utilities
│       └── video_utils.py          # Video processing utilities
│
├── notebooks/                       # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_body_agent_dev.ipynb
│   ├── 03_face_agent_dev.ipynb
│   ├── 04_speech_agent_dev.ipynb
│   ├── 05_fusion_experiments.ipynb
│   └── 06_results_analysis.ipynb
│
├── scripts/                         # Standalone scripts
│   ├── download_datasets.py        # Download required datasets
│   ├── preprocess_data.py          # Run preprocessing pipeline
│   ├── train_body_agent.py         # Train body agent
│   ├── train_face_agent.py         # Train face agent
│   ├── train_speech_agent.py       # Train speech agent
│   ├── train_fusion.py             # Train fusion model
│   ├── evaluate_model.py           # Evaluate on test set
│   └── export_model.py             # Export for deployment
│
├── app/                             # Application code
│   ├── __init__.py
│   ├── main.py                     # Main application entry point
│   ├── gui/                        # GUI components
│   │   ├── __init__.py
│   │   ├── main_window.py         # Main window (Tkinter/PyQt)
│   │   ├── visualization_panel.py  # Real-time visualization
│   │   └── control_panel.py       # User controls
│   │
│   └── streamlit_app.py           # Streamlit web interface
│
├── tests/                           # Unit tests
│   ├── __init__.py
│   ├── test_body_agent.py
│   ├── test_face_agent.py
│   ├── test_speech_agent.py
│   ├── test_fusion.py
│   └── test_integration.py
│
├── docs/                            # Documentation
│   ├── architecture.md             # System architecture
│   ├── api_reference.md            # API documentation
│   ├── user_guide.md              # User guide
│   ├── development_guide.md       # Development setup
│   └── research_notes.md          # Research notes and findings
│
├── results/                         # Experimental results
│   ├── metrics/                    # Evaluation metrics
│   ├── plots/                      # Visualization plots
│   ├── logs/                       # Training logs
│   └── ablation_studies/           # Ablation study results
│
├── deployment/                      # Deployment files
│   ├── Dockerfile                  # Docker configuration
│   ├── docker-compose.yml          # Docker compose
│   ├── requirements_prod.txt       # Production dependencies
│   └── serve_model.py             # Model serving script
│
└── presentations/                   # Presentations and reports
    ├── project_proposal.pdf
    ├── mid_term_presentation.pptx
    ├── final_presentation.pptx
    └── final_report.pdf
```

---

## Detailed Component Descriptions

### 1. **config/**
Contains all configuration files to manage hyperparameters, paths, and settings without hardcoding.

**config.yaml example:**
```yaml
data:
  raw_data_path: "./data/raw/"
  processed_data_path: "./data/processed/"
  batch_size: 32
  
models:
  body_agent:
    model_type: "lstm"
    hidden_size: 256
    num_layers: 2
  
  face_agent:
    backbone: "resnet50"
    pretrained: true
    
  speech_agent:
    feature_type: "mfcc"
    n_mfcc: 40
    
training:
  epochs: 50
  learning_rate: 0.001
  optimizer: "adam"
```

---

### 2. **data/**
Organized storage for datasets. Keep raw data separate from processed features.

**Best practices:**
- Add `data/` to `.gitignore` (datasets are large)
- Use symbolic links if datasets are stored elsewhere
- Maintain a `data/README.md` with dataset sources and citations

---

### 3. **src/agents/**
Core modules for each specialized agent.

**body_agent/pose_estimator.py** - Example structure:
```python
import mediapipe as mp
import cv2
import numpy as np

class PoseEstimator:
    def __init__(self, model_complexity=1):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=0.5
        )
    
    def extract_pose(self, frame):
        """Extract pose landmarks from frame"""
        results = self.pose.process(frame)
        if results.pose_landmarks:
            landmarks = self._extract_landmarks(results.pose_landmarks)
            return landmarks
        return None
    
    def _extract_landmarks(self, pose_landmarks):
        """Convert landmarks to numpy array"""
        landmarks = []
        for landmark in pose_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)
```

---

### 4. **src/fusion/**
Implements different fusion strategies.

**attention_fusion.py** - Example:
```python
import torch
import torch.nn as nn

class AttentionFusion(nn.Module):
    def __init__(self, body_dim, face_dim, speech_dim, hidden_dim, num_classes):
        super().__init__()
        
        # Modality-specific projections
        self.body_proj = nn.Linear(body_dim, hidden_dim)
        self.face_proj = nn.Linear(face_dim, hidden_dim)
        self.speech_proj = nn.Linear(speech_dim, hidden_dim)
        
        # Attention weights
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, body_feat, face_feat, speech_feat):
        # Project to common space
        body = self.body_proj(body_feat)
        face = self.face_proj(face_feat)
        speech = self.speech_proj(speech_feat)
        
        # Stack modalities
        features = torch.stack([body, face, speech], dim=0)
        
        # Apply attention
        fused, _ = self.attention(features, features, features)
        fused = fused.mean(dim=0)  # Average over modalities
        
        # Classify
        output = self.classifier(fused)
        return output
```

---

### 5. **scripts/**
Standalone executable scripts for common tasks.

**train_fusion.py** - Example:
```python
import torch
from src.training.trainer import Trainer
from src.models.multimodal_model import MultimodalModel
from src.data_processing.dataset_loader import MultimodalDataset

def main():
    # Load configuration
    config = load_config('config/config.yaml')
    
    # Initialize model
    model = MultimodalModel(config)
    
    # Load datasets
    train_dataset = MultimodalDataset('data/processed/train/')
    val_dataset = MultimodalDataset('data/processed/val/')
    
    # Initialize trainer
    trainer = Trainer(model, train_dataset, val_dataset, config)
    
    # Train
    trainer.train()

if __name__ == "__main__":
    main()
```

---

### 6. **app/**
User-facing application code.

**main.py** - Real-time application:
```python
import cv2
import torch
from src.inference.realtime_predictor import RealtimePredictor

def main():
    # Initialize predictor
    predictor = RealtimePredictor(
        body_model_path='models/body_agent/body_lstm.pth',
        face_model_path='models/face_agent/face_cnn.pth',
        speech_model_path='models/speech_agent/audio_model.pth',
        fusion_model_path='models/fusion/fusion_model.pth'
    )
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Predict psychological state
        state, confidence = predictor.predict(frame)
        
        # Display result
        cv2.putText(frame, f"{state}: {confidence:.2f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Psychological State Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

---

### 7. **tests/**
Unit tests for each component.

```python
import unittest
from src.agents.body_agent.pose_estimator import PoseEstimator

class TestBodyAgent(unittest.TestCase):
    def setUp(self):
        self.estimator = PoseEstimator()
    
    def test_pose_extraction(self):
        # Test with sample image
        frame = cv2.imread('tests/fixtures/sample_frame.jpg')
        landmarks = self.estimator.extract_pose(frame)
        self.assertIsNotNone(landmarks)
        self.assertEqual(landmarks.shape[0], 33)  # 33 pose landmarks
```

---

### 8. **deployment/**
Production deployment files.

**Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements_prod.txt .
RUN pip install --no-cache-dir -r requirements_prod.txt

COPY src/ ./src/
COPY models/ ./models/
COPY app/ ./app/

CMD ["python", "app/main.py"]
```

---

## Key Files to Create First

### 1. **requirements.txt**
```txt
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
mediapipe>=0.10.0
librosa>=0.10.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.28.0
pyyaml>=6.0
tqdm>=4.65.0
transformers>=4.30.0
```

### 2. **README.md**
Basic project information, setup instructions, and usage examples.

### 3. **.gitignore**
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/

# Data
data/
*.csv
*.h5
*.pkl

# Models
models/*.pth
models/*.pt

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
```

---

## Getting Started Checklist

1. ✅ Create project directory structure
2. ✅ Initialize Git repository
3. ✅ Create virtual environment
4. ✅ Install dependencies from requirements.txt
5. ✅ Set up config files
6. ✅ Download initial datasets
7. ✅ Implement basic data loaders
8. ✅ Start with one agent (recommend starting with Face Agent)
9. ✅ Build testing framework
10. ✅ Document as you go

---

**This structure provides modularity, scalability, and maintainability for your project!**

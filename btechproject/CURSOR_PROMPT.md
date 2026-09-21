# Cursor IDE Prompt for AI-Powered Psychological State Detection Project

## Project Context

You are helping develop an AI-powered system for detecting psychological states (stress, anxiety, calmness, confidence) by analyzing body language, facial expressions, and speech patterns. This is a B.Tech final year project at Delhi Technological University.

### Team Information
- **Team Members:** Nitin Kumar Jha, Rishek Raj, Risheek Yadav
- **Supervisor:** Dr. Dhirendra Kumar
- **Project:** MC401 – B. Tech Project-I

---

## System Architecture Overview

The system consists of four main components:

1. **Body Agent**: Analyzes human posture, gestures, and movement dynamics using pose estimation (MediaPipe/OpenPose)
2. **Face Agent**: Detects facial expressions and micro-movements using CNN models and facial landmark detection
3. **Speech Agent**: Extracts prosodic and spectral features (MFCCs, pitch, energy) from voice to assess tone and stress
4. **Fusion Agent**: Combines outputs from all three agents using attention-based multimodal fusion for final psychological state prediction

### Target Psychological States
- Primary: Calm, Stressed, Anxious, Confident, Frustrated
- Secondary: Depressed, Excited, Neutral

---

## Technical Stack

### Core Technologies
- **Deep Learning Framework:** PyTorch (primary) or TensorFlow
- **Computer Vision:** OpenCV, MediaPipe, Dlib
- **Audio Processing:** Librosa, PyAudio, OpenSMILE
- **Pre-trained Models:** 
  - Face: ResNet50, EfficientNet, VGGFace
  - Speech: wav2vec 2.0, HuBERT
  - Body: MediaPipe Pose

### Key Datasets
- **Body Language:** EmoPain, BoLD
- **Facial Expressions:** AffectNet, FER-2013, CK+, RAF-DB
- **Speech Emotion:** IEMOCAP, RAVDESS, MSP-IMPROV
- **Multimodal:** GEMEP, CMU-MOSEI

---

## Project Structure

```
psychological-state-detection/
├── config/                    # Configuration files
├── data/                      # Datasets (raw and processed)
├── models/                    # Trained model checkpoints
├── src/
│   ├── agents/               # Body, Face, Speech agents
│   ├── fusion/               # Fusion mechanisms
│   ├── data_processing/      # Data preprocessing
│   ├── models/               # Model architectures
│   ├── training/             # Training pipeline
│   ├── inference/            # Real-time inference
│   └── utils/                # Utilities
├── scripts/                   # Standalone scripts
├── app/                       # Real-time application
├── tests/                     # Unit tests
└── notebooks/                 # Jupyter notebooks
```

---

## Development Guidelines

### Code Style
- Follow PEP 8 conventions
- Use type hints for function parameters and returns
- Write comprehensive docstrings (Google style)
- Keep functions focused and modular
- Use meaningful variable names

### Best Practices
1. **Modular Design:** Each agent should be independently testable
2. **Configuration-Driven:** Use YAML configs instead of hardcoded values
3. **Logging:** Implement proper logging for debugging
4. **Versioning:** Track model versions and experiments
5. **Documentation:** Document all major functions and classes
6. **Error Handling:** Implement robust exception handling
7. **Testing:** Write unit tests for critical components

### Performance Considerations
- Target real-time performance: 15-30 FPS
- Use multi-threading for parallel processing
- Implement frame buffering and synchronization
- Optimize model inference (quantization if needed)
- Profile code to identify bottlenecks

---

## Implementation Phases

### Phase 1: Setup & Foundation (Weeks 1-2)
**Current Priority**

**Tasks:**
1. Set up project structure with all directories
2. Create configuration files (config.yaml, model_config.yaml)
3. Set up requirements.txt with all dependencies
4. Initialize Git repository with proper .gitignore
5. Download and organize datasets
6. Implement basic data loaders

**Deliverables:**
- Complete project skeleton
- Working virtual environment
- Dataset download scripts
- Basic data visualization notebook

---

### Phase 2: Body Agent Development (Weeks 3-4)

**Tasks:**
1. Implement pose estimation using MediaPipe
2. Extract pose features (keypoints, angles, distances)
3. Create temporal feature extraction (motion vectors)
4. Design and train LSTM/GRU model for sequence modeling
5. Evaluate on body language datasets

**Key Files to Create:**
```
src/agents/body_agent/
├── pose_estimator.py          # MediaPipe integration
├── feature_extractor.py       # Extract pose features
├── temporal_model.py          # LSTM/GRU architecture
└── body_classifier.py         # Classification head
```

**Example Implementation Pattern:**
```python
class PoseEstimator:
    """Extract body pose landmarks from video frames"""
    
    def __init__(self, model_complexity: int = 1):
        """Initialize MediaPipe Pose"""
        pass
    
    def extract_pose(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract 33 pose landmarks from frame"""
        pass
    
    def extract_features(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Extract geometric features from landmarks"""
        # Calculate angles, distances, ratios
        pass
```

---

### Phase 3: Face Agent Development (Weeks 4-5)

**Tasks:**
1. Implement facial landmark detection (468 points with MediaPipe)
2. Extract facial features and Action Units
3. Fine-tune CNN on facial emotion datasets
4. Add attention mechanism for micro-expressions
5. Evaluate on FER datasets

**Key Files to Create:**
```
src/agents/face_agent/
├── face_detector.py           # Face detection and alignment
├── landmark_extractor.py      # Facial landmark extraction
├── emotion_cnn.py            # CNN architecture
└── micro_expression.py       # Micro-expression detection
```

**Model Architecture Guidance:**
- Use transfer learning (ResNet50 or EfficientNet-B0 pretrained on ImageNet)
- Fine-tune last few layers on emotion datasets
- Input size: 224x224 RGB images
- Output: Emotion probabilities (7-8 classes)

---

### Phase 4: Speech Agent Development (Weeks 5-6)

**Tasks:**
1. Implement audio preprocessing pipeline
2. Extract acoustic features (MFCCs, pitch, energy, formants)
3. Train audio classification model (CNN-LSTM)
4. Fine-tune wav2vec 2.0 for emotion recognition
5. Evaluate on speech emotion datasets

**Key Files to Create:**
```
src/agents/speech_agent/
├── audio_processor.py         # Audio preprocessing
├── feature_extractor.py       # Extract MFCCs, pitch, etc.
├── speech_model.py           # CNN-LSTM architecture
└── pretrained_wrapper.py     # wav2vec/HuBERT wrapper
```

**Feature Extraction Guidance:**
```python
def extract_audio_features(audio_path: str) -> Dict[str, np.ndarray]:
    """Extract comprehensive audio features"""
    features = {
        'mfcc': librosa.feature.mfcc(n_mfcc=40),
        'pitch': librosa.piptrack(),
        'energy': librosa.feature.rms(),
        'zcr': librosa.feature.zero_crossing_rate(),
        'spectral_centroid': librosa.feature.spectral_centroid()
    }
    return features
```

---

### Phase 5: Fusion Agent Development (Weeks 7-8)

**Tasks:**
1. Implement early fusion (feature concatenation)
2. Implement late fusion (decision-level voting)
3. Implement attention-based hybrid fusion
4. Train end-to-end multimodal model
5. Perform ablation studies

**Key Files to Create:**
```
src/fusion/
├── early_fusion.py            # Concatenate features
├── late_fusion.py            # Ensemble predictions
├── attention_fusion.py       # Attention mechanism
└── ensemble.py               # Weighted ensemble
```

**Fusion Architecture:**
```python
class AttentionFusion(nn.Module):
    """Attention-based multimodal fusion"""
    
    def __init__(self, body_dim, face_dim, speech_dim, num_classes):
        super().__init__()
        # Project each modality to common space
        # Apply multi-head attention
        # Classification head
        pass
    
    def forward(self, body_feat, face_feat, speech_feat):
        # Project features
        # Apply attention to learn modality importance
        # Fuse and classify
        pass
```

---

### Phase 6: Real-Time System (Weeks 9-10)

**Tasks:**
1. Implement multi-threaded real-time pipeline
2. Optimize inference speed
3. Create GUI using Streamlit or PyQt5
4. Add visualization for all modalities
5. Implement session logging

**Real-Time Pipeline Architecture:**
```python
class RealtimePredictor:
    """Real-time psychological state detection"""
    
    def __init__(self):
        self.body_agent = BodyAgent()
        self.face_agent = FaceAgent()
        self.speech_agent = SpeechAgent()
        self.fusion_agent = FusionAgent()
        
        # Threading for parallel processing
        self.video_thread = None
        self.audio_thread = None
    
    def start(self):
        """Start real-time detection"""
        # Start video capture thread
        # Start audio capture thread
        # Process and fuse predictions
        pass
```

---

## Specific Coding Instructions

### When Creating Each Module:

1. **Start with Interface Definition**
   ```python
   class AgentInterface:
       """Base interface for all agents"""
       
       def preprocess(self, input_data): pass
       def extract_features(self, preprocessed): pass
       def predict(self, features): pass
   ```

2. **Implement Data Loaders First**
   ```python
   class MultimodalDataset(torch.utils.data.Dataset):
       """Load synchronized video and audio data"""
       
       def __init__(self, data_dir, annotations_file):
           # Load file paths and labels
           pass
       
       def __getitem__(self, idx):
           # Return body, face, speech features + label
           pass
   ```

3. **Use Configuration Files**
   ```python
   import yaml
   
   def load_config(config_path='config/config.yaml'):
       with open(config_path, 'r') as f:
           config = yaml.safe_load(f)
       return config
   ```

4. **Implement Proper Logging**
   ```python
   import logging
   
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
       handlers=[
           logging.FileHandler('logs/training.log'),
           logging.StreamHandler()
       ]
   )
   logger = logging.getLogger(__name__)
   ```

5. **Add Model Checkpointing**
   ```python
   def save_checkpoint(model, optimizer, epoch, path):
       torch.save({
           'epoch': epoch,
           'model_state_dict': model.state_dict(),
           'optimizer_state_dict': optimizer.state_dict(),
       }, path)
   ```

---

## Testing Strategy

### Unit Tests for Each Agent
```python
import unittest

class TestBodyAgent(unittest.TestCase):
    def setUp(self):
        self.agent = BodyAgent()
    
    def test_pose_extraction(self):
        frame = cv2.imread('tests/fixtures/sample.jpg')
        pose = self.agent.extract_pose(frame)
        self.assertIsNotNone(pose)
        self.assertEqual(pose.shape, (33, 3))
    
    def test_feature_extraction(self):
        landmarks = np.random.rand(33, 3)
        features = self.agent.extract_features(landmarks)
        self.assertIn('shoulder_angle', features)
```

### Integration Tests
```python
def test_end_to_end_pipeline():
    """Test complete pipeline from input to prediction"""
    video_path = 'tests/fixtures/test_video.mp4'
    audio_path = 'tests/fixtures/test_audio.wav'
    
    predictor = RealtimePredictor()
    state, confidence = predictor.predict(video_path, audio_path)
    
    assert state in ['Calm', 'Stressed', 'Anxious', 'Confident']
    assert 0 <= confidence <= 1
```

---

## Common Pitfalls to Avoid

1. **Synchronization Issues**: Ensure audio and video are properly aligned (use timestamps)
2. **Data Leakage**: Keep test set completely separate, don't use test data for any hyperparameter tuning
3. **Overfitting**: Use proper regularization, dropout, and data augmentation
4. **Class Imbalance**: Handle imbalanced datasets with weighted loss or oversampling
5. **Real-Time Performance**: Profile code early and optimize bottlenecks
6. **Memory Leaks**: Properly release resources (camera, audio stream, GPU memory)

---

## Debugging Tips

### For Body Agent:
- Visualize pose landmarks on frames to verify detection
- Check for missing landmarks (partial occlusion)
- Ensure consistent coordinate normalization

### For Face Agent:
- Test with various lighting conditions
- Handle multiple faces in frame (select closest/largest)
- Account for head pose variations

### For Speech Agent:
- Verify audio sample rate (16kHz or 44.1kHz)
- Check for silence detection and VAD
- Normalize audio amplitude

### For Fusion:
- Start with simple averaging before complex attention
- Perform ablation studies (remove one modality at a time)
- Visualize attention weights to understand modality importance

---

## Evaluation Metrics

Implement these metrics in `src/training/metrics.py`:

```python
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def evaluate_model(y_true, y_pred):
    """Calculate comprehensive evaluation metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    return metrics
```

---

## Example Workflow for Starting Development

### Step 1: Initial Setup
```bash
# Create project directory
mkdir psychological-state-detection
cd psychological-state-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Create directory structure
mkdir -p config data/{raw,processed} models src/{agents,fusion,data_processing,models,training,inference,utils} scripts app tests docs results

# Initialize Git
git init
```

### Step 2: Install Dependencies
```bash
pip install torch torchvision opencv-python mediapipe librosa numpy pandas scikit-learn matplotlib seaborn streamlit pyyaml tqdm transformers
pip freeze > requirements.txt
```

### Step 3: Create Configuration
```yaml
# config/config.yaml
project:
  name: "psychological-state-detection"
  version: "0.1.0"

data:
  raw_path: "data/raw/"
  processed_path: "data/processed/"
  batch_size: 32
  num_workers: 4

training:
  epochs: 50
  learning_rate: 0.001
  optimizer: "adam"
  device: "cuda"  # or "cpu"

models:
  body_agent:
    model_type: "lstm"
    hidden_size: 256
    num_layers: 2
  
  face_agent:
    backbone: "resnet50"
    pretrained: true
    num_classes: 7
  
  speech_agent:
    feature_type: "mfcc"
    n_mfcc: 40
```

### Step 4: Start with Body Agent
Begin implementing `src/agents/body_agent/pose_estimator.py` and test with sample videos.

---

## Questions to Ask Cursor During Development

### General Development
- "Create a base trainer class that handles training loop, validation, and checkpointing"
- "Implement a configuration loader that reads YAML files and validates parameters"
- "Design a logger utility that saves both to file and console with different log levels"

### Agent Development
- "Implement a MediaPipe pose estimator that extracts 33 body landmarks from video frames"
- "Create a feature extractor that calculates joint angles and distances from pose landmarks"
- "Design an LSTM model for temporal modeling of pose sequences"
- "Implement a facial emotion CNN using transfer learning from ResNet50"
- "Create an audio feature extractor that computes MFCCs, pitch, and energy"

### Fusion
- "Implement an attention-based fusion mechanism that combines body, face, and speech features"
- "Create an ensemble model that performs weighted averaging of agent predictions"
- "Design an early fusion strategy that concatenates features before classification"

### Real-Time System
- "Implement a multi-threaded real-time prediction system for video and audio"
- "Create a Streamlit interface for real-time psychological state detection"
- "Design a frame buffer system to synchronize audio and video streams"

### Testing & Evaluation
- "Create unit tests for the pose estimator class"
- "Implement evaluation metrics including accuracy, F1-score, and confusion matrix"
- "Design an ablation study framework to test each modality independently"

---

## Success Criteria

### Minimum Viable Product (MVP)
- ✅ Three working agents (Body, Face, Speech)
- ✅ Basic fusion mechanism (at least late fusion)
- ✅ Real-time detection at 10+ FPS
- ✅ Accuracy > 60% on test set

### Target Goals
- ✅ Attention-based fusion
- ✅ Real-time detection at 20+ FPS
- ✅ Accuracy > 75% on benchmark datasets
- ✅ Web demo deployment

### Stretch Goals
- ✅ Research paper submission
- ✅ Mobile app prototype
- ✅ Interpretability dashboard

---

## Additional Resources

### Papers to Reference
1. "FAF: A Novel Multimodal Emotion Recognition Approach" (Fang et al., 2022)
2. "Multimodal Emotion Recognition Based on Facial Expressions, Speech, and Body Gestures" (Jingjie Yan et al., 2024)
3. "A Multimodal Emotion Recognition System" (Kris Kraack, 2024)

### Useful Libraries Documentation
- PyTorch: https://pytorch.org/docs/
- MediaPipe: https://google.github.io/mediapipe/
- Librosa: https://librosa.org/doc/
- HuggingFace Transformers: https://huggingface.co/docs/transformers/

### Dataset Sources
- AffectNet: http://mohammadmahoor.com/affectnet/
- RAVDESS: https://zenodo.org/record/1188976
- IEMOCAP: https://sail.usc.edu/iemocap/

---

## Final Notes for Cursor

When developing this project:
1. **Prioritize modularity**: Each component should work independently
2. **Test frequently**: Write tests as you develop, not after
3. **Document thoroughly**: Future you will thank present you
4. **Profile early**: Identify performance bottlenecks before they become problems
5. **Version everything**: Track experiments, datasets, and models
6. **Ask for clarification**: If requirements are unclear, ask the team

**Remember**: This is a research project. Experimentation and iteration are expected. Don't aim for perfection in the first implementation—aim for working code that you can improve iteratively.

---

## Ready to Start?

Begin with Phase 1 (Setup & Foundation) and work through each phase systematically. Use this prompt as a reference guide throughout development.

**First command to give to Cursor:**
"Help me set up the initial project structure for psychological state detection with directories for config, data, models, src, scripts, app, tests, and docs. Create the basic files including requirements.txt, config.yaml, README.md, and .gitignore."

Good luck with your project! 🚀

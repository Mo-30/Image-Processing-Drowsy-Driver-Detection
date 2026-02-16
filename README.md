# 🚗 Drowsy Driver Detection System

A real-time computer vision system that detects driver drowsiness using traditional image processing techniques and facial landmark analysis. This project demonstrates how classical computer vision combined with MediaPipe can effectively monitor driver alertness and contribute to road safety.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.11.0-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.21-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies](#technologies)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributors](#contributors)
- [License](#license)

## 🎯 Overview

Driver drowsiness is a major cause of road accidents worldwide. This project implements a non-intrusive drowsiness detection system that monitors eye closure and yawning patterns using computer vision. The system processes video frames in real-time, calculates Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR), and triggers alerts when drowsiness is detected.

### Key Highlights
- ✅ **Real-time detection** using webcam feed
- ✅ **Image processing version** for static image analysis
- ✅ **Adaptive thresholding** for personalized detection
- ✅ **CLAHE enhancement** for varying lighting conditions
- ✅ **Audible alerts** when drowsiness detected
- ✅ **No deep learning required** - lightweight and efficient

## ✨ Features

### Version 1: Static Image Analysis
- Processes pre-recorded driver images
- Calculates person-specific EAR/MAR thresholds
- Handles varying lighting conditions
- Visualizes detection results with bounding boxes

### Version 2: Real-Time Detection
- Live webcam integration
- Instant drowsiness alerts with sound
- Enhanced image preprocessing pipeline
- Continuous monitoring with visual feedback
- Frame-by-frame analysis with low latency

## 🛠 Technologies

### Core Libraries
- **Python 3.11+** - Programming language
- **OpenCV** - Image processing and computer vision
- **MediaPipe** - Real-time face mesh detection and landmark extraction
- **NumPy** - Numerical computations and geometry calculations
- **Matplotlib** - Visualization (static version)
- **Pandas** - Data handling and analysis (static version)

### Image Enhancement Techniques
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization)
- **Gamma Correction** - Adaptive brightness adjustment
- **Histogram Equalization** - Global contrast enhancement
- **Grayscale Conversion** - Simplified processing

## 🏗 System Architecture

```
┌─────────────────┐
│  Video Input    │
│  (Webcam/Image) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocessing   │
│ - Grayscale     │
│ - CLAHE/Gamma   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ MediaPipe       │
│ Face Mesh       │
│ 478 Landmarks   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Calculate       │
│ EAR & MAR       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Threshold       │
│ Comparison      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Alert System    │
│ (Visual/Audio)  │
└─────────────────┘
```

## 💻 Installation

### Prerequisites
- Python 3.11 or higher
- Webcam (for real-time version)
- Windows/Linux/MacOS

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/drowsy-driver-detection.git
cd drowsy-driver-detection
```

2. **Install dependencies**
```bash
pip install opencv-python
pip install mediapipe
pip install numpy
pip install matplotlib
pip install pandas
```

Or install all at once:
```bash
pip install opencv-python mediapipe numpy matplotlib pandas
```

3. **For sound alerts (Windows)**
```python
# winsound comes pre-installed with Python on Windows
# For Linux/Mac, consider using alternatives like pygame or playsound
```

## 🚀 Usage

### Static Image Analysis (Version 1)

```python
# Open digital-image-drowsy.ipynb in Jupyter
jupyter notebook digital-image-drowsy.ipynb

# The notebook will:
# 1. Load image dataset
# 2. Apply preprocessing
# 3. Detect facial landmarks
# 4. Calculate EAR/MAR for each person
# 5. Determine adaptive thresholds
# 6. Classify and visualize results
```

### Real-Time Detection (Version 2)

```python
# Open Digital_Image__Real_Time_.ipynb in Jupyter
jupyter notebook Digital_Image__Real_Time_.ipynb

# Or run as a standalone script:
python drowsy_detection_realtime.py

# Press 'q' to quit the application
```

### Sample Code

```python
import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Calculate EAR
def calculate_ear(landmarks, eye_indices):
    # Vertical distances
    v1 = np.linalg.norm(landmarks[eye_indices[1]] - landmarks[eye_indices[5]])
    v2 = np.linalg.norm(landmarks[eye_indices[2]] - landmarks[eye_indices[4]])
    
    # Horizontal distance
    h = np.linalg.norm(landmarks[eye_indices[0]] - landmarks[eye_indices[3]])
    
    ear = (v1 + v2) / (2.0 * h)
    return ear

# Thresholds
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6
```

## 🔍 How It Works

### 1. Face Detection & Landmark Extraction
MediaPipe detects 478 facial landmarks in real-time, providing precise coordinates for eyes, mouth, and other facial features.

### 2. Eye Aspect Ratio (EAR)
```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```
- **p1, p4**: Horizontal eye corners
- **p2, p3, p5, p6**: Vertical eye points
- EAR ≈ 0.3 when eye is open
- EAR < 0.25 when eye is closed

### 3. Mouth Aspect Ratio (MAR)
```
MAR = (||upper_lip - lower_lip||) / (||left_corner - right_corner||)
```
- Higher MAR indicates yawning
- MAR > 0.6 suggests drowsiness

### 4. Image Preprocessing

#### CLAHE (Contrast Limited Adaptive Histogram Equalization)
```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(gray)
```
- Enhances local contrast
- Prevents over-amplification of noise
- Essential for dark/low-light conditions

#### Gamma Correction
```python
gamma = 1.3
inv_gamma = 1.0 / gamma
table = np.array([((i / 255.0) ** inv_gamma) * 255 
                  for i in range(256)]).astype("uint8")
corrected = cv2.LUT(gray, table)
```
- Adjusts overall brightness
- Balances lighting dynamically

### 5. Adaptive Thresholding
Instead of fixed thresholds, the system:
- Calculates average EAR/MAR for each person
- Determines person-specific baselines
- Improves accuracy across different individuals

### 6. Alert System
```python
if ear < EAR_THRESHOLD or mar > MAR_THRESHOLD:
    drowsy_frame_count += 1
    if drowsy_frame_count >= CONSECUTIVE_FRAMES:
        # ALERT: Driver is drowsy!
        winsound.Beep(1000, 300)  # Sound alert
        cv2.putText(frame, "DROWSY!", ...)
```

## 📁 Project Structure

```
drowsy-driver-detection/
│
├── digital-image-drowsy.ipynb          # Version 1: Static image analysis
├── Digital_Image__Real_Time_.ipynb     # Version 2: Real-time detection
├── Drowsy_Driver_detection.pdf         # Project presentation
├── README.md                            # This file
│
├── datasets/                            # (Not included in repo)
│   ├── alert/                          # Alert driver images
│   └── drowsy/                         # Drowsy driver images
│
├── results/                             # Output visualizations
│   ├── processed_images/
│   └── detection_metrics/
│
└── requirements.txt                     # Python dependencies
```

## 📊 Results

### Performance Metrics
- **Detection Accuracy**: High accuracy with adaptive thresholding
- **Processing Speed**: 20-30 FPS on standard webcam
- **False Positive Rate**: Minimized through consecutive frame checks
- **Lighting Robustness**: Effective in various lighting conditions

### Challenges Overcome
✅ **Varying Lighting Conditions**
- Solution: CLAHE + Gamma correction

✅ **Individual Differences**
- Solution: Person-specific adaptive thresholds

✅ **False Positives from Blinks**
- Solution: Consecutive frame counting (typically 3-5 frames)

✅ **Landmark Detection Failures**
- Solution: Preprocessing enhancement improves MediaPipe accuracy

## 🔮 Future Improvements

### Short-term
- [ ] Add head pose estimation for nodding detection
- [ ] Implement temporal smoothing for EAR/MAR values
- [ ] Create dataset logging for model training
- [ ] Add configuration file for customizable thresholds

### Long-term
- [ ] Deep learning model integration (CNN/LSTM)
- [ ] Multi-modal detection (steering patterns, lane deviation)
- [ ] Mobile app deployment
- [ ] Cloud-based alert system for fleet management
- [ ] Integration with vehicle CAN bus
- [ ] Fatigue level scoring (0-100)

## 👥 Contributors

This project was developed as part of a Digital Image Processing course:

- **Mohamed Sherif** - Implementation & Documentation
- **Mohamed Osama** - Testing & Validation
- **Ali Zorkany** - Research & Analysis
- **Yousef el Ghazali** - Dataset Preparation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe** team at Google for the excellent face mesh solution
- **OpenCV** community for comprehensive computer vision tools
- Research papers on drowsiness detection for methodology guidance
- Digital Image Processing course instructors

## 📚 References

1. Soukupová, T., & Čech, J. (2016). Real-Time Eye Blink Detection using Facial Landmarks. *21st Computer Vision Winter Workshop*
2. Google MediaPipe Face Mesh: https://google.github.io/mediapipe/solutions/face_mesh
3. OpenCV CLAHE Documentation: https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html


---

**Made with ❤️ for road safety**

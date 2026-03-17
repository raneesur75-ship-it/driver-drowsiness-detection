# 🚗 Driver Drowsiness Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.6%2B-green)](https://opencv.org)

&gt; Real-time driver drowsiness detection using CNN (MobileNetV2) and Computer Vision.
&gt; **96.5% accuracy** | **4-class classification** | **Real-time alarm system**

## 🎯 Features

- **Real-time Detection**: Webcam-based monitoring with MediaPipe Face Mesh
- **Dual Analysis**: Eye Aspect Ratio (EAR) + Mouth Aspect Ratio (MAR) + CNN classification
- **AI-Powered**: MobileNetV2 CNN with 96.5% test accuracy
- **Smart Augmentation**: Stable Diffusion for synthetic training data generation
- **Professional UI**: Live graphs, fatigue score, status indicators
- **Multi-trigger Alarm**: Sound alerts for closed eyes, yawning, head nodding

## 📊 Performance Metrics

| Class | Accuracy | Notes |
|-------|----------|-------|
| Closed_Eyes | ~99% | Almost perfect detection |
| Open_Eyes | ~99% | Clear pattern recognition |
| No_Yawn | ~95% | Good distinction |
| Yawn | 95.8% | High recall for safety |

**Overall Test Accuracy: 96.50%** on 572 test images

## 📚 Dataset

- **Total images**: 2,845  
- **Split**: 80% training + 20% testing (572 test images)  
- **4 Classes**: Closed_Eyes, Open_Eyes, No_Yawn, Yawn (balanced)

**Sources**:
- Public Kaggle datasets  
- Self-collected photos/videos from online sources  
- Augmented with OpenCV + Stable Diffusion (synthetic images)

Full dataset is large, so not on GitHub. Sample images are in `data/sample/`.  
See details → [data/README.md](data/README.md)

## 💻 Installation & How to Run

### 1. Clone the repo
```bash
git clone https://github.com/raneesur75-ship-it/driver-drowsiness-detection-using-deep-learning.git
cd driver-drowsiness-detection-using-deep-learning
```
### 2. Install packages

```
pip install -r requirements.txt

```
### 3. Run the real-time demo

```
python src/realtime_detector.py

```


###Note: This is the correct file name (realtime_detector.py).
Your webcam will open instantly with live detection, fatigue score, graphs, and alarm sound!





## 🏗️ System Architecture

The system uses a **hybrid approach** (fast rule-based + deep learning) for reliable real-time drowsiness detection.

### Flow:
1. **Webcam Input** → Live video frames  
2. **MediaPipe Face Mesh** → 468 facial landmarks  
3. **EAR + MAR Calculation** → Quick eye/yawn detection  
4. **MobileNetV2 CNN** → Classifies eyes & mouth (96.5% accurate)  
5. **Decision Fusion** → Combines everything into Fatigue Score  
6. **Alert System** → Alarm + UI graphs if drowsy  

**Why it works**: Lightweight (MobileNetV2 runs on laptop/mobile), very accurate, and almost no false alarms.

(Architecture diagram coming soon — we will add screenshot here next!)

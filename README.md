# 🚗 Driver Drowsiness Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.6%2B-green)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-purple)](https://mediapipe.dev)

> Real-time driver drowsiness detection using CNN (MobileNetV2) and Computer Vision.  
> **96.5% accuracy** | **4-class classification** | **Real-time alarm system**

## 📸 Demo
<p align="center">
  <img src="driver-drowsiness-detection-demo.gif" alt="Demo: Real-time drowsiness detection" width="700"/>
</p>

## 📸 Demo Screenshots

Here’s how the system looks in real-time:

<p align="center">
  <img src="screenshots/capture_1773825174.png" alt="Live Detection - Eyes Closed & Alarm" width="45%">
  <img src="screenshots/capture_1773825177.png" alt="Fatigue Score & Live Graphs" width="45%">
</p>

<p align="center">
  <img src="screenshots/capture_1773825185.png" alt="Alarm Triggering" width="60%">
</p>

## 🎯 Why This Project Matters

Drowsy driving causes thousands of road accidents every year.  
This system detects early fatigue signs and alerts the driver in real-time — a low-cost, lightweight safety solution that can save lives.

## 🔥 Key Features

- **Real-time Detection**: Webcam monitoring with MediaPipe Face Mesh (468 landmarks)
- **Hybrid Intelligence**: Eye Aspect Ratio (EAR) + Mouth Aspect Ratio (MAR) + MobileNetV2 CNN
- **AI-Powered**: 96.5% test accuracy on 4 classes (Closed_Eyes, Open_Eyes, No_Yawn, Yawn)
- **Smart Augmentation**: Stable Diffusion for synthetic data
- **Professional UI**: Live fatigue score (0–100%), real-time graphs, status indicators
- **Multi-trigger Alarm**: Eye closure, yawning, head nodding + sound alerts

## 📊 Performance Metrics

| Class       | Accuracy | Notes                     |
|-------------|----------|---------------------------|
| Closed_Eyes | ~99%     | Almost perfect detection  |
| Open_Eyes   | ~99%     | Clear pattern recognition |
| No_Yawn     | ~95%     | Good distinction          |
| Yawn        | 95.8%    | High recall for safety    |

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
```
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
### Note:
Your webcam 🎥 will open instantly with live detection, fatigue score, graphs, and alarm sound!


## 🏗️ System Architecture

```mermaid
graph TD
    A[Webcam Input] --> B[MediaPipe Face Mesh<br/>468 landmarks]
    B --> C[EAR + MAR Calculation]
    B --> D[MobileNetV2 CNN<br/>Eye/Mouth Classification]
    C --> E[Decision Fusion<br/>Fatigue Score]
    D --> E
    E --> F{Score > Threshold?}
    F -->|Yes| G[Multi-Trigger Alarm<br/>+ UI Graphs]
    F -->|No| H[Continue Monitoring]
```

```
Webcam → Face Mesh → EAR/MAR → CNN → Fatigue Score → Alarm
```
The system uses a hybrid approach (fast rule-based + deep learning) for reliable real-time drowsiness detection.

### 💡 Why This Works

Lightweight → runs on laptop

Hybrid approach → reduces false alarms

Real-time processing → practical deployment

## 🚀 Key Innovations

✅ Hybrid CNN + EAR + MAR system

✅ AI-based data augmentation using Stable Diffusion

✅ Real-time fatigue scoring (not just classification)

✅ Multi-condition alarm logic

## 🔮 Future Improvements

✅Night vision (infrared camera)

✅Mobile app (Android/iOS)

✅Cloud-based monitoring system

✅Voice-based alerts.



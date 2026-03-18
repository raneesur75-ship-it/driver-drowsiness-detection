# 🚗 Driver Drowsiness Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.6%2B-green)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-purple)](https://mediapipe.dev)

> Real-time driver drowsiness detection using CNN (MobileNetV2) and Computer Vision.  
> **96.5% accuracy** | **4-class classification** | **Real-time alarm system**

## 📸 Demo (Add screenshots/GIF here — very important for clients)

![Live Detection](screenshots/demo.gif)  
![Fatigue Score & Graphs](screenshots/graphs.png)  
![Alarm Triggering](screenshots/alarm.png)

*(Upload 2-3 screenshots/GIFs in the `screenshots/` folder and replace the links above)*

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
2. Install packages
```

pip install -r requirements.txt

```
3. Run the real-time demo
```

python src/realtime_detector.py

```
Note: Your webcam will open instantly with live detection, fatigue score, graphs, and alarm sound! System ArchitectureThe system uses a hybrid approach (fast rule-based + deep learning) for reliable real-time drowsiness detection.Flow:Webcam Input → Live video frames  
MediaPipe Face Mesh → 468 facial landmarks  
EAR + MAR Calculation → Quick eye/yawn detection  
MobileNetV2 CNN → Classifies eyes & mouth (96.5% accurate)  
Decision Fusion → Combines everything into Fatigue Score  
Alert System → Alarm + UI graphs if drowsy

Why it works: Lightweight (runs on laptop/mobile), very accurate, and almost no false alarms. Key InnovationsHybrid CNN + EAR + MAR system  
AI-based data augmentation using Stable Diffusion  
Real-time fatigue scoring (not just classification)  
Multi-condition alarm logic

 Future ImprovementsNight vision support (infrared camera)  
Mobile app (Android/iOS deployment)  
Cloud-based monitoring  
Voice-based alerts


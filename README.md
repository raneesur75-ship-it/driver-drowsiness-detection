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

## Dataset

- **Total images**: 2,845  
- **Split**: 80% training (2,273 images) + 20% testing (572 images)  
- **4 Classes**: Closed_Eyes, Open_Eyes, No_Yawn, Yawn (balanced)

**Sources**:
- Public datasets from Kaggle
- Self-collected photos and videos from online sources
- Augmented using OpenCV (frame extraction) + Stable Diffusion (synthetic images)

Full dataset is too big for GitHub, but you can see sample images inside the `data/sample/` folder.  
More details → [data/README.md](data/README.md)

## 🏗️ System Architecture

The system follows a hybrid approach combining deep learning classification with traditional computer vision metrics for reliable real-time drowsiness detection.

### Main Components & Flow:
1. **Video Input**  
   Webcam captures live video frames.

2. **Face Detection & Landmarks**  
   Uses MediaPipe Face Mesh to detect face and extract 468 facial landmarks.

3. **Feature Extraction**  
   - Eye Aspect Ratio (EAR): Measures if eyes are closed (low EAR = drowsy).  
   - Mouth Aspect Ratio (MAR): Detects yawning (high MAR = yawn).  
   - These act as fast, rule-based triggers.

4. **Deep Learning Classification**  
   MobileNetV2 model classifies cropped eye/mouth regions into:  
   - Closed_Eyes / Open_Eyes  
   - Yawn / No_Yawn  
   (Trained on augmented dataset → 96.5% accuracy)

5. **Decision Fusion**  
   Combines:  
   - EAR/MAR thresholds (quick detection)  
   + CNN predictions (more accurate but slower)  
   → Computes a "Fatigue Score". If score > threshold → trigger alarm + warning.

6. **Output / UI**  
   - Real-time overlay on video  
   - Live graphs (EAR/MAR over time)  
   - Audio/visual alerts (beep + red screen)

### Future Diagram
(Architecture diagram coming soon — will show data flow from input to alert)

This hybrid design makes the system lightweight (MobileNetV2 runs on mobile/edge), accurate, and reduces false alarms.

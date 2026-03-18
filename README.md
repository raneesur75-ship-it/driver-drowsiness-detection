# 🚗 Driver Drowsiness Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.6%2B-green)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

&gt; Real-time driver drowsiness detection using CNN (MobileNetV2) and Computer Vision.
&gt; **96.5% accuracy** | **4-class classification** | **Real-time alarm system**

![Demo](docs/images/demo_screenshot.png)  &lt;!-- Add screenshot later --&gt;

## 🎯 Features

- **Real-time Detection**: Webcam-based monitoring with 30+ FPS
- **Dual Analysis**: Eye Aspect Ratio (EAR) + Mouth Aspect Ratio (MAR)
- **AI-Powered**: MobileNetV2 CNN with 96.5% test accuracy
- **Smart Augmentation**: Stable Diffusion for synthetic training data
- **Professional UI**: Live graphs, fatigue score, status indicators
- **Multi-trigger Alarm**: Sound alert for closed eyes, yawning, head nodding

## 📊 Performance

| Class | Accuracy | Notes |
|-------|----------|-------|
| Closed_Eyes | ~99% | Almost perfect detection |
| Open_Eyes | ~99% | Clear pattern recognition |
| No_Yawn | ~95% | Good distinction |
| Yawn | 95.8% | High recall for safety |

**Overall Test Accuracy: 96.50%**

## 🏗️ Architecture

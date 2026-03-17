'''"""
Real-time Driver Drowsiness Detection System
Combines CNN classification with geometric analysis (EAR/MAR) and head pose detection.
"""

import os
import time
import math
import threading
import winsound
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from scipy.spatial import distance as dist


# Configuration
MODEL_PATH = "./models/best_model.keras"
CLASS_NAMES = ['Closed_Eyes', 'No_Yawn', 'Open_Eyes', 'Yawn']
YAWN_INDEX = 3

# Detection thresholds
EAR_THRESH = 0.25
MAR_THRESH = 0.5
MAR_HIGH_THRESH = 0.7
CNN_YAWN_CONFIDENCE = 0.72
HEAD_PITCH_THRESH = 15.0

# Alert timing
WARNING_FRAMES = 5
ALARM_FRAMES = 15
COOLDOWN_FRAMES = 3

# Display settings
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 650

# Color scheme
COLORS = {
    'bg': (10, 10, 20),
    'panel': (20, 20, 40),
    'border': (255, 255, 255),
    'text': (200, 200, 200),
    'accent': (255, 100, 0),
    'alert': (0, 0, 255),
    'warning': (0, 140, 255),
    'safe': (0, 255, 0),
    'info': (0, 255, 255),
    'eye_open': (0, 255, 0),
    'eye_closed': (0, 0, 255),
    'mouth_open': (0, 255, 0),
    'mouth_normal': (255, 100, 0),
    'no_face': (128, 128, 128),
    'head_pose': (255, 255, 0)
}


class AlarmSystem:
    """Professional alarm with configurable beep patterns."""
    
    def __init__(self):
        self.active = False
        self.stop_flag = False
        self.thread = None
        self.pattern = [(800, 200), (0, 100), (1000, 200), (0, 100), (1200, 400)]
        
    def start(self):
        if self.active:
            return
        self.active = True
        self.stop_flag = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        
    def _loop(self):
        while self.active and not self.stop_flag:
            for freq, duration in self.pattern:
                if self.stop_flag:
                    break
                if freq > 0:
                    winsound.Beep(freq, duration)
                else:
                    time.sleep(duration / 1000.0)
            time.sleep(0.1)
                    
    def stop(self):
        self.stop_flag = True
        self.active = False
        if self.thread:
            self.thread.join(timeout=0.5)
            self.thread = None


class FatigueTracker:
    """Tracks fatigue indicators over time."""
    
    def __init__(self):
        self.ear_history = deque(maxlen=100)
        self.mar_history = deque(maxlen=30)
        self.blink_times = deque(maxlen=20)
        self.yawn_active = False
        self.yawn_cooldown = 0
        self.yawn_score = 0.0
        self.fatigue_score = 0.0
        self.blink_rate = 0
        self.consecutive_closed = 0
        self.head_pose_history = deque(maxlen=30)
        self.head_nod_count = 0

    def update(self, ear, mar, yawn_confidence, eyes_closed, head_pose=None):
        current_time = time.time()
        self.ear_history.append(ear)
        self.mar_history.append(mar)
        
        if head_pose:
            pitch, yaw, roll = head_pose
            self.head_pose_history.append(abs(pitch))
            if len(self.head_pose_history) >= 10:
                recent_pitch = list(self.head_pose_history)[-10:]
                if np.var(recent_pitch) > 50:
                    self.head_nod_count += 1
        
        if not eyes_closed and yawn_confidence > 0.5:
            self.yawn_score = min(self.yawn_score + 0.3, 1.0)
        else:
            self.yawn_score = max(self.yawn_score - 0.15, 0.0)
        
        if self.yawn_score > 0.6:
            self.yawn_active = True
            self.yawn_cooldown = 3
        elif self.yawn_cooldown > 0:
            self.yawn_cooldown -= 1
            self.yawn_active = True
        else:
            self.yawn_active = False
        
        if eyes_closed:
            self.consecutive_closed += 1
        else:
            if 1 <= self.consecutive_closed <= 5:
                self.blink_times.append(current_time)
            self.consecutive_closed = 0
        
        self.blink_rate = len([t for t in self.blink_times if current_time - t < 60])
        self.fatigue_score = self._calculate_fatigue(head_pose)
        return self.fatigue_score
    
    def _calculate_fatigue(self, head_pose=None):
        score = 0
        if len(self.ear_history) >= 30:
            recent = list(self.ear_history)[-30:]
            closed_frames = sum(1 for e in recent if e < EAR_THRESH)
            perclos = (closed_frames / len(recent)) * 100
            if perclos > 10:
                score += min(perclos, 40)
        if self.blink_rate > 25:
            score += 20
        elif self.blink_rate < 8:
            score += 25
        if self.yawn_active:
            score += 35
        if head_pose:
            pitch, yaw, roll = head_pose
            if abs(pitch) > HEAD_PITCH_THRESH:
                score += 15
            if self.head_nod_count > 3:
                score += 20
        return min(score, 100)


class FaceDetector:
    """MediaPipe face mesh detection and analysis."""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark indices
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.MOUTH_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
        
        # Head pose estimation points
        self.model_points = np.array([
            (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
        ])
        self.pose_landmarks = [1, 152, 263, 33, 61, 291]

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.face_mesh.process(rgb)

    def calculate_ear(self, landmarks, eye_indices, frame_w, frame_h):
        try:
            points = [(landmarks[i].x * frame_w, landmarks[i].y * frame_h) for i in eye_indices]
            A = dist.euclidean(points[1], points[5])
            B = dist.euclidean(points[2], points[4])
            C = dist.euclidean(points[0], points[3])
            return (A + B) / (2.0 * C) if C > 0 else 0.3
        except:
            return 0.3

    def calculate_mar(self, landmarks, frame_w, frame_h):
        try:
            top = (landmarks[11].y + landmarks[12].y) / 2 * frame_h
            bottom = (landmarks[14].y + landmarks[16].y) / 2 * frame_h
            left = landmarks[61].x * frame_w
            right = landmarks[291].x * frame_w
            return abs(bottom - top) / abs(right - left) if abs(right - left) > 0 else 0.0
        except:
            return 0.0

    def estimate_head_pose(self, landmarks, frame_w, frame_h):
        try:
            points = np.array([(landmarks[i].x * frame_w, landmarks[i].y * frame_h) 
                              for i in self.pose_landmarks], dtype="double")
            focal = frame_w
            center = (frame_w / 2, frame_h / 2)
            camera_matrix = np.array([[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]], dtype="double")
            dist_coeffs = np.zeros((4, 1))
            
            success, rvec, tvec = cv2.solvePnP(self.model_points, points, camera_matrix, dist_coeffs)
            if not success:
                return (0.0, 0.0, 0.0)
            
            rot_matrix, _ = cv2.Rodrigues(rvec)
            proj_matrix = np.hstack((rot_matrix, np.zeros((3, 1))))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
            return tuple(math.radians(max(min(e, 180), -180)) for e in euler_angles.flatten())
        except:
            return (0.0, 0.0, 0.0)

    def draw_landmarks(self, frame, landmarks, eyes_closed, yawning, left_ear=None, right_ear=None):
        h, w = frame.shape[:2]
        left_points, right_points, mouth_points = [], [], []
        
        # Collect points
        for idx in self.LEFT_EYE:
            x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            left_points.append((x, y))
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
            
        for idx in self.RIGHT_EYE:
            x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            right_points.append((x, y))
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
            
        for idx in self.MOUTH_OUTER:
            x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            mouth_points.append((x, y))

        # Draw eyes
        color = COLORS['eye_closed'] if eyes_closed else COLORS['eye_open']
        
        if len(left_points) == 6:
            for i in range(6):
                cv2.line(frame, left_points[i], left_points[(i+1)%6], color, 2)
            if left_ear is not None:
                lx, ly = left_points[0]
                cv2.putText(frame, f"L:{left_ear:.2f}", (lx-55, ly-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        if len(right_points) == 6:
            for i in range(6):
                cv2.line(frame, right_points[i], right_points[(i+1)%6], color, 2)
            if right_ear is not None:
                rx, ry = right_points[3]
                cv2.putText(frame, f"R:{right_ear:.2f}", (rx+20, ry-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Draw mouth
        if len(mouth_points) >= 4:
            pts = np.array(mouth_points, np.int32)
            if cv2.contourArea(pts) > 100:
                m_color = COLORS['mouth_open'] if yawning else COLORS['mouth_normal']
                cv2.polylines(frame, [pts], True, m_color, 2)
                if yawning:
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [pts], (0, 100, 0))
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        return frame

    def draw_head_pose(self, frame, landmarks, head_pose, frame_w, frame_h):
        if head_pose is None:
            return frame
        try:
            pitch, yaw, roll = head_pose
            pitch_deg = math.degrees(pitch)
            yaw_deg = math.degrees(yaw)
            nx, ny = int(landmarks[1].x * frame_w), int(landmarks[1].y * frame_h)
            
            color = COLORS['alert'] if abs(pitch_deg) > HEAD_PITCH_THRESH else COLORS['head_pose']
            cv2.putText(frame, f"P:{pitch_deg:.1f} Y:{yaw_deg:.1f}", (nx+25, ny-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            
            if abs(pitch_deg) > HEAD_PITCH_THRESH:
                cv2.circle(frame, (nx, ny), 5, COLORS['alert'], -1)
                cv2.circle(frame, (nx, ny), 8, COLORS['alert'], 2)
        except:
            pass
        return frame


class UIRenderer:
    """Renders the application interface."""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.canvas = None
        self.panel_x = 660
        self.panel_y = 80
        self.panel_w = 230
        self.panel_h = 480

    def create_base(self, frame, face_detected=True):
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.canvas[:] = COLORS['bg']
        
        # Video area
        video_w, video_h = 640, 480
        self.video_rect = (10, 80, video_w, video_h)
        
        if frame is not None:
            resized = cv2.resize(frame, (video_w, video_h))
            x, y, w, h = self.video_rect
            self.canvas[y:y+h, x:x+w] = resized
            cv2.rectangle(self.canvas, (x-2, y-2), (x+w+2, y+h+2), COLORS['border'], 2)
        
        if not face_detected:
            x, y, w, h = self.video_rect
            cv2.putText(self.canvas, "NO FACE DETECTED", (x+w//2-140, y+h//2), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.9, COLORS['alert'], 2)
        
        # Side panel
        overlay = self.canvas.copy()
        cv2.rectangle(overlay, (self.panel_x, self.panel_y), 
                     (self.panel_x+self.panel_w, self.panel_y+self.panel_h), 
                     COLORS['panel'], -1)
        cv2.addWeighted(overlay, 0.9, self.canvas, 0.1, 0, self.canvas)
        cv2.rectangle(self.canvas, (self.panel_x, self.panel_y), 
                     (self.panel_x+self.panel_w, self.panel_y+self.panel_h), 
                     COLORS['border'], 1)
        return self

    def draw_header(self, status, reason="", face_detected=True):
        overlay = self.canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, self.canvas, 0.3, 0, self.canvas)
        
        if not face_detected:
            color, text, bg = COLORS['no_face'], "NO FACE", (40, 40, 40)
        elif status == "ALARM":
            color, text, bg = COLORS['alert'], "!!! ALARM !!!", (0, 0, 80)
        elif status == "WARNING":
            color, text, bg = COLORS['warning'], "WARNING", (0, 40, 80)
        else:
            color, text, bg = COLORS['safe'], "AWAKE", (0, 60, 0)
        
        overlay = self.canvas.copy()
        cv2.rectangle(overlay, (10, 10), (320, 60), bg, -1)
        cv2.addWeighted(overlay, 0.8, self.canvas, 0.2, 0, self.canvas)
        cv2.rectangle(self.canvas, (10, 10), (320, 60), color, 2)
        cv2.putText(self.canvas, text, (25, 48), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)
        
        if reason:
            cv2.putText(self.canvas, reason, (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.putText(self.canvas, "DROWSINESS DETECTION", (self.width-280, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['border'], 1)
        cv2.putText(self.canvas, "SYSTEM v1.0", (self.width-280, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['accent'], 1)
        return self

    def draw_metrics(self, tracker, ear, mar, alert_frames, eyes_closed, 
                    left_ear, right_ear, alert_type, cnn_conf, yawn_score, 
                    cnn_class, head_pose=None, face_detected=True, ear_thresh=0.25):
        x = self.panel_x + 15
        y = self.panel_y + 30
        line_height = 26
        
        cv2.putText(self.canvas, "METRICS", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['border'], 2)
        y += line_height + 2
        
        if not face_detected:
            cv2.putText(self.canvas, "No face detected", (x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['no_face'], 2)
            return self
        
        # Eye status
        eye_status = "CLOSED" if eyes_closed else "OPEN"
        eye_color = COLORS['eye_closed'] if eyes_closed else COLORS['eye_open']
        cv2.putText(self.canvas, f"Eyes: {eye_status}", (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, eye_color, 2)
        y += line_height
        
        # Individual EAR values
        if left_ear is not None and right_ear is not None:
            cv2.putText(self.canvas, f"L: {left_ear:.2f} R: {right_ear:.2f}", (x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text'], 1)
            y += line_height
        
        cv2.putText(self.canvas, f"EAR avg: {ear:.2f}", (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text'], 1)
        y += line_height
        
        # Head pose
        if head_pose:
            pitch_deg = math.degrees(head_pose[0])
            yaw_deg = math.degrees(head_pose[1])
            color = COLORS['alert'] if abs(pitch_deg) > HEAD_PITCH_THRESH else COLORS['head_pose']
            cv2.putText(self.canvas, f"Head: P{pitch_deg:.0f} Y{yaw_deg:.0f}", (x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            y += line_height
            if tracker.head_nod_count > 0:
                cv2.putText(self.canvas, f"Nods: {tracker.head_nod_count}", (x, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS['warning'], 1)
                y += line_height
        
        # Fatigue score
        fat_color = COLORS['safe'] if tracker.fatigue_score < 30 else \
                   (COLORS['warning'] if tracker.fatigue_score < 60 else COLORS['alert'])
        cv2.putText(self.canvas, f"Fatigue: {tracker.fatigue_score:.0f}%", (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, fat_color, 1)
        y += line_height
        
        cv2.putText(self.canvas, f"MAR: {mar:.2f}", (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text'], 1)
        y += line_height
        cv2.putText(self.canvas, f"Threshold: {ear_thresh:.2f}", (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS['info'], 1)
        y += line_height
        cv2.putText(self.canvas, f"CNN: {cnn_class}", (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['accent'], 1)
        y += line_height
        cv2.putText(self.canvas, f"Y-Conf: {cnn_conf:.2f}", (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS['info'], 1)
        y += line_height
        cv2.putText(self.canvas, f"Blinks: {tracker.blink_rate}/min", (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text'], 1)
        y += line_height + 5
        
        # Alert progress bar
        progress = min(alert_frames / ALARM_FRAMES * 100, 100)
        bar_w = 180
        filled = int(bar_w * progress / 100)
        bar_color = COLORS['warning'] if progress < 100 else COLORS['alert']
        
        cv2.putText(self.canvas, f"Progress: {progress:.0f}%", (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text'], 1)
        y += 18
        cv2.rectangle(self.canvas, (x, y), (x+bar_w, y+10), (40,40,40), -1)
        cv2.rectangle(self.canvas, (x, y), (x+filled, y+10), bar_color, -1)
        y += 22
        
        if alert_type:
            cv2.putText(self.canvas, f"Alert: {alert_type}", (x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLORS['warning'], 2)
        return self

    def draw_graph(self, data, title, x, y, w, h, threshold=None, color=None, face_detected=True):
        overlay = self.canvas.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (30, 30, 50), -1)
        cv2.addWeighted(overlay, 0.8, self.canvas, 0.2, 0, self.canvas)
        cv2.rectangle(self.canvas, (x, y), (x+w, y+h), COLORS['grid'], 1)
        cv2.putText(self.canvas, title, (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1)
        
        if not face_detected or len(data) < 2:
            return self
        
        if threshold:
            ty = y + h - int(threshold * (h-20))
            cv2.line(self.canvas, (x, ty), (x+w, ty), COLORS['alert'], 1, cv2.LINE_AA)
        
        points = []
        for i, v in enumerate(data):
            px = x + int((i / max(len(data)-1, 1)) * w)
            py = y + h - 5 - int(min(v, 1.0) * (h-25))
            points.append((px, py))
        
        graph_color = color if color else COLORS['accent']
        for i in range(len(points)-1):
            cv2.line(self.canvas, points[i], points[i+1], graph_color, 2, cv2.LINE_AA)
        return self

    def draw_footer(self, fps, frame_latency=0, detector_latency=0, face_detected=True):
        y = self.height - 30
        overlay = self.canvas.copy()
        cv2.rectangle(overlay, (0, self.height-40), (self.width, self.height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, self.canvas, 0.3, 0, self.canvas)
        
        cv2.putText(self.canvas, f"FPS: {fps:.0f}", (15, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['safe'], 1)
        cv2.putText(self.canvas, f"Frame: {frame_latency:.1f}ms", (95, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['info'], 1)
        cv2.putText(self.canvas, f"Detect: {detector_latency:.1f}ms", (220, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['info'], 1)
        cv2.putText(self.canvas, "[Q]uit | [S]hot | [R]eset | [F]ull | [H]elp | [+/-]", (380, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1)
        return self

    def get_frame(self):
        return self.canvas


class DrowsinessDetector:
    """Main application class."""
    
    def __init__(self):
        self.ui = UIRenderer(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.tracker = FatigueTracker()
        self.face_detector = FaceDetector()
        self.alarm = AlarmSystem()
        
        self.frame_count = 0
        self.fps = 0
        self.fps_time = time.time()
        self.alert_frames = 0
        self.alarm_on = False
        
        self.last_cnn_label = "Waiting..."
        self.last_cnn_conf = 0.0
        self.cached_cnn_conf = 0.0
        self.cnn_cache_frames = 0
        
        self.ear_history = deque(maxlen=50)
        self.mar_history = deque(maxlen=50)
        self.yawn_history = deque(maxlen=30)
        
        self.face_detected = False
        self.ear_thresh = EAR_THRESH
        self.frame_latency = 0.0
        self.detector_latency = 0.0
        
        self._initialize()

    def _initialize(self):
        print("="*60)
        print("DROWSINESS DETECTION SYSTEM")
        print("CNN + EAR/MAR Analysis + Head Pose")
        print("="*60)
        
        print("Loading CNN model...")
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print(f"Model loaded: {CLASS_NAMES}")
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            self.model = None
        
        print("Initializing camera...")
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open webcam")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print(f"Camera: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        print(f"EAR Threshold: {self.ear_thresh}")
        print("System ready. Press 'H' for help.\\n")
        
        cv2.namedWindow('Drowsiness Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Drowsiness Detection', WINDOW_WIDTH, WINDOW_HEIGHT)

    def get_cnn_prediction(self, frame, bbox):
        if self.model is None:
            return 0.0
        
        if self.cnn_cache_frames <= 0:
            try:
                x, y, w, h = bbox
                pad = int(h * 0.3)
                y1 = max(0, y - pad)
                y2 = min(frame.shape[0], y + h + pad)
                x1 = max(0, x - pad)
                x2 = min(frame.shape[1], x + w + pad)
                
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    img = cv2.resize(roi, (224, 224))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                    pred = self.model.predict(np.expand_dims(img, 0), verbose=0)[0]
                    
                    self.last_cnn_label = CLASS_NAMES[np.argmax(pred)]
                    self.cached_cnn_conf = float(pred[YAWN_INDEX])
                    self.last_cnn_conf = self.cached_cnn_conf
                    self.cnn_cache_frames = 3
            except:
                self.cached_cnn_conf = 0.0
        else:
            self.cnn_cache_frames -= 1
        
        return self.cached_cnn_conf

    def process_frame(self):
        start_time = time.time()
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            current_time = time.time()
            self.fps = 30 / (current_time - self.fps_time)
            self.fps_time = current_time
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Face detection
        detect_start = time.time()
        results = self.face_detector.detect(frame)
        self.detector_latency = (time.time() - detect_start) * 1000
        
        # Initialize variables
        ear = mar = left_ear = right_ear = 0.0
        status = "AWAKE"
        reason = alert_type = ""
        alarm_trigger = False
        eyes_closed = False
        cnn_conf = yawn_score = 0.0
        final_class = "Waiting..."
        bbox = None
        head_pose = None
        
        self.face_detected = (results.multi_face_landmarks is not None)
        
        if self.face_detected:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Calculate metrics
            left_ear = self.face_detector.calculate_ear(landmarks, self.face_detector.LEFT_EYE, w, h)
            right_ear = self.face_detector.calculate_ear(landmarks, self.face_detector.RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0
            mar = self.face_detector.calculate_mar(landmarks, w, h)
            head_pose = self.face_detector.estimate_head_pose(landmarks, w, h)
            
            self.ear_history.append(ear)
            self.mar_history.append(mar)
            eyes_closed = ear < self.ear_thresh
            
            # Get bounding box
            xs = [p.x for p in landmarks]
            ys = [p.y for p in landmarks]
            min_x = int(min(xs) * w)
            max_x = int(max(xs) * w)
            min_y = int(min(ys) * h)
            max_y = int(max(ys) * h)
            bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
            
            # CNN prediction
            cnn_conf = self.get_cnn_prediction(frame, bbox)
            
            # Determine class
            if eyes_closed and cnn_conf < 0.5:
                final_class = "Closed_Eyes"
            elif cnn_conf > CNN_YAWN_CONFIDENCE:
                final_class = "Yawn"
            else:
                final_class = self.last_cnn_label if self.last_cnn_label != "Waiting..." else "Open_Eyes"
            
            # Calculate yawn score
            mar_score = 1.0 if mar > MAR_HIGH_THRESH else \
                       ((mar - MAR_THRESH) / (MAR_HIGH_THRESH - MAR_THRESH) if mar > MAR_THRESH else 0.0)
            cnn_score = max(0, (cnn_conf - CNN_YAWN_CONFIDENCE) / (1.0 - CNN_YAWN_CONFIDENCE))
            yawn_score = min(1.0, mar_score * 0.5 + cnn_score * 0.5)
            self.yawn_history.append(yawn_score)
            
            # Update tracker
            self.tracker.update(ear, mar, yawn_score, eyes_closed, head_pose)
            
            # Alert logic
            if final_class == "Closed_Eyes":
                self.alert_frames += 1
                alert_type = "EYES"
            elif final_class == "Yawn" or self.tracker.yawn_active:
                self.alert_frames += 2
                alert_type = "YAWN"
            elif head_pose and math.degrees(head_pose[0]) > HEAD_PITCH_THRESH:
                self.alert_frames += 1
                alert_type = "NOD"
            else:
                self.alert_frames = max(0, self.alert_frames - COOLDOWN_FRAMES)
            
            # Set status
            if self.alert_frames >= ALARM_FRAMES:
                status = "ALARM"
                alarm_trigger = True
                reason = "Eyes Closed" if alert_type == "EYES" else \
                        ("Yawning" if alert_type == "YAWN" else "Head Nodding")
            elif self.alert_frames >= WARNING_FRAMES:
                status = "WARNING"
                reason = "Eyes Closing" if alert_type == "EYES" else \
                        ("Yawn Detected" if alert_type == "YAWN" else "Nodding Detected")
            
            # Alarm control
            if alarm_trigger:
                if not self.alarm_on:
                    self.alarm.start()
                    self.alarm_on = True
            else:
                if self.alarm_on:
                    self.alarm.stop()
                    self.alarm_on = False
            
            # Draw landmarks
            frame = self.face_detector.draw_landmarks(frame, landmarks, eyes_closed, 
                                                     self.tracker.yawn_active, left_ear, right_ear)
            frame = self.face_detector.draw_head_pose(frame, landmarks, head_pose, w, h)
            
            # Draw bounding box
            box_color = COLORS['alert'] if alarm_trigger else \
                       (COLORS['warning'] if status == "WARNING" else COLORS['safe'])
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), box_color, 2)
        else:
            # No face detected
            self.alert_frames = max(0, self.alert_frames - COOLDOWN_FRAMES)
            if self.alarm_on:
                self.alarm.stop()
                self.alarm_on = False
            self.tracker.consecutive_closed = 0
            final_class = "No Face"
        
        self.frame_latency = (time.time() - start_time) * 1000
        
        # Render UI
        self.ui.create_base(frame, self.face_detected)
        self.ui.draw_header(status, reason, self.face_detected)
        self.ui.draw_metrics(self.tracker, ear, mar, self.alert_frames, eyes_closed, 
                            left_ear, right_ear, alert_type, self.last_cnn_conf, 
                            yawn_score, final_class, head_pose, self.face_detected, self.ear_thresh)
        self.ui.draw_graph(list(self.ear_history)[-30:], "EAR", 660, 420, 230, 70, 
                          self.ear_thresh, COLORS['eye_open'], self.face_detected)
        self.ui.draw_graph(list(self.mar_history)[-30:], "MAR", 660, 500, 230, 70, 
                          MAR_THRESH, COLORS['mouth_open'], self.face_detected)
        self.ui.draw_footer(self.fps, self.frame_latency, self.detector_latency, self.face_detected)
        
        return self.ui.get_frame()

    def run(self):
        fullscreen = False
        show_help = False
        
        try:
            while True:
                display = self.process_frame()
                if display is None:
                    break
                
                # Help overlay
                if show_help:
                    overlay = display.copy()
                    cv2.rectangle(overlay, (200, 150), (700, 480), (0, 0, 0), -1)
                    cv2.rectangle(overlay, (200, 150), (700, 480), COLORS['border'], 2)
                    
                    help_text = [
                        "HELP - Keyboard Shortcuts",
                        "[Q] - Quit",
                        "[S] - Screenshot",
                        "[R] - Reset",
                        "[F] - Fullscreen",
                        "[H] - Help",
                        "[+/-] - Adjust EAR threshold"
                    ]
                    
                    for i, text in enumerate(help_text):
                        font_scale = 0.6 if i == 0 else 0.5
                        color = COLORS['info'] if i == 0 else COLORS['text']
                        thickness = 2 if i == 0 else 1
                        cv2.putText(overlay, text, (220, 190 + i*35), 
                                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
                    
                    cv2.addWeighted(overlay, 0.9, display, 0.1, 0, display)
                
                cv2.imshow('Drowsiness Detection', display)
                
                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"screenshot_{int(time.time())}.png"
                    cv2.imwrite(filename, display)
                    print(f"Screenshot saved: {filename}")
                elif key == ord('r'):
                    self.tracker = FatigueTracker()
                    self.alert_frames = 0
                    if self.alarm_on:
                        self.alarm.stop()
                        self.alarm_on = False
                    print("System reset")
                elif key == ord('f'):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty('Drowsiness Detection', cv2.WND_PROP_FULLSCREEN, 
                                         cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                    if not fullscreen:
                        cv2.resizeWindow('Drowsiness Detection', WINDOW_WIDTH, WINDOW_HEIGHT)
                elif key == ord('h'):
                    show_help = not show_help
                elif key in [ord('+'), ord('=')]:
                    self.ear_thresh = min(0.40, self.ear_thresh + 0.01)
                    print(f"EAR threshold: {self.ear_thresh:.2f}")
                elif key == ord('-'):
                    self.ear_thresh = max(0.10, self.ear_thresh - 0.01)
                    print(f"EAR threshold: {self.ear_thresh:.2f}")
                
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\\nStopped by user")
        finally:
            if self.alarm_on:
                self.alarm.stop()
            self.cap.release()
            cv2.destroyAllWindows()
            print("Shutdown complete")


if __name__ == "__main__":
    try:
        DrowsinessDetector().run()
    except Exception as e:
        print(f"Error: {e}")
        input("Press Enter to exit...")
'''

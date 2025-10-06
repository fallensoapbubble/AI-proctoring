"""
Fixed Gaze Detection Component - Simplified implementation without complex logging.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

from context_engine.interfaces import DetectionSource
from context_engine.models import DetectionEvent, DetectionType
from shared_utils.detection_utils import normalize_confidence


class FixedGazeDetector(DetectionSource):
    """Fixed gaze detector with simplified logging."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the gaze detector."""
        self.config = config or {}
        
        self.yaw_threshold = self.config.get('yaw_threshold', 20)
        self.pitch_min = self.config.get('pitch_min', -10)
        self.pitch_max = self.config.get('pitch_max', 15)
        self.roll_min = self.config.get('roll_min', 80)
        self.roll_max = self.config.get('roll_max', 110)
        
        self.face_mesh = None
        self.is_running = False
        
        # Detection statistics
        self.total_detections = 0
        self.suspicious_detections = 0
        
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.model_points = np.array([
                (0.0, 0.0, 0.0), (-30.0, -125.0, -30.0), (30.0, -125.0, -30.0),
                (-60.0, -70.0, -60.0), (60.0, -70.0, -60.0), (0.0, -150.0, -25.0)
            ], dtype="double")
        else:
            print("MediaPipe not available - gaze detection will use fallback method")
    
    def get_source_name(self) -> str:
        """Return the unique name of this detection source."""
        return "fixed_gaze_detector"
    
    def is_available(self) -> bool:
        """Check if MediaPipe face mesh is available."""
        return MEDIAPIPE_AVAILABLE
    
    def start_detection(self) -> None:
        """Initialize MediaPipe face mesh for detection."""
        try:
            if MEDIAPIPE_AVAILABLE:
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True
                )
            self.is_running = True
            self.total_detections = 0
            self.suspicious_detections = 0
            print("FixedGazeDetector started successfully")
        except Exception as e:
            print(f"Failed to start FixedGazeDetector: {e}")
            self.is_running = False
    
    def stop_detection(self) -> None:
        """Stop detection and cleanup resources."""
        try:
            if self.face_mesh and MEDIAPIPE_AVAILABLE:
                self.face_mesh.close()
                self.face_mesh = None
            self.is_running = False
            print("FixedGazeDetector stopped")
        except Exception as e:
            print(f"Error stopping FixedGazeDetector: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Return health status information for monitoring."""
        return {
            'source_name': self.get_source_name(),
            'is_running': self.is_running,
            'is_available': self.is_available(),
            'face_mesh_initialized': self.face_mesh is not None,
            'config': {
                'yaw_threshold': self.yaw_threshold,
                'pitch_range': [self.pitch_min, self.pitch_max],
                'roll_range': [self.roll_min, self.roll_max]
            }
        }
    
    def detect_gaze_direction(self, frame: np.ndarray) -> Optional[DetectionEvent]:
        """Analyze frame for suspicious gaze direction."""
        if not self.is_running:
            return None
        
        self.total_detections += 1
        
        try:
            if MEDIAPIPE_AVAILABLE and self.face_mesh is not None:
                return self._detect_with_mediapipe(frame)
            else:
                return self._detect_with_fallback(frame)
        except Exception as e:
            print(f"Error in gaze detection: {e}")
            return None
    
    def _detect_with_mediapipe(self, frame: np.ndarray) -> Optional[DetectionEvent]:
        """Detect gaze using MediaPipe face mesh."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        face_landmarks = results.multi_face_landmarks[0]
        landmarks_px = [
            (int(pt.x * frame.shape[1]), int(pt.y * frame.shape[0]))
            for pt in face_landmarks.landmark
        ]
        
        rotation_vector = self._get_head_pose(frame, landmarks_px)
        if rotation_vector is None:
            return None
        
        yaw, pitch, roll = self._rotation_vector_to_euler_angles(rotation_vector)
        
        is_suspicious = (
            abs(yaw) > self.yaw_threshold or
            pitch < self.pitch_min or
            pitch > self.pitch_max or
            roll < self.roll_min or
            roll > self.roll_max
        )
        
        if is_suspicious:
            self.suspicious_detections += 1
            
            yaw_deviation = min(abs(yaw) / self.yaw_threshold, 1.0)
            pitch_deviation = max(
                (self.pitch_min - pitch) / abs(self.pitch_min) if self.pitch_min != 0 else 0,
                (pitch - self.pitch_max) / self.pitch_max if self.pitch_max != 0 else 0,
                0.0
            )
            roll_deviation = max(
                (self.roll_min - roll) / self.roll_min if self.roll_min != 0 else 0,
                (roll - self.roll_max) / (180 - self.roll_max) if self.roll_max != 180 else 0,
                0.0
            )
            
            confidence = normalize_confidence(
                max(yaw_deviation, pitch_deviation, roll_deviation)
            )
            
            metadata = {
                'detection_method': 'mediapipe',
                'head_pose': {
                    'yaw': float(yaw),
                    'pitch': float(pitch),
                    'roll': float(roll)
                },
                'thresholds': {
                    'yaw_threshold': self.yaw_threshold,
                    'pitch_range': [self.pitch_min, self.pitch_max],
                    'roll_range': [self.roll_min, self.roll_max]
                }
            }
            
            return DetectionEvent(
                event_type=DetectionType.GAZE_AWAY,
                timestamp=datetime.now(),
                confidence=confidence,
                source=self.get_source_name(),
                metadata=metadata,
                frame_data=frame.copy()
            )
        
        return None
    
    def _detect_with_fallback(self, frame: np.ndarray) -> Optional[DetectionEvent]:
        """Fallback gaze detection using OpenCV face detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use Haar cascade for basic face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            # No face detected - suspicious
            self.suspicious_detections += 1
            return DetectionEvent(
                event_type=DetectionType.GAZE_AWAY,
                timestamp=datetime.now(),
                confidence=0.8,
                source=self.get_source_name(),
                metadata={
                    'detection_method': 'fallback',
                    'reason': 'no_face_detected',
                    'description': 'No face detected in frame'
                },
                frame_data=frame.copy()
            )
        
        # Simple position-based detection
        face = faces[0]
        x, y, w, h = face
        face_center_x = x + w // 2
        frame_center_x = frame.shape[1] // 2
        
        # Check if face is significantly off-center
        horizontal_offset = abs(face_center_x - frame_center_x) / frame.shape[1]
        
        if horizontal_offset > 0.3:  # Face is off-center
            self.suspicious_detections += 1
            confidence = min(horizontal_offset * 2, 1.0)
            
            return DetectionEvent(
                event_type=DetectionType.GAZE_AWAY,
                timestamp=datetime.now(),
                confidence=confidence,
                source=self.get_source_name(),
                metadata={
                    'detection_method': 'fallback',
                    'reason': 'face_off_center',
                    'horizontal_offset': float(horizontal_offset),
                    'face_position': [int(x), int(y), int(w), int(h)]
                },
                frame_data=frame.copy()
            )
        
        return None
    
    def _get_head_pose(self, frame: np.ndarray, landmarks_px: List[tuple]) -> Optional[np.ndarray]:
        """Calculate head pose from facial landmarks."""
        try:
            if len(landmarks_px) < 152:
                return None
                
            image_points = np.array([
                landmarks_px[1], landmarks_px[33], landmarks_px[263],
                landmarks_px[61], landmarks_px[291], landmarks_px[152]
            ], dtype="double")
            
            size = frame.shape
            focal_length = size[1]
            center = (size[1] // 2, size[0] // 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")
            
            dist_coeffs = np.zeros((4, 1))
            success, rotation_vector, _ = cv2.solvePnP(
                self.model_points, image_points, camera_matrix, dist_coeffs
            )
            
            return rotation_vector if success else None
        except Exception:
            return None
    
    def _rotation_vector_to_euler_angles(self, rotation_vector: np.ndarray) -> tuple:
        """Convert rotation vector to Euler angles."""
        try:
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            pitch = np.arctan2(
                -rotation_matrix[2, 0],
                np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2)
            )
            roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            
            return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)
        except Exception:
            return 0.0, 0.0, 0.0
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update detector configuration."""
        self.config.update(new_config)
        self.yaw_threshold = self.config.get('yaw_threshold', self.yaw_threshold)
        self.pitch_min = self.config.get('pitch_min', self.pitch_min)
        self.pitch_max = self.config.get('pitch_max', self.pitch_max)
        self.roll_min = self.config.get('roll_min', self.roll_min)
        self.roll_max = self.config.get('roll_max', self.roll_max)
        print(f"FixedGazeDetector configuration updated: {new_config}")
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return {
            'total_detections': self.total_detections,
            'suspicious_detections': self.suspicious_detections,
            'detection_rate': self.suspicious_detections / max(self.total_detections, 1),
            'mediapipe_available': MEDIAPIPE_AVAILABLE,
            'is_running': self.is_running
        }
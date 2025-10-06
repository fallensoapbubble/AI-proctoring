"""
Lip movement detection component for speech analysis.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

from context_engine.interfaces import DetectionSource
from context_engine.models import DetectionEvent, DetectionType
from shared_utils.detection_utils import normalize_confidence


class LipMovementDetector(DetectionSource):
    """Detects lip movement that could indicate speaking or communication."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the lip movement detector."""
        self.config = config or {}
        self.logger_name = "LipMovementDetector"
        
        self.movement_threshold = self.config.get('movement_threshold', 0.003)
        self.smoothing_window = self.config.get('smoothing_window', 3)
        
        self.face_mesh = None
        self.is_running = False
        
        # Detection statistics
        self.total_detections = 0
        self.movement_detections = 0
        
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.upper_lip_indices = [13, 14, 15, 16, 17]
            self.lower_lip_indices = [82, 81, 80, 191, 178]
        else:
            self.logger.warning("MediaPipe not available - lip movement detection will use fallback method")
        
        self.previous_lip_distances = []
        self.movement_history = []
    
    def get_source_name(self) -> str:
        """Return the unique name of this detection source."""
        return "lip_movement_detector"
    
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
            self.previous_lip_distances = []
            self.movement_history = []
            self.total_detections = 0
            self.movement_detections = 0
            self.logger.info("LipMovementDetector started successfully")
        except Exception as e:
            self.logger.error(f"Failed to start LipMovementDetector: {e}")
            self.is_running = False
    
    def stop_detection(self) -> None:
        """Stop detection and cleanup resources."""
        try:
            if self.face_mesh and MEDIAPIPE_AVAILABLE:
                self.face_mesh.close()
                self.face_mesh = None
            self.is_running = False
            self.previous_lip_distances = []
            self.movement_history = []
            self.logger.info("LipMovementDetector stopped")
        except Exception as e:
            self.logger.error(f"Error stopping LipMovementDetector: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Return health status information for monitoring."""
        return {
            'source_name': self.get_source_name(),
            'is_running': self.is_running,
            'is_available': self.is_available(),
            'face_mesh_initialized': self.face_mesh is not None,
            'movement_threshold': self.movement_threshold,
            'history_length': len(self.movement_history),
            'config': self.config
        }
    
    def detect_lip_movement(self, frame: np.ndarray) -> Optional[DetectionEvent]:
        """Analyze frame for lip movement indicating speech."""
        if not self.is_running:
            return None
        
        self.total_detections += 1
        
        try:
            if MEDIAPIPE_AVAILABLE and self.face_mesh is not None:
                return self._detect_with_mediapipe(frame)
            else:
                return self._detect_with_fallback(frame)
        except Exception as e:
            self.logger.error(f"Error in lip movement detection: {e}")
            return None
    
    def _detect_with_mediapipe(self, frame: np.ndarray) -> Optional[DetectionEvent]:
        """Detect lip movement using MediaPipe face mesh."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        face_landmarks = results.multi_face_landmarks[0]
        current_distance = self._calculate_lip_distance(face_landmarks.landmark)
        
        if current_distance is None:
            return None
        
        self.previous_lip_distances.append(current_distance)
        if len(self.previous_lip_distances) > self.smoothing_window:
            self.previous_lip_distances.pop(0)
        
        if len(self.previous_lip_distances) < 2:
            return None
        
        movement = abs(current_distance - self.previous_lip_distances[-2])
        
        self.movement_history.append(movement)
        if len(self.movement_history) > 10:
            self.movement_history.pop(0)
        
        if movement > self.movement_threshold:
            self.movement_detections += 1
            
            movement_confidence = min(movement / (self.movement_threshold * 2), 1.0)
            
            recent_movements = self.movement_history[-3:] if len(self.movement_history) >= 3 else self.movement_history
            avg_recent_movement = np.mean(recent_movements) if recent_movements else 0
            consistency_factor = min(avg_recent_movement / self.movement_threshold, 1.0)
            
            confidence = normalize_confidence((movement_confidence + consistency_factor) / 2)
            
            metadata = {
                'detection_method': 'mediapipe',
                'lip_distance': float(current_distance),
                'movement_magnitude': float(movement),
                'movement_threshold': self.movement_threshold,
                'movement_confidence': float(movement_confidence),
                'consistency_factor': float(consistency_factor),
                'recent_movements': [float(m) for m in recent_movements],
                'smoothing_window': self.smoothing_window
            }
            
            return DetectionEvent(
                event_type=DetectionType.LIP_MOVEMENT,
                timestamp=datetime.now(),
                confidence=confidence,
                source=self.get_source_name(),
                metadata=metadata,
                frame_data=frame.copy()
            )
        
        return None
    
    def _detect_with_fallback(self, frame: np.ndarray) -> Optional[DetectionEvent]:
        """Fallback lip movement detection using basic image analysis."""
        # Simple fallback: detect mouth region changes using basic CV
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use face detection to find mouth region
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None
        
        # Focus on the lower part of the face (mouth region)
        face = faces[0]
        x, y, w, h = face
        mouth_y = y + int(h * 0.6)  # Lower 40% of face
        mouth_h = int(h * 0.4)
        mouth_region = gray[mouth_y:mouth_y + mouth_h, x:x + w]
        
        if mouth_region.size == 0:
            return None
        
        # Calculate variance in mouth region (movement indicator)
        variance = np.var(mouth_region)
        
        # Store variance history
        if not hasattr(self, 'variance_history'):
            self.variance_history = []
        
        self.variance_history.append(variance)
        if len(self.variance_history) > 5:
            self.variance_history.pop(0)
        
        if len(self.variance_history) >= 2:
            variance_change = abs(variance - self.variance_history[-2])
            threshold = 100  # Adjust based on testing
            
            if variance_change > threshold:
                self.movement_detections += 1
                confidence = min(variance_change / (threshold * 2), 1.0)
                
                return DetectionEvent(
                    event_type=DetectionType.LIP_MOVEMENT,
                    timestamp=datetime.now(),
                    confidence=confidence,
                    source=self.get_source_name(),
                    metadata={
                        'detection_method': 'fallback',
                        'variance': float(variance),
                        'variance_change': float(variance_change),
                        'threshold': threshold,
                        'mouth_region_size': mouth_region.shape
                    },
                    frame_data=frame.copy()
                )
        
        return None
    
    def _calculate_lip_distance(self, landmarks) -> Optional[float]:
        """Calculate average distance between upper and lower lip landmarks."""
        try:
            distances = []
            min_pairs = min(len(self.upper_lip_indices), len(self.lower_lip_indices))
            
            for i in range(min_pairs):
                if (self.upper_lip_indices[i] < len(landmarks) and 
                    self.lower_lip_indices[i] < len(landmarks)):
                    upper_point = landmarks[self.upper_lip_indices[i]]
                    lower_point = landmarks[self.lower_lip_indices[i]]
                    
                    distance = np.sqrt(
                        (upper_point.x - lower_point.x)**2 +
                        (upper_point.y - lower_point.y)**2
                    )
                    distances.append(distance)
            
            return np.mean(distances) if distances else None
        except Exception:
            return None
    
    def reset_history(self) -> None:
        """Reset movement history and previous measurements."""
        self.previous_lip_distances = []
        self.movement_history = []
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update detector configuration."""
        self.config.update(new_config)
        self.movement_threshold = self.config.get('movement_threshold', self.movement_threshold)
        self.smoothing_window = self.config.get('smoothing_window', self.smoothing_window)
        
        if len(self.previous_lip_distances) > self.smoothing_window:
            self.previous_lip_distances = self.previous_lip_distances[-self.smoothing_window:]
    
    def get_movement_statistics(self) -> Dict[str, Any]:
        """Get statistics about recent lip movement patterns."""
        if not self.movement_history:
            return {
                'avg_movement': 0.0,
                'max_movement': 0.0,
                'movement_count': 0,
                'threshold_exceeded_count': 0
            }
        
        movements = np.array(self.movement_history)
        threshold_exceeded = np.sum(movements > self.movement_threshold)
        
        return {
            'avg_movement': float(np.mean(movements)),
            'max_movement': float(np.max(movements)),
            'movement_count': len(movements),
            'threshold_exceeded_count': int(threshold_exceeded),
            'threshold_exceeded_ratio': float(threshold_exceeded / len(movements))
        }

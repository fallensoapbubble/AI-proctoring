"""
Face spoofing detection component using YOLO for anti-spoofing.

This module extracts the face spoof detection logic from backapp.py into a
modular, testable component that implements the DetectionSource interface.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from context_engine.interfaces import DetectionSource
from context_engine.models import DetectionEvent, DetectionType
from shared_utils.detection_utils import normalize_confidence


class FaceSpoofDetector(DetectionSource):
    """
    Detects face spoofing attempts using YOLO-based anti-spoofing models.
    
    This detector identifies attempts to bypass face recognition using photos,
    videos, masks, or other spoofing techniques.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the face spoof detector.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.FaceSpoofDetector")
        
        # Detection parameters
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.model_path = self.config.get('model_path', 'yolov8n.pt')
        
        # YOLO model
        self.model = None
        self.is_running = False
        
        # Spoof detection classes (depends on model training)
        # Typically: 0 = real face, 1 = spoof
        self.spoof_class_id = self.config.get('spoof_class_id', 1)
        self.real_class_id = self.config.get('real_class_id', 0)
        
        # Detection history for stability
        self.detection_history = []
        self.history_size = self.config.get('history_size', 5)
        
        # Spoof detection thresholds
        self.spoof_confidence_threshold = self.config.get('spoof_confidence_threshold', 0.7)
        self.real_confidence_threshold = self.config.get('real_confidence_threshold', 0.85)
        
        # Detection statistics
        self.total_detections = 0
        self.spoof_detections = 0
        
        if not YOLO_AVAILABLE:
            self.logger.warning("YOLO not available - install ultralytics package")
        
        self.logger.info("FaceSpoofDetector initialized")
    
    def get_source_name(self) -> str:
        """Return the unique name of this detection source."""
        return "face_spoof_detector"
    
    def is_available(self) -> bool:
        """Check if YOLO model can be loaded."""
        if not YOLO_AVAILABLE:
            return False
        
        try:
            if self.model is None:
                # Try to load model to check availability
                test_model = YOLO(self.model_path)
                return True
            return True
        except Exception as e:
            self.logger.error(f"FaceSpoofDetector availability check failed: {e}")
            return False
    
    def start_detection(self) -> None:
        """Load YOLO model for detection."""
        try:
            if YOLO_AVAILABLE:
                self.model = YOLO(self.model_path)
            self.is_running = True
            self.detection_history = []
            self.total_detections = 0
            self.spoof_detections = 0
            self.logger.info(f"FaceSpoofDetector started with model: {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to start FaceSpoofDetector: {e}")
            raise
    
    def stop_detection(self) -> None:
        """Stop detection and cleanup resources."""
        try:
            self.model = None
            self.is_running = False
            self.detection_history = []
            self.logger.info("FaceSpoofDetector stopped")
        except Exception as e:
            self.logger.error(f"Error stopping FaceSpoofDetector: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Return health status information for monitoring."""
        return {
            'source_name': self.get_source_name(),
            'is_running': self.is_running,
            'is_available': self.is_available(),
            'model_loaded': self.model is not None,
            'model_path': self.model_path,
            'confidence_threshold': self.confidence_threshold,
            'spoof_confidence_threshold': self.spoof_confidence_threshold,
            'real_confidence_threshold': self.real_confidence_threshold,
            'detection_history_size': len(self.detection_history),
            'config': self.config
        }
    
    def detect_face_spoof(self, frame: np.ndarray) -> Optional[DetectionEvent]:
        """
        Analyze frame for face spoofing attempts.
        
        Args:
            frame: Input video frame
            
        Returns:
            DetectionEvent if spoof detected, None otherwise
        """
        if not self.is_running:
            return None
        
        self.total_detections += 1
        
        try:
            if YOLO_AVAILABLE and self.model is not None:
                return self._detect_with_yolo(frame)
            else:
                return self._detect_with_fallback(frame)
        except Exception as e:
            self.logger.error(f"Error in face spoof detection: {e}")
            return None
    
    def _detect_with_yolo(self, frame: np.ndarray) -> Optional[DetectionEvent]:
        """Detect face spoofing using YOLO."""
        # Convert BGR to RGB for YOLO
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run YOLO inference
        results = self.model.predict(rgb_frame, verbose=False)
        
        spoof_detections = []
        real_detections = []
        
        # Process detection results
        for result in results:
            if result.boxes is None:
                continue
            
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = box.conf[0].item()
                
                if confidence < self.confidence_threshold:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                detection_info = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'class_id': class_id,
                    'area': (x2 - x1) * (y2 - y1),
                    'center': [(x1 + x2) // 2, (y1 + y2) // 2]
                }
                
                if class_id == self.spoof_class_id:
                    spoof_detections.append(detection_info)
                    
                    # Draw spoof detection box (optional)
                    if self.config.get('draw_boxes', False):
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(
                            frame,
                            f"Spoof ({confidence:.2f})",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            2
                        )
                
                elif class_id == self.real_class_id:
                    real_detections.append(detection_info)
                    
                    # Draw real face detection box (optional)
                    if self.config.get('draw_boxes', False):
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"Real ({confidence:.2f})",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2
                        )
        
        # Update detection history
        spoof_count = len(spoof_detections)
        self.detection_history.append({
            'spoof_count': spoof_count,
            'real_count': len(real_detections),
            'timestamp': datetime.now()
        })
        
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)
        
        # Determine if we should trigger a spoof alert
        return self._analyze_spoof_detections(spoof_detections, real_detections, frame, 'yolo')
    
    def _detect_with_fallback(self, frame: np.ndarray) -> Optional[DetectionEvent]:
        """Fallback spoof detection using basic image analysis."""
        # Simple fallback: analyze image quality and patterns
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use face detection to find face region
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None
        
        # Analyze the largest face
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        face_roi = gray[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            return None
        
        # Simple spoof indicators
        # 1. Check for uniform brightness (photo characteristic)
        brightness_std = np.std(face_roi)
        
        # 2. Check for lack of texture (printed photo)
        laplacian_var = cv2.Laplacian(face_roi, cv2.CV_64F).var()
        
        # Simple thresholds (adjust based on testing)
        is_spoof = brightness_std < 20 or laplacian_var < 100
        
        if is_spoof:
            self.spoof_detections += 1
            
            confidence = 0.6  # Conservative confidence for fallback method
            
            return DetectionEvent(
                event_type=DetectionType.FACE_SPOOF,
                timestamp=datetime.now(),
                confidence=confidence,
                source=self.get_source_name(),
                metadata={
                    'detection_method': 'fallback',
                    'brightness_std': float(brightness_std),
                    'laplacian_var': float(laplacian_var),
                    'face_bbox': [x, y, x + w, y + h],
                    'spoof_indicators': {
                        'low_brightness_variation': brightness_std < 20,
                        'low_texture': laplacian_var < 100
                    }
                },
                frame_data=frame.copy()
            )
        
        return None
    
    def _analyze_spoof_detections(
        self,
        spoof_detections: List[Dict[str, Any]],
        real_detections: List[Dict[str, Any]],
        frame: np.ndarray,
        method: str = 'yolo'
    ) -> Optional[DetectionEvent]:
        """
        Analyze spoof and real detections to determine if alert should be triggered.
        
        Args:
            spoof_detections: List of spoof detections
            real_detections: List of real face detections
            frame: Original frame
            
        Returns:
            DetectionEvent if spoof should be reported, None otherwise
        """
        # Case 1: Direct spoof detection with high confidence
        if spoof_detections:
            self.spoof_detections += 1
            best_spoof = max(spoof_detections, key=lambda x: x['confidence'])
            
            if best_spoof['confidence'] >= self.spoof_confidence_threshold:
                confidence = normalize_confidence(best_spoof['confidence'])
                
                metadata = {
                    'detection_method': method,
                    'detection_type': 'direct_spoof',
                    'spoof_detections': spoof_detections,
                    'real_detections': real_detections,
                    'best_spoof_confidence': best_spoof['confidence'],
                    'spoof_threshold': self.spoof_confidence_threshold,
                    'detection_history': self._get_recent_history_summary()
                }
                
                return DetectionEvent(
                    event_type=DetectionType.FACE_SPOOF,
                    timestamp=datetime.now(),
                    confidence=confidence,
                    source=self.get_source_name(),
                    metadata=metadata,
                    frame_data=frame.copy()
                )
        
        # Case 2: Low confidence real face detection (possible spoof)
        if real_detections and not spoof_detections:
            best_real = max(real_detections, key=lambda x: x['confidence'])
            
            if best_real['confidence'] < self.real_confidence_threshold:
                # Calculate confidence based on how low the real face confidence is
                confidence_deficit = self.real_confidence_threshold - best_real['confidence']
                spoof_confidence = normalize_confidence(confidence_deficit * 2)  # Amplify the signal
                
                # Only trigger if we have consistent low confidence
                if self._is_consistently_low_confidence() and spoof_confidence > 0.5:
                    self.spoof_detections += 1
                    metadata = {
                        'detection_method': method,
                        'detection_type': 'low_confidence_real',
                        'real_detections': real_detections,
                        'best_real_confidence': best_real['confidence'],
                        'real_threshold': self.real_confidence_threshold,
                        'confidence_deficit': confidence_deficit,
                        'detection_history': self._get_recent_history_summary()
                    }
                    
                    return DetectionEvent(
                        event_type=DetectionType.FACE_SPOOF,
                        timestamp=datetime.now(),
                        confidence=spoof_confidence,
                        source=self.get_source_name(),
                        metadata=metadata,
                        frame_data=frame.copy()
                    )
        
        return None
    
    def _is_consistently_low_confidence(self) -> bool:
        """
        Check if recent real face detections have consistently low confidence.
        
        Returns:
            True if consistently low confidence, False otherwise
        """
        if len(self.detection_history) < 3:
            return False
        
        recent_history = self.detection_history[-3:]
        low_confidence_frames = 0
        
        for entry in recent_history:
            if entry['real_count'] > 0 and entry['spoof_count'] == 0:
                # This indicates low confidence real detection
                low_confidence_frames += 1
        
        return low_confidence_frames >= 2
    
    def _get_recent_history_summary(self) -> Dict[str, Any]:
        """
        Get summary of recent detection history.
        
        Returns:
            Dictionary containing history summary
        """
        if not self.detection_history:
            return {'total_frames': 0, 'spoof_frames': 0, 'real_frames': 0}
        
        recent = self.detection_history[-5:]  # Last 5 frames
        
        spoof_frames = sum(1 for entry in recent if entry['spoof_count'] > 0)
        real_frames = sum(1 for entry in recent if entry['real_count'] > 0)
        
        return {
            'total_frames': len(recent),
            'spoof_frames': spoof_frames,
            'real_frames': real_frames,
            'spoof_rate': spoof_frames / len(recent),
            'real_rate': real_frames / len(recent)
        }
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about recent spoof detections.
        
        Returns:
            Dictionary containing detection statistics
        """
        if not self.detection_history:
            return {
                'total_frames': 0,
                'frames_with_spoof': 0,
                'frames_with_real': 0,
                'spoof_detection_rate': 0.0,
                'real_detection_rate': 0.0
            }
        
        spoof_frames = sum(1 for entry in self.detection_history if entry['spoof_count'] > 0)
        real_frames = sum(1 for entry in self.detection_history if entry['real_count'] > 0)
        total_frames = len(self.detection_history)
        
        return {
            'total_frames': total_frames,
            'frames_with_spoof': spoof_frames,
            'frames_with_real': real_frames,
            'spoof_detection_rate': spoof_frames / total_frames,
            'real_detection_rate': real_frames / total_frames,
            'avg_spoof_per_frame': sum(entry['spoof_count'] for entry in self.detection_history) / total_frames,
            'avg_real_per_frame': sum(entry['real_count'] for entry in self.detection_history) / total_frames
        }
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update detector configuration.
        
        Args:
            new_config: New configuration parameters
        """
        self.config.update(new_config)
        
        # Update parameters
        self.confidence_threshold = self.config.get('confidence_threshold', self.confidence_threshold)
        self.spoof_confidence_threshold = self.config.get('spoof_confidence_threshold', self.spoof_confidence_threshold)
        self.real_confidence_threshold = self.config.get('real_confidence_threshold', self.real_confidence_threshold)
        self.spoof_class_id = self.config.get('spoof_class_id', self.spoof_class_id)
        self.real_class_id = self.config.get('real_class_id', self.real_class_id)
        self.history_size = self.config.get('history_size', self.history_size)
        
        # Adjust history size if changed
        if len(self.detection_history) > self.history_size:
            self.detection_history = self.detection_history[-self.history_size:]
        
        # If model path changed, need to restart detection
        new_model_path = self.config.get('model_path', self.model_path)
        if new_model_path != self.model_path:
            self.model_path = new_model_path
            if self.is_running:
                self.logger.info("Model path changed, restarting detection")
                self.stop_detection()
                self.start_detection()
        
        self.logger.info(f"FaceSpoofDetector configuration updated: {new_config}")
    
    def reset_history(self) -> None:
        """Reset detection history."""
        self.detection_history = []
        self.logger.info("FaceSpoofDetector history reset")
    
    def set_thresholds(self, spoof_threshold: float, real_threshold: float) -> None:
        """
        Set detection thresholds.
        
        Args:
            spoof_threshold: Confidence threshold for spoof detection
            real_threshold: Minimum confidence threshold for real face
        """
        if not (0.0 <= spoof_threshold <= 1.0) or not (0.0 <= real_threshold <= 1.0):
            raise ValueError("Thresholds must be between 0.0 and 1.0")
        
        self.spoof_confidence_threshold = spoof_threshold
        self.real_confidence_threshold = real_threshold
        
        self.logger.info(f"FaceSpoofDetector thresholds updated - Spoof: {spoof_threshold}, Real: {real_threshold}")
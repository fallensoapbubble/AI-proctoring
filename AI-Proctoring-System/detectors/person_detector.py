"""
Person detection component using YOLO for multiple people detection.

This module extracts the person detection logic from backapp.py into a
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


class PersonDetector(DetectionSource):
    """
    Detects multiple people in the frame using YOLO.
    
    This detector identifies when more than one person is present in the
    exam environment, which could indicate unauthorized assistance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the person detector.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.PersonDetector")
        
        # Detection parameters
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.iou_threshold = self.config.get('iou_threshold', 0.45)
        self.model_path = self.config.get('model_path', 'yolov8n.pt')
        self.max_expected_people = self.config.get('max_expected_people', 1)
        
        # Model setup
        self.model = None
        self.is_running = False
        
        # Person class ID in COCO dataset
        self.person_class_id = 0
        
        # Detection history for stability
        self.detection_history = []
        self.history_size = self.config.get('history_size', 5)
        
        # Detection statistics
        self.total_detections = 0
        self.multiple_people_detections = 0
        
        if not YOLO_AVAILABLE:
            self.logger.warning("YOLO not available - install ultralytics package")
        
        self.logger.info("PersonDetector initialized")
    
    def get_source_name(self) -> str:
        """Return the unique name of this detection source."""
        return "person_detector"
    
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
            self.logger.error(f"PersonDetector availability check failed: {e}")
            return False
    
    def start_detection(self) -> None:
        """Load YOLO model for detection."""
        try:
            if YOLO_AVAILABLE:
                self.model = YOLO(self.model_path)
            self.is_running = True
            self.detection_history = []
            self.total_detections = 0
            self.multiple_people_detections = 0
            self.logger.info(f"PersonDetector started with model: {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to start PersonDetector: {e}")
            raise
    
    def stop_detection(self) -> None:
        """Stop detection and cleanup resources."""
        try:
            self.model = None
            self.is_running = False
            self.detection_history = []
            self.logger.info("PersonDetector stopped")
        except Exception as e:
            self.logger.error(f"Error stopping PersonDetector: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Return health status information for monitoring."""
        return {
            'source_name': self.get_source_name(),
            'is_running': self.is_running,
            'is_available': self.is_available(),
            'model_loaded': self.model is not None,
            'model_path': self.model_path,

            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'max_expected_people': self.max_expected_people,
            'detection_history_size': len(self.detection_history),
            'config': self.config
        }
    
    def detect_people(self, frame: np.ndarray) -> Optional[DetectionEvent]:
        """
        Analyze frame for multiple people.
        
        Args:
            frame: Input video frame
            
        Returns:
            DetectionEvent if multiple people detected, None otherwise
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
            self.logger.error(f"Error in person detection: {e}")
            return None
    
    def _detect_with_yolo(self, frame: np.ndarray) -> Optional[DetectionEvent]:
        """Detect people using YOLO."""
        # Run inference with ultralytics YOLO
        results = self.model(frame, verbose=False, conf=self.confidence_threshold, iou=self.iou_threshold)
        
        person_detections = []
        
        # Process detections
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                conf = box.conf[0].item()
                
                if class_id == self.person_class_id:  # Person class
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    detection_info = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'area': (x2 - x1) * (y2 - y1),
                        'center': [(x1 + x2) // 2, (y1 + y2) // 2]
                    }
                    person_detections.append(detection_info)
                    
                    # Draw bounding box for visualization (optional)
                    if self.config.get('draw_boxes', False):
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"Person ({conf:.2f})",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2
                        )
        
        people_count = len(person_detections)
        
        # Update detection history
        self.detection_history.append(people_count)
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)
        
        # Check if we have more people than expected
        if people_count > self.max_expected_people:
            self.multiple_people_detections += 1
            
            # Calculate confidence based on number of people and detection stability
            excess_people = people_count - self.max_expected_people
            base_confidence = min(excess_people * 0.3 + 0.4, 1.0)
            
            # Consider detection stability
            stability_factor = self._calculate_stability_factor(people_count)
            final_confidence = normalize_confidence(base_confidence * stability_factor)
            
            metadata = {
                'detection_method': 'yolo',
                'people_count': people_count,
                'expected_count': self.max_expected_people,
                'excess_people': excess_people,
                'detections': person_detections,
                'stability_factor': float(stability_factor),
                'detection_history': self.detection_history.copy(),
                'model_path': self.model_path,
                'avg_confidence': float(np.mean([d['confidence'] for d in person_detections])) if person_detections else 0.0
            }
            
            return DetectionEvent(
                event_type=DetectionType.MULTIPLE_PEOPLE,
                timestamp=datetime.now(),
                confidence=final_confidence,
                source=self.get_source_name(),
                metadata=metadata,
                frame_data=frame.copy()
            )
        
        return None
    
    def _detect_with_fallback(self, frame: np.ndarray) -> Optional[DetectionEvent]:
        """Fallback person detection using Haar cascades."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use Haar cascade for person detection (upper body)
        try:
            person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
            people = person_cascade.detectMultiScale(gray, 1.1, 4)
            
            people_count = len(people)
            
            # Update detection history
            self.detection_history.append(people_count)
            if len(self.detection_history) > self.history_size:
                self.detection_history.pop(0)
            
            if people_count > self.max_expected_people:
                self.multiple_people_detections += 1
                
                excess_people = people_count - self.max_expected_people
                confidence = min(excess_people * 0.4 + 0.5, 1.0)
                
                person_detections = []
                for (x, y, w, h) in people:
                    person_detections.append({
                        'bbox': [x, y, x + w, y + h],
                        'confidence': 0.7,  # Default confidence for Haar cascade
                        'area': w * h,
                        'center': [x + w // 2, y + h // 2]
                    })
                
                return DetectionEvent(
                    event_type=DetectionType.MULTIPLE_PEOPLE,
                    timestamp=datetime.now(),
                    confidence=confidence,
                    source=self.get_source_name(),
                    metadata={
                        'detection_method': 'fallback',
                        'people_count': people_count,
                        'expected_count': self.max_expected_people,
                        'excess_people': excess_people,
                        'detections': person_detections,
                        'detection_history': self.detection_history.copy()
                    },
                    frame_data=frame.copy()
                )
        
        except Exception as e:
            self.logger.error(f"Error in fallback person detection: {e}")
        
        return None
    
    def _calculate_stability_factor(self, current_count: int) -> float:
        """
        Calculate stability factor based on detection consistency.
        
        Args:
            current_count: Current number of people detected
            
        Returns:
            Stability factor between 0.5 and 1.0
        """
        if len(self.detection_history) < 3:
            return 0.5
        
        # Check consistency of multiple people detections
        recent_history = self.detection_history[-3:]
        multiple_people_frames = sum(1 for count in recent_history if count > self.max_expected_people)
        
        # Calculate consistency ratio
        consistency_ratio = multiple_people_frames / len(recent_history)
        
        # Consider variance in people count
        if len(recent_history) > 1:
            variance = np.var(recent_history)
            variance_factor = max(0.5, 1.0 - variance * 0.1)
        else:
            variance_factor = 1.0
        
        # Combine consistency and variance factors
        stability_factor = 0.5 + (consistency_ratio * variance_factor * 0.5)
        
        return min(stability_factor, 1.0)
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about recent person detections.
        
        Returns:
            Dictionary containing detection statistics
        """
        if not self.detection_history:
            return {
                'total_frames': 0,
                'frames_with_multiple_people': 0,
                'multiple_people_rate': 0.0,
                'avg_people_per_frame': 0.0,
                'max_people_in_frame': 0,
                'min_people_in_frame': 0
            }
        
        frames_with_multiple = sum(1 for count in self.detection_history if count > self.max_expected_people)
        total_people = sum(self.detection_history)
        
        return {
            'total_frames': len(self.detection_history),
            'frames_with_multiple_people': frames_with_multiple,
            'multiple_people_rate': frames_with_multiple / len(self.detection_history),
            'avg_people_per_frame': total_people / len(self.detection_history),
            'max_people_in_frame': max(self.detection_history),
            'min_people_in_frame': min(self.detection_history),
            'expected_people': self.max_expected_people
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
        self.iou_threshold = self.config.get('iou_threshold', self.iou_threshold)
        self.max_expected_people = self.config.get('max_expected_people', self.max_expected_people)
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
        
        self.logger.info(f"PersonDetector configuration updated: {new_config}")
    
    def reset_history(self) -> None:
        """Reset detection history."""
        self.detection_history = []
        self.logger.info("PersonDetector history reset")
    
    def set_expected_people_count(self, count: int) -> None:
        """
        Set the expected number of people in the frame.
        
        Args:
            count: Expected number of people (typically 1 for exams)
        """
        if count < 0:
            raise ValueError("Expected people count must be non-negative")
        
        self.max_expected_people = count
        self.logger.info(f"PersonDetector expected people count set to: {count}")
    
    def get_current_people_count(self) -> int:
        """
        Get the most recent people count.
        
        Returns:
            Number of people detected in the last frame, or 0 if no recent detections
        """
        return self.detection_history[-1] if self.detection_history else 0
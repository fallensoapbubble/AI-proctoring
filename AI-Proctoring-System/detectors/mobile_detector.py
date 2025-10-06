"""
Mobile device detection component using YOLO object detection.

This module extracts the mobile phone detection logic from backapp.py into a
modular, testable component that implements the DetectionSource interface.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from context_engine.interfaces import DetectionSource
from context_engine.models import DetectionEvent, DetectionType
from shared_utils.detection_utils import normalize_confidence


class MobileDetector(DetectionSource):
    """
    Detects mobile phones and electronic devices using YOLO object detection.
    
    This detector identifies mobile phones, tablets, and other electronic devices
    that could be used for cheating during an exam.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the mobile detector.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.MobileDetector")
        
        # Detection parameters
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        self.model_path = self.config.get('model_path', 'yolov8s.pt')
        
        # YOLO model
        self.model = None
        self.is_running = False
        
        # Mobile phone class ID in COCO dataset (cell phone = 67)
        self.mobile_class_ids = self.config.get('mobile_class_ids', [67])
        
        # Detection history for stability
        self.detection_history = []
        self.history_size = self.config.get('history_size', 5)
        
        # Detection statistics
        self.total_detections = 0
        self.mobile_detections = 0
        
        if not YOLO_AVAILABLE:
            self.logger.warning("YOLO not available - install ultralytics package")
        
        self.logger.info("MobileDetector initialized")
    
    def get_source_name(self) -> str:
        """Return the unique name of this detection source."""
        return "mobile_detector"
    
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
            self.logger.error(f"MobileDetector availability check failed: {e}")
            return False
    
    def start_detection(self) -> None:
        """Load YOLO model for detection."""
        try:
            if YOLO_AVAILABLE:
                self.model = YOLO(self.model_path)
            self.is_running = True
            self.detection_history = []
            self.total_detections = 0
            self.mobile_detections = 0
            self.logger.info(f"MobileDetector started with model: {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to start MobileDetector: {e}")
            raise
    
    def stop_detection(self) -> None:
        """Stop detection and cleanup resources."""
        try:
            self.model = None
            self.is_running = False
            self.detection_history = []
            self.logger.info("MobileDetector stopped")
        except Exception as e:
            self.logger.error(f"Error stopping MobileDetector: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Return health status information for monitoring."""
        return {
            'source_name': self.get_source_name(),
            'is_running': self.is_running,
            'is_available': self.is_available(),
            'model_loaded': self.model is not None,
            'model_path': self.model_path,
            'confidence_threshold': self.confidence_threshold,
            'mobile_class_ids': self.mobile_class_ids,
            'detection_history_size': len(self.detection_history),
            'config': self.config
        }
    
    def detect_mobile_devices(self, frame: np.ndarray) -> Optional[DetectionEvent]:
        """
        Analyze frame for mobile devices.
        
        Args:
            frame: Input video frame
            
        Returns:
            DetectionEvent if mobile device detected, None otherwise
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
            self.logger.error(f"Error in mobile device detection: {e}")
            return None
    
    def _detect_with_yolo(self, frame: np.ndarray) -> Optional[DetectionEvent]:
        """Detect mobile devices using YOLO."""
        # Run YOLO inference
        results = self.model(frame, verbose=False)
        
        mobile_detections = []
        
        # Process detection results
        for result in results:
            if result.boxes is None:
                continue
            
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = box.conf[0].item()
                
                # Check if it's a mobile device class with sufficient confidence
                if class_id in self.mobile_class_ids and confidence > self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    detection_info = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_id': class_id,
                        'area': (x2 - x1) * (y2 - y1)
                    }
                    mobile_detections.append(detection_info)
                    
                    # Draw bounding box for visualization (optional)
                    if self.config.get('draw_boxes', False):
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(
                            frame, 
                            f"Mobile ({confidence:.2f})", 
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (255, 0, 0), 
                            2
                        )
        
        # Update detection history
        self.detection_history.append(len(mobile_detections))
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)
        
        # Only trigger if we have consistent detections
        if mobile_detections and self._is_detection_stable():
            self.mobile_detections += 1
            
            # Use the highest confidence detection
            best_detection = max(mobile_detections, key=lambda x: x['confidence'])
            
            # Calculate final confidence considering stability
            stability_factor = self._calculate_stability_factor()
            final_confidence = normalize_confidence(
                best_detection['confidence'] * stability_factor
            )
            
            metadata = {
                'detection_method': 'yolo',
                'detections_count': len(mobile_detections),
                'best_detection': best_detection,
                'all_detections': mobile_detections,
                'stability_factor': float(stability_factor),
                'detection_history': self.detection_history.copy(),
                'model_path': self.model_path,
                'class_ids_detected': list(set(d['class_id'] for d in mobile_detections))
            }
            
            return DetectionEvent(
                event_type=DetectionType.MOBILE_DETECTED,
                timestamp=datetime.now(),
                confidence=final_confidence,
                source=self.get_source_name(),
                metadata=metadata,
                frame_data=frame.copy()
            )
        
        return None
    
    def _detect_with_fallback(self, frame: np.ndarray) -> Optional[DetectionEvent]:
        """Fallback mobile detection using basic image analysis."""
        # Simple fallback: look for rectangular objects that might be phones
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use edge detection to find rectangular objects
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mobile_candidates = []
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter by size and aspect ratio (typical phone dimensions)
            if 1000 < area < 50000:  # Reasonable size range
                aspect_ratio = h / w if w > 0 else 0
                if 1.5 < aspect_ratio < 3.0:  # Phone-like aspect ratio
                    mobile_candidates.append({
                        'bbox': [x, y, x + w, y + h],
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'confidence': min(area / 10000, 1.0)  # Simple confidence based on size
                    })
        
        if mobile_candidates:
            self.mobile_detections += 1
            
            # Use the largest candidate
            best_candidate = max(mobile_candidates, key=lambda x: x['area'])
            
            return DetectionEvent(
                event_type=DetectionType.MOBILE_DETECTED,
                timestamp=datetime.now(),
                confidence=best_candidate['confidence'],
                source=self.get_source_name(),
                metadata={
                    'detection_method': 'fallback',
                    'candidates_count': len(mobile_candidates),
                    'best_candidate': best_candidate,
                    'all_candidates': mobile_candidates
                },
                frame_data=frame.copy()
            )
        
        return None
    
    def _is_detection_stable(self) -> bool:
        """
        Check if mobile device detections are stable over recent frames.
        
        Returns:
            True if detections are stable, False otherwise
        """
        if len(self.detection_history) < 3:
            return False
        
        # Check if we have detections in at least half of recent frames
        recent_detections = self.detection_history[-3:]
        detection_frames = sum(1 for count in recent_detections if count > 0)
        
        return detection_frames >= len(recent_detections) // 2
    
    def _calculate_stability_factor(self) -> float:
        """
        Calculate stability factor based on detection consistency.
        
        Returns:
            Stability factor between 0.5 and 1.0
        """
        if len(self.detection_history) < 2:
            return 0.5
        
        # Calculate consistency of detections
        non_zero_detections = sum(1 for count in self.detection_history if count > 0)
        consistency_ratio = non_zero_detections / len(self.detection_history)
        
        # Map consistency to stability factor (0.5 to 1.0)
        stability_factor = 0.5 + (consistency_ratio * 0.5)
        
        return stability_factor
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about recent mobile device detections.
        
        Returns:
            Dictionary containing detection statistics
        """
        if not self.detection_history:
            return {
                'total_frames': 0,
                'frames_with_detections': 0,
                'detection_rate': 0.0,
                'avg_detections_per_frame': 0.0,
                'max_detections_in_frame': 0
            }
        
        frames_with_detections = sum(1 for count in self.detection_history if count > 0)
        total_detections = sum(self.detection_history)
        
        return {
            'total_frames': len(self.detection_history),
            'frames_with_detections': frames_with_detections,
            'detection_rate': frames_with_detections / len(self.detection_history),
            'avg_detections_per_frame': total_detections / len(self.detection_history),
            'max_detections_in_frame': max(self.detection_history) if self.detection_history else 0
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
        self.mobile_class_ids = self.config.get('mobile_class_ids', self.mobile_class_ids)
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
        
        self.logger.info(f"MobileDetector configuration updated: {new_config}")
    
    def reset_history(self) -> None:
        """Reset detection history."""
        self.detection_history = []
        self.logger.info("MobileDetector history reset")
    
    def set_class_ids(self, class_ids: List[int]) -> None:
        """
        Set the class IDs to detect as mobile devices.
        
        Args:
            class_ids: List of COCO class IDs to detect
        """
        self.mobile_class_ids = class_ids
        self.logger.info(f"MobileDetector class IDs updated: {class_ids}")
    
    def get_supported_classes(self) -> Dict[int, str]:
        """
        Get mapping of supported class IDs to class names.
        
        Returns:
            Dictionary mapping class IDs to names
        """
        # Common COCO classes that might be relevant for cheating detection
        return {
            67: "cell phone",
            72: "tv",
            73: "laptop",
            76: "keyboard",
            77: "mouse",
            78: "remote",
            # Add more as needed
        }
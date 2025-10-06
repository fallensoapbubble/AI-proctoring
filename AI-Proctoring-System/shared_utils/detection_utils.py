"""
Detection-related utility functions and detector classes.

This module provides utilities for processing detection results,
normalizing confidence scores, and formatting detection data
that are shared between different detection components.
It also includes the core detector classes extracted from backapp.py.
"""

import numpy as np
import cv2
import torch
import speech_recognition as sr
import mediapipe as mp
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
from ultralytics import YOLO
import logging


def normalize_confidence(confidence: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Normalize confidence score to specified range.
    
    Args:
        confidence: Raw confidence score
        min_val: Minimum output value
        max_val: Maximum output value
        
    Returns:
        Normalized confidence score
    """
    return max(min_val, min(confidence, max_val))


def calculate_detection_metrics(
    detections: List[Dict[str, Any]],
    time_window_seconds: int = 60
) -> Dict[str, Any]:
    """
    Calculate metrics for a set of detections within a time window.
    
    Args:
        detections: List of detection dictionaries
        time_window_seconds: Time window for metric calculation
        
    Returns:
        Dictionary containing detection metrics
    """
    if not detections:
        return {
            'total_detections': 0,
            'average_confidence': 0.0,
            'detection_rate': 0.0,
            'peak_confidence': 0.0,
            'detection_types': {}
        }
    
    # Filter detections within time window
    current_time = datetime.now()
    cutoff_time = current_time - timedelta(seconds=time_window_seconds)
    
    recent_detections = []
    for detection in detections:
        if 'timestamp' in detection:
            if isinstance(detection['timestamp'], str):
                det_time = datetime.fromisoformat(detection['timestamp'])
            else:
                det_time = detection['timestamp']
            
            if det_time >= cutoff_time:
                recent_detections.append(detection)
    
    if not recent_detections:
        return {
            'total_detections': 0,
            'average_confidence': 0.0,
            'detection_rate': 0.0,
            'peak_confidence': 0.0,
            'detection_types': {}
        }
    
    # Calculate metrics
    confidences = [d.get('confidence', 0.0) for d in recent_detections]
    detection_types = {}
    
    for detection in recent_detections:
        det_type = detection.get('event_type', 'unknown')
        if det_type not in detection_types:
            detection_types[det_type] = 0
        detection_types[det_type] += 1
    
    return {
        'total_detections': len(recent_detections),
        'average_confidence': np.mean(confidences) if confidences else 0.0,
        'detection_rate': len(recent_detections) / time_window_seconds,
        'peak_confidence': max(confidences) if confidences else 0.0,
        'detection_types': detection_types
    }


def format_detection_result(
    detection_type: str,
    confidence: float,
    metadata: Optional[Dict[str, Any]] = None,
    source: str = "unknown"
) -> Dict[str, Any]:
    """
    Format detection result into standardized dictionary.
    
    Args:
        detection_type: Type of detection
        confidence: Confidence score
        metadata: Optional metadata dictionary
        source: Source component name
        
    Returns:
        Formatted detection result dictionary
    """
    return {
        'event_type': detection_type,
        'confidence': normalize_confidence(confidence),
        'timestamp': datetime.now().isoformat(),
        'source': source,
        'metadata': metadata or {}
    }


def calculate_temporal_correlation(
    events: List[Dict[str, Any]],
    max_time_gap_seconds: float = 5.0
) -> float:
    """
    Calculate temporal correlation score for a list of events.
    
    Args:
        events: List of event dictionaries with timestamps
        max_time_gap_seconds: Maximum time gap for correlation
        
    Returns:
        Correlation score between 0.0 and 1.0
    """
    if len(events) < 2:
        return 0.0
    
    # Sort events by timestamp
    sorted_events = sorted(events, key=lambda x: x.get('timestamp', ''))
    
    total_gaps = 0
    valid_gaps = 0
    
    for i in range(1, len(sorted_events)):
        prev_time_str = sorted_events[i-1].get('timestamp', '')
        curr_time_str = sorted_events[i].get('timestamp', '')
        
        try:
            prev_time = datetime.fromisoformat(prev_time_str)
            curr_time = datetime.fromisoformat(curr_time_str)
            
            gap_seconds = (curr_time - prev_time).total_seconds()
            total_gaps += gap_seconds
            
            if gap_seconds <= max_time_gap_seconds:
                valid_gaps += 1
                
        except (ValueError, TypeError):
            continue
    
    if len(sorted_events) <= 1:
        return 0.0
    
    # Calculate correlation based on valid gaps and average gap time
    gap_ratio = valid_gaps / (len(sorted_events) - 1)
    avg_gap = total_gaps / (len(sorted_events) - 1) if len(sorted_events) > 1 else max_time_gap_seconds
    
    # Normalize average gap (closer gaps = higher correlation)
    gap_score = max(0.0, 1.0 - (avg_gap / max_time_gap_seconds))
    
    return gap_ratio * gap_score


def filter_low_confidence_detections(
    detections: List[Dict[str, Any]],
    min_confidence: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Filter out detections below minimum confidence threshold.
    
    Args:
        detections: List of detection dictionaries
        min_confidence: Minimum confidence threshold
        
    Returns:
        Filtered list of detections
    """
    return [
        detection for detection in detections
        if detection.get('confidence', 0.0) >= min_confidence
    ]


def group_detections_by_type(
    detections: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group detections by their event type.
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Dictionary mapping event types to lists of detections
    """
    grouped = {}
    
    for detection in detections:
        event_type = detection.get('event_type', 'unknown')
        if event_type not in grouped:
            grouped[event_type] = []
        grouped[event_type].append(detection)
    
    return grouped


def calculate_confidence_trend(
    detections: List[Dict[str, Any]],
    window_size: int = 5
) -> List[float]:
    """
    Calculate moving average confidence trend.
    
    Args:
        detections: List of detection dictionaries sorted by timestamp
        window_size: Size of moving average window
        
    Returns:
        List of moving average confidence values
    """
    if len(detections) < window_size:
        return [d.get('confidence', 0.0) for d in detections]
    
    confidences = [d.get('confidence', 0.0) for d in detections]
    trend = []
    
    for i in range(len(confidences) - window_size + 1):
        window_avg = np.mean(confidences[i:i + window_size])
        trend.append(window_avg)
    
    return trend


def detect_confidence_spikes(
    detections: List[Dict[str, Any]],
    spike_threshold: float = 0.3
) -> List[int]:
    """
    Detect sudden spikes in confidence scores.
    
    Args:
        detections: List of detection dictionaries
        spike_threshold: Minimum increase to consider a spike
        
    Returns:
        List of indices where spikes occur
    """
    if len(detections) < 2:
        return []
    
    confidences = [d.get('confidence', 0.0) for d in detections]
    spikes = []
    
    for i in range(1, len(confidences)):
        increase = confidences[i] - confidences[i-1]
        if increase >= spike_threshold:
            spikes.append(i)
    
    return spikes


class DetectionProcessor:
    """Base class for all detection processors."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self._is_initialized = False
    
    def is_available(self) -> bool:
        """Check if this detector is available and functional."""
        return self._is_initialized
    
    def get_health_status(self) -> Dict[str, Any]:
        """Return health status information."""
        return {
            'name': self.name,
            'initialized': self._is_initialized,
            'available': self.is_available()
        }


class GazeDetector(DetectionProcessor):
    """Detector for gaze direction and head pose analysis."""
    
    def __init__(self):
        super().__init__("GazeDetector")
        try:
            # Initialize MediaPipe face mesh
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False, 
                max_num_faces=1, 
                refine_landmarks=True
            )
            
            # Head pose model points
            self.model_points = np.array([
                (0.0, 0.0, 0.0),        # Nose
                (-30.0, -125.0, -30.0), # Left eye
                (30.0, -125.0, -30.0),  # Right eye
                (-60.0, -70.0, -60.0),  # Left mouth
                (60.0, -70.0, -60.0),   # Right mouth
                (0.0, -150.0, -25.0)    # Chin
            ], dtype="double")
            
            self._is_initialized = True
            self.logger.info("GazeDetector initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize GazeDetector: {e}")
            self._is_initialized = False
    
    def detect_gaze_away(self, frame: np.ndarray) -> bool:
        """
        Detect if the person is looking away from the camera.
        
        Args:
            frame: Input video frame
            
        Returns:
            True if gaze is away, False otherwise
        """
        if not self._is_initialized:
            return False
        
        try:
            # Process frame with MediaPipe
            results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Convert landmarks to pixel coordinates
                    landmarks_px = [
                        (int(pt.x * frame.shape[1]), int(pt.y * frame.shape[0])) 
                        for pt in face_landmarks.landmark
                    ]
                    
                    # Calculate head pose
                    rotation_vector = self._get_head_pose(frame, landmarks_px)
                    yaw, pitch, roll = self._rotation_vector_to_euler_angles(rotation_vector)
                    
                    # Check if head pose indicates looking away
                    if abs(yaw) > 20 or pitch < -10 or pitch > 15 or roll < 80 or roll > 110:
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in gaze detection: {e}")
            return False
    
    def _get_head_pose(self, frame: np.ndarray, landmarks: List[Tuple[int, int]]) -> np.ndarray:
        """Calculate head pose from facial landmarks."""
        image_points = np.array([
            (landmarks[1][0], landmarks[1][1]),    # Nose
            (landmarks[33][0], landmarks[33][1]),  # Left eye
            (landmarks[263][0], landmarks[263][1]), # Right eye
            (landmarks[61][0], landmarks[61][1]),   # Left mouth
            (landmarks[291][0], landmarks[291][1]), # Right mouth
            (landmarks[152][0], landmarks[152][1])  # Chin
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
        
        return rotation_vector
    
    def _rotation_vector_to_euler_angles(self, rotation_vector: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation vector to Euler angles."""
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        pitch = np.arctan2(-rotation_matrix[2, 0], 
                          np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        
        return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)


class LipMovementDetector(DetectionProcessor):
    """Detector for lip movement analysis."""
    
    def __init__(self):
        super().__init__("LipMovementDetector")
        try:
            # Initialize MediaPipe face mesh
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False, 
                max_num_faces=1, 
                refine_landmarks=True
            )
            
            # Lip landmark indices
            self.UPPER_LIP = [13, 14, 15, 16, 17]
            self.LOWER_LIP = [82, 81, 80, 191, 178]
            
            self.prev_lip_distance = None
            self.lip_movement_threshold = 0.003
            
            self._is_initialized = True
            self.logger.info("LipMovementDetector initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LipMovementDetector: {e}")
            self._is_initialized = False
    
    def detect_lip_movement(self, frame: np.ndarray) -> bool:
        """
        Detect lip movement that might indicate speaking.
        
        Args:
            frame: Input video frame
            
        Returns:
            True if lip movement detected, False otherwise
        """
        if not self._is_initialized:
            return False
        
        try:
            # Process frame with MediaPipe
            results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Calculate current lip distance
                    lip_distance = self._calculate_lip_distance(face_landmarks.landmark)
                    
                    # Check for movement compared to previous frame
                    if (self.prev_lip_distance is not None and 
                        abs(lip_distance - self.prev_lip_distance) > self.lip_movement_threshold):
                        self.prev_lip_distance = lip_distance
                        return True
                    
                    self.prev_lip_distance = lip_distance
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in lip movement detection: {e}")
            return False
    
    def _calculate_lip_distance(self, landmarks) -> float:
        """Calculate average distance between upper and lower lip landmarks."""
        return np.mean([
            np.linalg.norm(
                np.array([landmarks[self.UPPER_LIP[i]].x, landmarks[self.UPPER_LIP[i]].y]) -
                np.array([landmarks[self.LOWER_LIP[i]].x, landmarks[self.LOWER_LIP[i]].y])
            ) for i in range(len(self.UPPER_LIP))
        ])


class SpeechAnalyzer(DetectionProcessor):
    """Analyzer for speech recognition and cheating keyword detection."""
    
    def __init__(self):
        super().__init__("SpeechAnalyzer")
        try:
            self.recognizer = sr.Recognizer()
            self.cheating_keywords = [
                "answer", "solution", "help", "cheat", "google", 
                "search", "tell me", "what is", "how to"
            ]
            
            self._is_initialized = True
            self.logger.info("SpeechAnalyzer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SpeechAnalyzer: {e}")
            self._is_initialized = False
    
    def detect_cheating(self, text: str) -> bool:
        """
        Detect cheating-related keywords in speech text.
        
        Args:
            text: Recognized speech text
            
        Returns:
            True if cheating keywords detected, False otherwise
        """
        if not self._is_initialized or not text:
            return False
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.cheating_keywords)


class ObjectDetector(DetectionProcessor):
    """Detector for objects like mobile phones and face spoofing."""
    
    def __init__(self):
        super().__init__("ObjectDetector")
        try:
            # Initialize YOLO models
            self.face_model = YOLO("yolov8n.pt")
            self.mobile_model = YOLO("yolov8s.pt")
            
            self.FACE_CONF_THRESHOLD = 0.5
            self.MOBILE_CONF_THRESHOLD = 0.6
            
            self._is_initialized = True
            self.logger.info("ObjectDetector initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ObjectDetector: {e}")
            self._is_initialized = False
    
    def detect_mobile_phone(self, frame: np.ndarray) -> bool:
        """
        Detect mobile phones in the frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            True if mobile phone detected, False otherwise
        """
        if not self._is_initialized:
            return False
        
        try:
            results = self.mobile_model(frame)
            
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    conf = box.conf[0].item()
                    
                    # Class 67 is typically cell phone in COCO dataset
                    if class_id == 67 and conf > self.MOBILE_CONF_THRESHOLD:
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in mobile phone detection: {e}")
            return False
    
    def detect_face_spoof(self, frame: np.ndarray) -> bool:
        """
        Detect face spoofing attempts.
        
        Args:
            frame: Input video frame
            
        Returns:
            True if face spoof detected, False otherwise
        """
        if not self._is_initialized:
            return False
        
        try:
            results = self.face_model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), verbose=False)
            
            for r in results:
                for box in r.boxes:
                    conf = box.conf[0].item()
                    if conf < self.FACE_CONF_THRESHOLD:
                        continue
                    
                    # Assuming class 1 is spoof in the trained model
                    if int(box.cls[0]) == 1:
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in face spoof detection: {e}")
            return False


class PersonDetector(DetectionProcessor):
    """Detector for counting people in the frame."""
    
    def __init__(self):
        super().__init__("PersonDetector")
        try:
            # Initialize YOLO for person detection using ultralytics
            self.model = YOLO("yolov8n.pt")  # Use YOLOv8 nano for faster inference
            
            self._is_initialized = True
            self.logger.info("PersonDetector initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize PersonDetector: {e}")
            self._is_initialized = False
    
    def detect_people(self, frame: np.ndarray) -> int:
        """
        Count the number of people in the frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            Number of people detected
        """
        if not self._is_initialized:
            return 0
        
        try:
            # Run inference with ultralytics YOLO
            results = self.model(frame, verbose=False)
            
            # Count people (class 0 in COCO dataset)
            count = 0
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    conf = box.conf[0].item()
                    
                    # Class 0 is person in COCO dataset
                    if class_id == 0 and conf > 0.5:
                        count += 1
            
            return count
            
        except Exception as e:
            self.logger.error(f"Error in person detection: {e}")
            return 0
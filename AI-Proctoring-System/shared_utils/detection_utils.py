"""
Detection Utilities - Common functions for detection processing.
"""

import numpy as np
from typing import Union, List, Tuple, Optional


def normalize_confidence(confidence: Union[float, int]) -> float:
    """
    Normalize confidence score to [0, 1] range.
    
    Args:
        confidence: Raw confidence score
        
    Returns:
        Normalized confidence in [0, 1] range
    """
    try:
        conf_float = float(confidence)
        return max(0.0, min(1.0, conf_float))
    except (ValueError, TypeError):
        return 0.0


def calculate_bbox_area(bbox: Union[List, Tuple]) -> float:
    """
    Calculate area of bounding box.
    
    Args:
        bbox: Bounding box as [x, y, width, height] or [x1, y1, x2, y2]
        
    Returns:
        Area of bounding box
    """
    if len(bbox) != 4:
        return 0.0
    
    try:
        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            # Format: [x1, y1, x2, y2]
            return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        else:
            # Format: [x, y, width, height]
            return bbox[2] * bbox[3]
    except (ValueError, TypeError, IndexError):
        return 0.0


def calculate_bbox_overlap(bbox1: Union[List, Tuple], bbox2: Union[List, Tuple]) -> float:
    """
    Calculate overlap ratio between two bounding boxes.
    
    Args:
        bbox1: First bounding box [x, y, width, height]
        bbox2: Second bounding box [x, y, width, height]
        
    Returns:
        Overlap ratio [0, 1]
    """
    try:
        # Convert to [x1, y1, x2, y2] format
        x1_1, y1_1, x2_1, y2_1 = bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]
        x1_2, y1_2, x2_2, y2_2 = bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        if union_area <= 0:
            return 0.0
        
        return intersection_area / union_area
        
    except (ValueError, TypeError, IndexError, ZeroDivisionError):
        return 0.0


def smooth_confidence_sequence(confidences: List[float], window_size: int = 3) -> List[float]:
    """
    Apply smoothing to a sequence of confidence scores.
    
    Args:
        confidences: List of confidence scores
        window_size: Size of smoothing window
        
    Returns:
        Smoothed confidence scores
    """
    if not confidences or window_size <= 1:
        return confidences
    
    smoothed = []
    half_window = window_size // 2
    
    for i in range(len(confidences)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(confidences), i + half_window + 1)
        
        window_values = confidences[start_idx:end_idx]
        smoothed_value = sum(window_values) / len(window_values)
        smoothed.append(smoothed_value)
    
    return smoothed


def calculate_detection_stability(recent_detections: List[bool], min_stability: float = 0.6) -> float:
    """
    Calculate stability score for recent detections.
    
    Args:
        recent_detections: List of boolean detection results
        min_stability: Minimum stability threshold
        
    Returns:
        Stability score [0, 1]
    """
    if not recent_detections:
        return 0.0
    
    # Calculate consistency
    positive_detections = sum(recent_detections)
    consistency_ratio = positive_detections / len(recent_detections)
    
    # Apply minimum stability threshold
    if consistency_ratio < min_stability:
        return 0.0
    
    return consistency_ratio


def filter_detections_by_confidence(detections: List[dict], min_confidence: float = 0.5) -> List[dict]:
    """
    Filter detections by minimum confidence threshold.
    
    Args:
        detections: List of detection dictionaries with 'confidence' key
        min_confidence: Minimum confidence threshold
        
    Returns:
        Filtered detections
    """
    filtered = []
    for detection in detections:
        try:
            confidence = detection.get('confidence', 0.0)
            if normalize_confidence(confidence) >= min_confidence:
                filtered.append(detection)
        except (KeyError, TypeError):
            continue
    
    return filtered


def calculate_frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Calculate normalized difference between two frames.
    
    Args:
        frame1: First frame
        frame2: Second frame
        
    Returns:
        Normalized difference [0, 1]
    """
    try:
        if frame1.shape != frame2.shape:
            return 1.0
        
        # Convert to grayscale if needed
        if len(frame1.shape) == 3:
            frame1_gray = np.mean(frame1, axis=2)
            frame2_gray = np.mean(frame2, axis=2)
        else:
            frame1_gray = frame1
            frame2_gray = frame2
        
        # Calculate mean absolute difference
        diff = np.mean(np.abs(frame1_gray - frame2_gray))
        
        # Normalize to [0, 1] range (assuming 8-bit images)
        normalized_diff = diff / 255.0
        
        return min(1.0, normalized_diff)
        
    except Exception:
        return 1.0  # Maximum difference on error
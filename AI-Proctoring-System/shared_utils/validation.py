"""
Validation utilities for the AI proctoring system.

This module provides validation functions for various data structures
and configurations used throughout the system.
"""

from typing import Dict, Any, List, Tuple
import re


def validate_detection_threshold(threshold: float) -> bool:
    """
    Validate detection threshold value.
    
    Args:
        threshold: Threshold value to validate
        
    Returns:
        True if valid, False otherwise
    """
    return isinstance(threshold, (int, float)) and 0.0 <= threshold <= 1.0


def validate_confidence_score(confidence: float) -> bool:
    """
    Validate confidence score value.
    
    Args:
        confidence: Confidence score to validate
        
    Returns:
        True if valid, False otherwise
    """
    return isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0


def validate_session_id(session_id: str) -> bool:
    """
    Validate session ID format.
    
    Args:
        session_id: Session ID to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(session_id, str):
        return False
    
    # Check if it's a valid UUID format
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    return bool(re.match(uuid_pattern, session_id, re.IGNORECASE))


def validate_alert_data(alert_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate alert data structure.
    
    Args:
        alert_dict: Alert data dictionary
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Required fields
    required_fields = ['alert_id', 'timestamp', 'severity', 'primary_detection']
    for field in required_fields:
        if field not in alert_dict:
            errors.append(f"Missing required field: {field}")
    
    # Validate severity
    valid_severities = ['low', 'medium', 'high', 'critical']
    if 'severity' in alert_dict and alert_dict['severity'] not in valid_severities:
        errors.append(f"Invalid severity: {alert_dict['severity']}")
    
    # Validate confidence breakdown
    if 'confidence_breakdown' in alert_dict:
        for key, value in alert_dict['confidence_breakdown'].items():
            if not validate_confidence_score(value):
                errors.append(f"Invalid confidence score for {key}: {value}")
    
    return len(errors) == 0, errors


def validate_system_configuration(config_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate system configuration data.
    
    Args:
        config_dict: Configuration data dictionary
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Validate detection thresholds
    if 'detection_thresholds' in config_dict:
        for detection_type, threshold in config_dict['detection_thresholds'].items():
            if not validate_detection_threshold(threshold):
                errors.append(f"Invalid threshold for {detection_type}: {threshold}")
    
    # Validate time windows
    time_fields = ['correlation_window_seconds', 'evidence_retention_days']
    for field in time_fields:
        if field in config_dict:
            value = config_dict[field]
            if not isinstance(value, int) or value <= 0:
                errors.append(f"Invalid {field}: {value}")
    
    return len(errors) == 0, errors


def validate_detection_event(event_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate detection event data.
    
    Args:
        event_dict: Detection event data dictionary
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Required fields
    required_fields = ['event_type', 'timestamp', 'confidence', 'source']
    for field in required_fields:
        if field not in event_dict:
            errors.append(f"Missing required field: {field}")
    
    # Validate confidence
    if 'confidence' in event_dict:
        if not validate_confidence_score(event_dict['confidence']):
            errors.append(f"Invalid confidence score: {event_dict['confidence']}")
    
    # Validate event type
    valid_event_types = [
        'gaze_away', 'lip_movement', 'suspicious_speech', 'multiple_people',
        'mobile_detected', 'face_spoof', 'head_pose_suspicious', 'audio_anomaly'
    ]
    if 'event_type' in event_dict and event_dict['event_type'] not in valid_event_types:
        errors.append(f"Invalid event type: {event_dict['event_type']}")
    
    return len(errors) == 0, errors


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace unsafe characters
    unsafe_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(unsafe_chars, '_', filename)
    
    # Limit length
    if len(sanitized) > 255:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        max_name_len = 255 - len(ext) - 1 if ext else 255
        sanitized = name[:max_name_len] + ('.' + ext if ext else '')
    
    return sanitized


def validate_file_path(file_path: str) -> bool:
    """
    Validate file path for security.
    
    Args:
        file_path: File path to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(file_path, str):
        return False
    
    # Check for path traversal attempts
    if '..' in file_path or file_path.startswith('/'):
        return False
    
    # Check for null bytes
    if '\x00' in file_path:
        return False
    
    return True
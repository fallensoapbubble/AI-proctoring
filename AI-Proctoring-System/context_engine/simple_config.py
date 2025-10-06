"""
Simple System Configuration - Minimal implementation for basic functionality.

This is a simplified version that provides basic configuration capabilities.
"""

from typing import Dict, Any


class SimpleSystemConfiguration:
    """
    Simplified system configuration that provides basic settings.
    """
    
    def __init__(self):
        """Initialize simple configuration."""
        self.detection_thresholds = {
            'gaze_away': 0.7,
            'lip_movement': 0.6,
            'suspicious_speech': 0.8,
            'multiple_people': 0.9,
            'mobile_detected': 0.8,
            'face_spoof': 0.9
        }
        
        self.correlation_window_seconds = 5
        self.max_correlation_events = 10
        self.evidence_retention_days = 30
        self.max_evidence_file_size_mb = 100
        self.deployment_mode = 'development'
        self.max_processing_latency_ms = 500
        self.enable_gpu_acceleration = False
        
        print("SimpleSystemConfiguration initialized")
    
    def get_threshold(self, detection_type) -> float:
        """Get threshold for detection type."""
        if hasattr(detection_type, 'value'):
            return self.detection_thresholds.get(detection_type.value, 0.5)
        return self.detection_thresholds.get(str(detection_type), 0.5)
    
    def update_threshold(self, detection_type, threshold: float):
        """Update threshold for detection type."""
        if hasattr(detection_type, 'value'):
            self.detection_thresholds[detection_type.value] = threshold
        else:
            self.detection_thresholds[str(detection_type)] = threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'detection_thresholds': self.detection_thresholds,
            'correlation_window_seconds': self.correlation_window_seconds,
            'max_correlation_events': self.max_correlation_events,
            'evidence_retention_days': self.evidence_retention_days,
            'max_evidence_file_size_mb': self.max_evidence_file_size_mb,
            'deployment_mode': self.deployment_mode,
            'max_processing_latency_ms': self.max_processing_latency_ms,
            'enable_gpu_acceleration': self.enable_gpu_acceleration
        }
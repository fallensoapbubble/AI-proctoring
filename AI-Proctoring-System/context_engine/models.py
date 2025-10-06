"""
Core data models for the context-aware alerts system.

This module defines the fundamental data structures used throughout
the context analysis engine for detection events, analysis results,
alerts, and system configuration.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import numpy as np
import uuid


class DetectionType(Enum):
    """Types of detection events that can be processed by the system."""
    GAZE_AWAY = "gaze_away"
    LIP_MOVEMENT = "lip_movement"
    SUSPICIOUS_SPEECH = "suspicious_speech"
    MULTIPLE_PEOPLE = "multiple_people"
    MOBILE_DETECTED = "mobile_detected"
    FACE_SPOOF = "face_spoof"
    HEAD_POSE_SUSPICIOUS = "head_pose_suspicious"
    AUDIO_ANOMALY = "audio_anomaly"


class AlertSeverity(Enum):
    """Alert severity levels for graduated response."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertRecommendation(Enum):
    """Recommended actions based on analysis results."""
    IGNORE = "ignore"
    MONITOR = "monitor"
    ALERT_LOW = "alert_low"
    ALERT_MEDIUM = "alert_medium"
    ALERT_HIGH = "alert_high"
    IMMEDIATE_INTERVENTION = "immediate_intervention"


class DeploymentMode(Enum):
    """System deployment modes for configuration management."""
    DEVELOPMENT = "development"
    DOCKER = "docker"
    PRODUCTION = "production"


@dataclass
class DetectionEvent:
    """
    Standardized format for all detection signals from various sources.
    
    This class represents a single detection event with metadata,
    confidence scores, and optional frame data for evidence collection.
    """
    event_type: DetectionType
    timestamp: datetime
    confidence: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    frame_data: Optional[np.ndarray] = None
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __post_init__(self):
        """Validate detection event data after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        if not isinstance(self.timestamp, datetime):
            raise ValueError("Timestamp must be a datetime object")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection event to dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence,
            'source': self.source,
            'metadata': self.metadata,
            'has_frame_data': self.frame_data is not None
        }


@dataclass
class CorrelatedEvent:
    """Represents a group of temporally related detection events."""
    primary_event: DetectionEvent
    related_events: List[DetectionEvent]
    correlation_score: float
    time_span_seconds: float
    
    def get_all_events(self) -> List[DetectionEvent]:
        """Get all events including primary and related events."""
        return [self.primary_event] + self.related_events


@dataclass
class AnalysisResult:
    """
    Result of contextual analysis performed on detection events.
    
    Contains the analysis outcome, confidence scores, and reasoning
    for alert decision making.
    """
    primary_event: DetectionEvent
    correlated_events: List[DetectionEvent] = field(default_factory=list)
    confidence_score: float = 0.0
    contextual_factors: Dict[str, float] = field(default_factory=dict)
    recommendation: AlertRecommendation = AlertRecommendation.IGNORE
    reasoning: str = ""
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate analysis result data."""
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError(f"Confidence score must be between 0.0 and 1.0, got {self.confidence_score}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis result to dictionary for serialization."""
        return {
            'primary_event': self.primary_event.to_dict(),
            'correlated_events': [event.to_dict() for event in self.correlated_events],
            'confidence_score': self.confidence_score,
            'contextual_factors': self.contextual_factors,
            'recommendation': self.recommendation.value,
            'reasoning': self.reasoning,
            'analysis_timestamp': self.analysis_timestamp.isoformat()
        }


@dataclass
class Alert:
    """
    Represents a generated alert with evidence and contextual information.
    
    Contains all information needed for administrator review including
    severity, evidence files, and detailed reasoning.
    """
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    severity: AlertSeverity = AlertSeverity.LOW
    primary_detection: DetectionType = DetectionType.GAZE_AWAY
    contributing_detections: List[DetectionType] = field(default_factory=list)
    confidence_breakdown: Dict[str, float] = field(default_factory=dict)
    evidence_files: List[str] = field(default_factory=list)
    contextual_reasoning: str = ""
    session_id: Optional[str] = None
    student_id: Optional[str] = None
    resolved: bool = False
    
    def validate_alert(self) -> tuple[bool, list[str]]:
        """
        Validate the current alert data.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        from shared_utils.validation import validate_alert_data
        alert_dict = self.to_dict()
        return validate_alert_data(alert_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity.value,
            'primary_detection': self.primary_detection.value,
            'contributing_detections': [det.value for det in self.contributing_detections],
            'confidence_breakdown': self.confidence_breakdown,
            'evidence_files': self.evidence_files,
            'contextual_reasoning': self.contextual_reasoning,
            'session_id': self.session_id,
            'student_id': self.student_id,
            'resolved': self.resolved
        }


@dataclass
class CombinationRule:
    """Defines how different detection types should be combined."""
    detection_types: List[DetectionType]
    weight_multiplier: float
    time_window_seconds: int
    minimum_confidence: float
    
    def applies_to(self, events: List[DetectionEvent]) -> bool:
        """Check if this rule applies to the given events."""
        event_types = {event.event_type for event in events}
        rule_types = set(self.detection_types)
        return rule_types.issubset(event_types)


@dataclass
class SystemConfiguration:
    """
    System-wide configuration for detection thresholds and behavior.
    
    Manages all configurable aspects of the context analysis engine
    including detection sensitivity, correlation windows, and deployment settings.
    """
    # Detection thresholds for individual signal types
    detection_thresholds: Dict[DetectionType, float] = field(default_factory=lambda: {
        DetectionType.GAZE_AWAY: 0.7,
        DetectionType.LIP_MOVEMENT: 0.6,
        DetectionType.SUSPICIOUS_SPEECH: 0.8,
        DetectionType.MULTIPLE_PEOPLE: 0.9,
        DetectionType.MOBILE_DETECTED: 0.8,
        DetectionType.FACE_SPOOF: 0.9,
        DetectionType.HEAD_POSE_SUSPICIOUS: 0.6,
        DetectionType.AUDIO_ANOMALY: 0.7
    })
    
    # Temporal correlation settings
    correlation_window_seconds: int = 5
    max_correlation_events: int = 10
    
    # Alert combination rules
    alert_combination_rules: Dict[str, CombinationRule] = field(default_factory=dict)
    
    # Evidence and storage settings
    evidence_retention_days: int = 30
    max_evidence_file_size_mb: int = 100
    
    # System deployment settings
    deployment_mode: DeploymentMode = DeploymentMode.DEVELOPMENT
    
    # Performance settings
    max_processing_latency_ms: int = 500
    enable_gpu_acceleration: bool = True
    
    def __post_init__(self):
        """Initialize default combination rules if not provided."""
        if not self.alert_combination_rules:
            self.alert_combination_rules = self._create_default_combination_rules()
    
    def _create_default_combination_rules(self) -> Dict[str, CombinationRule]:
        """Create default combination rules for common detection patterns."""
        return {
            'gaze_and_speech': CombinationRule(
                detection_types=[DetectionType.GAZE_AWAY, DetectionType.SUSPICIOUS_SPEECH],
                weight_multiplier=1.5,
                time_window_seconds=3,
                minimum_confidence=0.6
            ),
            'lip_and_audio': CombinationRule(
                detection_types=[DetectionType.LIP_MOVEMENT, DetectionType.AUDIO_ANOMALY],
                weight_multiplier=1.3,
                time_window_seconds=2,
                minimum_confidence=0.5
            ),
            'multiple_people_mobile': CombinationRule(
                detection_types=[DetectionType.MULTIPLE_PEOPLE, DetectionType.MOBILE_DETECTED],
                weight_multiplier=2.0,
                time_window_seconds=5,
                minimum_confidence=0.7
            )
        }
    
    def get_threshold(self, detection_type: DetectionType) -> float:
        """Get the confidence threshold for a specific detection type."""
        return self.detection_thresholds.get(detection_type, 0.5)
    
    def update_threshold(self, detection_type: DetectionType, threshold: float) -> None:
        """Update the confidence threshold for a detection type."""
        from shared_utils.validation import validate_detection_threshold
        if not validate_detection_threshold(threshold):
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        self.detection_thresholds[detection_type] = threshold
    
    def validate_configuration(self) -> tuple[bool, list[str]]:
        """
        Validate the current configuration settings.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        from shared_utils.validation import validate_system_configuration
        config_dict = self.to_dict()
        return validate_system_configuration(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            'detection_thresholds': {dt.value: threshold for dt, threshold in self.detection_thresholds.items()},
            'correlation_window_seconds': self.correlation_window_seconds,
            'max_correlation_events': self.max_correlation_events,
            'evidence_retention_days': self.evidence_retention_days,
            'max_evidence_file_size_mb': self.max_evidence_file_size_mb,
            'deployment_mode': self.deployment_mode.value,
            'max_processing_latency_ms': self.max_processing_latency_ms,
            'enable_gpu_acceleration': self.enable_gpu_acceleration
        }
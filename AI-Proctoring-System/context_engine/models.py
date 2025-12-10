"""
Context Engine Models - Data models for detection events and analysis results.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
import numpy as np


class DetectionType(Enum):
    """Enumeration of detection types."""
    GAZE_AWAY = "GAZE_AWAY"
    MOBILE_DETECTED = "MOBILE_DETECTED"
    SPOOF_DETECTED = "SPOOF_DETECTED"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    SPEECH_DETECTED = "SPEECH_DETECTED"
    NO_FACE = "NO_FACE"
    PERSON_COUNT_VIOLATION = "PERSON_COUNT_VIOLATION"
    AUDIO_ANOMALY = "AUDIO_ANOMALY"
    UNKNOWN = "UNKNOWN"


class AlertRecommendation(Enum):
    """Enumeration of alert recommendations."""
    IGNORE = "ignore"
    MONITOR = "monitor"
    ALERT = "alert"
    IMMEDIATE_ACTION = "immediate_action"


class DetectionEvent:
    """
    Represents a single detection event from any detection source.
    """
    
    def __init__(
        self,
        event_type: DetectionType,
        timestamp: datetime,
        confidence: float,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        frame_data: Optional[np.ndarray] = None,
        audio_data: Optional[bytes] = None
    ):
        """Initialize a detection event."""
        self.event_type = event_type
        self.timestamp = timestamp
        self.confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        self.source = source
        self.metadata = metadata or {}
        self.frame_data = frame_data
        self.audio_data = audio_data
        
        # Generate unique ID
        self.event_id = f"{source}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value if isinstance(self.event_type, DetectionType) else str(self.event_type),
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence,
            'source': self.source,
            'metadata': self.metadata,
            'has_frame_data': self.frame_data is not None,
            'has_audio_data': self.audio_data is not None
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"DetectionEvent({self.event_type.value}, {self.confidence:.2f}, {self.source})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"DetectionEvent(event_type={self.event_type}, "
                f"timestamp={self.timestamp}, confidence={self.confidence}, "
                f"source='{self.source}', metadata_keys={list(self.metadata.keys())})")


class AnalysisResult:
    """
    Represents the result of context analysis on one or more detection events.
    """
    
    def __init__(
        self,
        primary_event: DetectionEvent,
        correlated_events: List[DetectionEvent],
        confidence_score: float,
        contextual_factors: Dict[str, Any],
        recommendation: AlertRecommendation,
        reasoning: str
    ):
        """Initialize analysis result."""
        self.primary_event = primary_event
        self.correlated_events = correlated_events
        self.confidence_score = max(0.0, min(1.0, confidence_score))
        self.contextual_factors = contextual_factors
        self.recommendation = recommendation
        self.reasoning = reasoning
        self.analysis_timestamp = datetime.now()
        
        # Generate unique ID
        self.analysis_id = f"analysis_{self.analysis_timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'analysis_id': self.analysis_id,
            'primary_event': self.primary_event.to_dict(),
            'correlated_events': [event.to_dict() for event in self.correlated_events],
            'confidence_score': self.confidence_score,
            'contextual_factors': self.contextual_factors,
            'recommendation': self.recommendation.value,
            'reasoning': self.reasoning,
            'analysis_timestamp': self.analysis_timestamp.isoformat()
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"AnalysisResult({self.recommendation.value}, {self.confidence_score:.2f})"


class SessionContext:
    """
    Maintains context information for a proctoring session.
    """
    
    def __init__(self, session_id: str):
        """Initialize session context."""
        self.session_id = session_id
        self.start_time = datetime.now()
        self.events: List[DetectionEvent] = []
        self.analysis_results: List[AnalysisResult] = []
        self.metadata: Dict[str, Any] = {}
        self.is_active = True
    
    def add_event(self, event: DetectionEvent) -> None:
        """Add a detection event to the session."""
        self.events.append(event)
    
    def add_analysis_result(self, result: AnalysisResult) -> None:
        """Add an analysis result to the session."""
        self.analysis_results.append(result)
    
    def get_recent_events(self, seconds: int = 30) -> List[DetectionEvent]:
        """Get events from the last N seconds."""
        cutoff_time = datetime.now().timestamp() - seconds
        return [
            event for event in self.events
            if event.timestamp.timestamp() > cutoff_time
        ]
    
    def get_event_counts_by_type(self) -> Dict[str, int]:
        """Get count of events by type."""
        counts = {}
        for event in self.events:
            event_type = event.event_type.value if isinstance(event.event_type, DetectionType) else str(event.event_type)
            counts[event_type] = counts.get(event_type, 0) + 1
        return counts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'event_count': len(self.events),
            'analysis_count': len(self.analysis_results),
            'event_counts_by_type': self.get_event_counts_by_type(),
            'metadata': self.metadata,
            'is_active': self.is_active
        }
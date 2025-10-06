"""
Base interfaces and abstract classes for the context analysis engine.

This module defines the contracts that all components must implement
to ensure consistent behavior and enable extensibility.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .models import DetectionEvent, AnalysisResult, Alert, SystemConfiguration


class DetectionSource(ABC):
    """
    Abstract base class for all detection sources.
    
    Detection sources are components that generate DetectionEvent objects
    from various inputs (camera, microphone, etc.).
    """
    
    @abstractmethod
    def get_source_name(self) -> str:
        """Return the unique name of this detection source."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this detection source is available and functional."""
        pass
    
    @abstractmethod
    def start_detection(self) -> None:
        """Start the detection process."""
        pass
    
    @abstractmethod
    def stop_detection(self) -> None:
        """Stop the detection process and cleanup resources."""
        pass
    
    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Return health status information for monitoring."""
        pass


class EventProcessor(ABC):
    """
    Abstract base class for components that process detection events.
    
    Event processors take DetectionEvent objects and perform analysis,
    correlation, or other processing operations.
    """
    
    @abstractmethod
    def process_event(self, event: DetectionEvent) -> Optional[AnalysisResult]:
        """
        Process a single detection event.
        
        Args:
            event: The detection event to process
            
        Returns:
            Analysis result if processing generates output, None otherwise
        """
        pass
    
    @abstractmethod
    def process_batch(self, events: List[DetectionEvent]) -> List[AnalysisResult]:
        """
        Process a batch of detection events.
        
        Args:
            events: List of detection events to process
            
        Returns:
            List of analysis results
        """
        pass


class AlertHandler(ABC):
    """
    Abstract base class for alert handling components.
    
    Alert handlers are responsible for processing generated alerts
    and taking appropriate actions (notifications, storage, etc.).
    """
    
    @abstractmethod
    def handle_alert(self, alert: Alert) -> bool:
        """
        Handle a generated alert.
        
        Args:
            alert: The alert to handle
            
        Returns:
            True if alert was handled successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def get_alert_history(self, limit: Optional[int] = None) -> List[Alert]:
        """
        Retrieve alert history.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of historical alerts
        """
        pass


class ConfigurationProvider(ABC):
    """
    Abstract base class for configuration providers.
    
    Configuration providers supply system configuration from various
    sources (files, environment variables, databases, etc.).
    """
    
    @abstractmethod
    def load_configuration(self) -> SystemConfiguration:
        """Load and return system configuration."""
        pass
    
    @abstractmethod
    def save_configuration(self, config: SystemConfiguration) -> bool:
        """
        Save system configuration.
        
        Args:
            config: Configuration to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def watch_for_changes(self, callback) -> None:
        """
        Watch for configuration changes and call callback when detected.
        
        Args:
            callback: Function to call when configuration changes
        """
        pass


class EvidenceCollector(ABC):
    """
    Abstract base class for evidence collection components.
    
    Evidence collectors are responsible for capturing and storing
    evidence related to detection events and alerts.
    """
    
    @abstractmethod
    def collect_evidence(self, event: DetectionEvent) -> List[str]:
        """
        Collect evidence for a detection event.
        
        Args:
            event: Detection event to collect evidence for
            
        Returns:
            List of evidence file paths
        """
        pass
    
    @abstractmethod
    def store_evidence(self, evidence_data: bytes, filename: str) -> str:
        """
        Store evidence data to persistent storage.
        
        Args:
            evidence_data: Raw evidence data
            filename: Desired filename for the evidence
            
        Returns:
            Path to stored evidence file
        """
        pass
    
    @abstractmethod
    def cleanup_old_evidence(self, retention_days: int) -> int:
        """
        Clean up evidence older than specified retention period.
        
        Args:
            retention_days: Number of days to retain evidence
            
        Returns:
            Number of evidence files cleaned up
        """
        pass


class PerformanceMonitor(ABC):
    """
    Abstract base class for performance monitoring components.
    
    Performance monitors track system metrics and provide
    insights into system health and performance.
    """
    
    @abstractmethod
    def record_processing_time(self, component: str, duration_ms: float) -> None:
        """
        Record processing time for a component.
        
        Args:
            component: Name of the component
            duration_ms: Processing duration in milliseconds
        """
        pass
    
    @abstractmethod
    def record_event_count(self, event_type: str, count: int = 1) -> None:
        """
        Record event count for metrics.
        
        Args:
            event_type: Type of event
            count: Number of events (default: 1)
        """
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        pass
    
    @abstractmethod
    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        pass
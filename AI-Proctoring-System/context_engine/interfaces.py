"""
Context Engine Interfaces - Base classes for detection sources and analyzers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class DetectionSource(ABC):
    """
    Abstract base class for all detection sources.
    """
    
    @abstractmethod
    def get_source_name(self) -> str:
        """Return the unique name of this detection source."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this detection source is available and can be used."""
        pass
    
    @abstractmethod
    def start_detection(self) -> None:
        """Start the detection process."""
        pass
    
    @abstractmethod
    def stop_detection(self) -> None:
        """Stop the detection process and cleanup resources."""
        pass
    
    def get_health_status(self) -> Dict[str, Any]:
        """Return health status information for monitoring."""
        return {
            'source_name': self.get_source_name(),
            'is_available': self.is_available(),
            'status': 'unknown'
        }
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """Update configuration for this detection source."""
        pass


class ContextAnalyzer(ABC):
    """
    Abstract base class for context analyzers.
    """
    
    @abstractmethod
    def process_event(self, event) -> Optional[Any]:
        """Process a single detection event and return analysis result."""
        pass
    
    def process_batch(self, events: List) -> List:
        """Process a batch of events."""
        results = []
        for event in events:
            result = self.process_event(event)
            if result:
                results.append(result)
        return results
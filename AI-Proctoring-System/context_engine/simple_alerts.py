"""
Simple Alert Manager - Minimal implementation for basic functionality.

This is a simplified version that provides basic alert functionality
without requiring complex dependencies.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from .models import Alert, AlertSeverity, DetectionType


class SimpleAlertManager:
    """
    Simplified alert manager that provides basic alert functionality.
    """
    
    def __init__(self, config=None):
        """Initialize the simple alert manager."""
        self.config = config
        self.alerts = []
        print("SimpleAlertManager initialized")
    
    def generate_alert(self, analysis_result, session_id=None, student_id=None) -> Optional[Alert]:
        """Generate a simple alert."""
        try:
            # Create basic alert
            alert = Alert(
                timestamp=datetime.now(),
                severity=AlertSeverity.LOW,
                primary_detection=DetectionType.GAZE_AWAY,  # Default
                contributing_detections=[],
                confidence_breakdown={'overall': 0.5},
                evidence_files=[],
                contextual_reasoning="Simple alert generated",
                session_id=session_id,
                student_id=student_id
            )
            
            self.alerts.append(alert)
            return alert
            
        except Exception as e:
            print(f"Error generating simple alert: {e}")
            return None
    
    def get_alert_history(self, limit: Optional[int] = None) -> List[Alert]:
        """Get alert history."""
        if limit:
            return self.alerts[-limit:]
        return self.alerts.copy()
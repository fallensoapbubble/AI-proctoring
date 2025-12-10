"""
Shared Utilities Package - Common utilities for the AI Proctoring System.
"""

# Make detection_utils available at package level
try:
    from .detection_utils import normalize_confidence
except ImportError:
    # Fallback implementation
    def normalize_confidence(confidence):
        """Normalize confidence to [0, 1] range."""
        return max(0.0, min(1.0, float(confidence)))

__all__ = ['normalize_confidence']
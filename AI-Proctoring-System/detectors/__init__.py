"""
Detection components module for the AI Proctoring System.

This module contains the core detection components for the proctoring system.
"""

# Import fixed detectors (core implementation)
from .fixed_gaze_detector import FixedGazeDetector
from .fixed_mobile_detector import FixedMobileDetector
from .fixed_detector_manager import FixedDetectorManager

__all__ = [
    'FixedGazeDetector',
    'FixedMobileDetector',
    'FixedDetectorManager'
]
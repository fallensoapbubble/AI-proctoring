"""
Detection components module for the AI Proctoring System.

This module contains modular detection components that extract specific
detection logic from the monolithic backend into reusable, testable classes.
"""

# Import available detectors with error handling
try:
    from .gaze_detector import GazeDetector
except ImportError:
    GazeDetector = None

try:
    from .lip_movement_detector import LipMovementDetector
except ImportError:
    LipMovementDetector = None

try:
    from .speech_analyzer import SpeechAnalyzer
except ImportError:
    SpeechAnalyzer = None

try:
    from .mobile_detector import MobileDetector
except ImportError:
    MobileDetector = None

try:
    from .person_detector import PersonDetector
except ImportError:
    PersonDetector = None

try:
    from .face_spoof_detector import FaceSpoofDetector
except ImportError:
    FaceSpoofDetector = None

# Import fixed detectors (these should always work)
from .fixed_gaze_detector import FixedGazeDetector
from .fixed_mobile_detector import FixedMobileDetector
from .fixed_detector_manager import FixedDetectorManager

# Import simple detector manager
try:
    from .simple_detector_manager import SimpleDetectorManager
except ImportError:
    SimpleDetectorManager = None

__all__ = [
    'GazeDetector',
    'LipMovementDetector', 
    'SpeechAnalyzer',
    'MobileDetector',
    'PersonDetector',
    'FaceSpoofDetector',
    'FixedGazeDetector',
    'FixedMobileDetector',
    'FixedDetectorManager',
    'SimpleDetectorManager'
]
"""
Context-Aware Alerts Engine

This module provides intelligent multi-modal analysis for AI proctoring systems.
It combines detection signals from various sources to reduce false positives
and improve detection accuracy through contextual reasoning.
"""

__version__ = "1.0.0"
__author__ = "AI Proctoring System"

from .models import DetectionEvent, AnalysisResult, Alert, SystemConfiguration

# Import other components when they are implemented
# from .analyzer import ContextCueAnalyzer
# from .correlation import CorrelationManager
# from .config import ConfigurationService
# from .alerts import AlertManager

__all__ = [
    'DetectionEvent',
    'AnalysisResult', 
    'Alert',
    'SystemConfiguration',
    # 'ContextCueAnalyzer',
    # 'CorrelationManager',
    # 'ConfigurationService',
    # 'AlertManager'
]
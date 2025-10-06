"""
Shared utilities module for the AI Proctoring System.

This module contains common functionality used across both the Flask web application
and the computer vision backend to eliminate code duplication and ensure consistency.
"""

from .common import *
from .file_utils import *
from .validation import *
# Note: detection_utils not imported by default due to heavy dependencies (cv2, torch, etc.)
# Import detection_utils explicitly when needed

__all__ = [
    # Common utilities
    'get_timestamp',
    'generate_session_id',
    'setup_logging',
    'safe_json_loads',
    
    # Detection utilities (import detection_utils explicitly when needed)
    # 'normalize_confidence',
    # 'calculate_detection_metrics', 
    # 'format_detection_result',
    
    # File utilities
    'ensure_directory_exists',
    'save_evidence_file',
    'cleanup_old_files',
    'get_file_size_mb',
    
    # Validation utilities
    'validate_confidence_score',
    'validate_timestamp',
    'validate_detection_type',
    'sanitize_filename'
]
"""
Common utility functions shared across the application.

This module provides basic utilities for timestamps, session management,
logging, and JSON handling that are used by both the web app and CV backend.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
import os


class ComponentStatus(Enum):
    """Status of service components."""
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class ServiceEvent:
    """Internal event for communication between service components."""
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime
    source: str


@dataclass
class ComponentHealth:
    """Health status of a service component."""
    name: str
    status: ComponentStatus
    last_check: datetime
    error_count: int = 0
    last_error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_usage: float
    memory_usage: float
    detection_latency: float
    alert_frequency: float
    error_rate: float
    uptime: float
    active_sessions: int


def get_timestamp() -> datetime:
    """
    Get current timestamp as datetime object.
    
    Returns:
        Current datetime with microsecond precision
    """
    return datetime.now()


def get_timestamp_string(dt: Optional[datetime] = None) -> str:
    """
    Get timestamp as ISO format string.
    
    Args:
        dt: Datetime object to format, uses current time if None
        
    Returns:
        ISO format timestamp string
    """
    if dt is None:
        dt = get_timestamp()
    return dt.isoformat()


def generate_session_id() -> str:
    """
    Generate a unique session ID.
    
    Returns:
        UUID4 string for session identification
    """
    return str(uuid.uuid4())


def generate_unique_id() -> str:
    """
    Generate a unique identifier.
    
    Returns:
        UUID4 string for general identification purposes
    """
    return str(uuid.uuid4())


def setup_logging(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration for a component.
    
    Args:
        name: Logger name
        level: Logging level (default: INFO)
        log_file: Optional log file path
        format_string: Optional custom format string
        
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def safe_json_loads(json_string: str, default: Any = None) -> Any:
    """
    Safely parse JSON string with fallback.
    
    Args:
        json_string: JSON string to parse
        default: Default value to return on parse error
        
    Returns:
        Parsed JSON object or default value
    """
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError, ValueError):
        return default


def safe_json_dumps(obj: Any, default: Any = None, **kwargs) -> str:
    """
    Safely serialize object to JSON string.
    
    Args:
        obj: Object to serialize
        default: Default serializer function for non-serializable objects
        **kwargs: Additional arguments for json.dumps
        
    Returns:
        JSON string representation
    """
    try:
        return json.dumps(obj, default=default, **kwargs)
    except (TypeError, ValueError) as e:
        # Return error information as JSON
        return json.dumps({
            'error': 'Serialization failed',
            'message': str(e),
            'type': type(obj).__name__
        })


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f} {size_names[i]}"


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp a value between minimum and maximum bounds.
    
    Args:
        value: Value to clamp
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Clamped value
    """
    return max(min_value, min(value, max_value))


def get_environment_variable(
    name: str,
    default: Optional[str] = None,
    required: bool = False
) -> Optional[str]:
    """
    Get environment variable with optional default and validation.
    
    Args:
        name: Environment variable name
        default: Default value if not found
        required: Whether the variable is required
        
    Returns:
        Environment variable value or default
        
    Raises:
        ValueError: If required variable is not found
    """
    value = os.environ.get(name, default)
    
    if required and value is None:
        raise ValueError(f"Required environment variable '{name}' not found")
    
    return value


def merge_dictionaries(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries with later ones taking precedence.
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result

def create_directories(directories: list[str]) -> None:
    """
    Create multiple directories if they don't exist.
    
    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to create directory {directory}: {e}")


def setup_logging() -> logging.Logger:
    """
    Set up default logging configuration for the application.
    
    Returns:
        Configured logger instance
    """
    return setup_logging(
        name="unified_proctoring_service",
        level=logging.INFO,
        log_file="logs/proctoring_service.log"
    )
"""
Data models and enums for the proctoring service.

This module contains only data classes and enums without external dependencies
to avoid import issues during testing and initialization.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
from enum import Enum


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
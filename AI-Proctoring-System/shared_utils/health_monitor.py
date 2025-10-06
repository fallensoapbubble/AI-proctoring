"""
Health monitoring and error recovery utilities for the unified proctoring service.

This module provides comprehensive health monitoring, performance tracking,
and automatic error recovery mechanisms for service components.
"""

import threading
import time
import logging
import psutil
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from .data_models import ComponentStatus, ComponentHealth, SystemMetrics, ServiceEvent


class HealthMonitor:
    """
    Monitors the health of service components and system performance.
    
    Provides health checks, error recovery, and performance monitoring.
    """
    
    def __init__(self, service_instance):
        self.service = service_instance
        self.logger = logging.getLogger(f"{__name__}.HealthMonitor")
        self.components: Dict[str, ComponentHealth] = {}
        self.metrics_history: List[SystemMetrics] = []
        self.start_time = datetime.now()
        self._monitoring = False
        self._monitor_thread = None
        
        # Performance thresholds
        self.cpu_threshold = 80.0  # %
        self.memory_threshold = 85.0  # %
        self.latency_threshold = 2.0  # seconds
        self.error_rate_threshold = 0.1  # 10%
        
        # Initialize component health tracking
        self._init_component_health()
    
    def _init_component_health(self) -> None:
        """Initialize health tracking for all components."""
        components = [
            'web_interface', 'cv_processing', 'event_bus', 
            'context_analyzer', 'alert_manager', 'gaze_detector',
            'lip_detector', 'speech_analyzer', 'object_detector', 'person_detector'
        ]
        
        for component in components:
            self.components[component] = ComponentHealth(
                name=component,
                status=ComponentStatus.STARTING,
                last_check=datetime.now()
            )
    
    def start_monitoring(self) -> None:
        """Start health monitoring in background thread."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("Health monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                self._check_system_health()
                self._collect_metrics()
                self._check_performance_thresholds()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    def _check_system_health(self) -> None:
        """Check health of all components."""
        try:
            # Check web interface
            self._update_component_health('web_interface', 
                                        ComponentStatus.HEALTHY if self.service._running else ComponentStatus.STOPPED)
            
            # Check CV processing
            self._update_component_health('cv_processing',
                                        ComponentStatus.HEALTHY if self.service._cv_processing else ComponentStatus.STOPPED)
            
            # Check event bus
            event_bus_healthy = (self.service.event_bus._processing and 
                               self.service.event_bus._processor_thread and 
                               self.service.event_bus._processor_thread.is_alive())
            self._update_component_health('event_bus',
                                        ComponentStatus.HEALTHY if event_bus_healthy else ComponentStatus.FAILED)
            
            # Check context analyzer
            try:
                analyzer_healthy = self.service.context_analyzer.is_healthy()
                self._update_component_health('context_analyzer',
                                            ComponentStatus.HEALTHY if analyzer_healthy else ComponentStatus.DEGRADED)
            except Exception as e:
                self._update_component_health('context_analyzer', ComponentStatus.FAILED, str(e))
            
            # Check alert manager
            try:
                alert_manager_healthy = self.service.alert_manager.is_healthy()
                self._update_component_health('alert_manager',
                                            ComponentStatus.HEALTHY if alert_manager_healthy else ComponentStatus.DEGRADED)
            except Exception as e:
                self._update_component_health('alert_manager', ComponentStatus.FAILED, str(e))
            
        except Exception as e:
            self.logger.error(f"Error checking system health: {e}")
    
    def _update_component_health(self, component: str, status: ComponentStatus, error: Optional[str] = None) -> None:
        """Update health status of a component."""
        if component in self.components:
            health = self.components[component]
            
            # Update error count if status changed to failed
            if status == ComponentStatus.FAILED and health.status != ComponentStatus.FAILED:
                health.error_count += 1
                health.last_error = error
            
            health.status = status
            health.last_check = datetime.now()
            
            # Log status changes
            if health.status != status:
                self.logger.info(f"Component {component} status changed to {status.value}")
    
    def _collect_metrics(self) -> None:
        """Collect system performance metrics."""
        try:
            # Get system metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Calculate uptime
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            # Get detection latency (placeholder - would be calculated from actual processing times)
            detection_latency = self._calculate_detection_latency()
            
            # Get alert frequency (alerts per minute)
            alert_frequency = self._calculate_alert_frequency()
            
            # Calculate error rate
            error_rate = self._calculate_error_rate()
            
            # Count active sessions (placeholder)
            active_sessions = self._count_active_sessions()
            
            metrics = SystemMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                detection_latency=detection_latency,
                alert_frequency=alert_frequency,
                error_rate=error_rate,
                uptime=uptime,
                active_sessions=active_sessions
            )
            
            # Store metrics (keep last 100 entries)
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > 100:
                self.metrics_history.pop(0)
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
    
    def _calculate_detection_latency(self) -> float:
        """Calculate average detection processing latency."""
        # Placeholder implementation - would track actual processing times
        return 0.5
    
    def _calculate_alert_frequency(self) -> float:
        """Calculate alert frequency (alerts per minute)."""
        # Placeholder implementation - would track actual alert generation
        return 0.1
    
    def _calculate_error_rate(self) -> float:
        """Calculate system error rate."""
        total_errors = sum(health.error_count for health in self.components.values())
        total_checks = len(self.components) * len(self.metrics_history)
        return total_errors / max(total_checks, 1)
    
    def _count_active_sessions(self) -> int:
        """Count active exam sessions."""
        # Placeholder implementation - would track actual sessions
        return 1 if self.service._cv_processing else 0
    
    def _check_performance_thresholds(self) -> None:
        """Check if performance metrics exceed thresholds."""
        if not self.metrics_history:
            return
        
        latest_metrics = self.metrics_history[-1]
        
        # Check CPU usage
        if latest_metrics.cpu_usage > self.cpu_threshold:
            self._trigger_performance_alert('cpu_high', 
                                          f"CPU usage {latest_metrics.cpu_usage:.1f}% exceeds threshold {self.cpu_threshold}%")
        
        # Check memory usage
        if latest_metrics.memory_usage > self.memory_threshold:
            self._trigger_performance_alert('memory_high',
                                          f"Memory usage {latest_metrics.memory_usage:.1f}% exceeds threshold {self.memory_threshold}%")
        
        # Check detection latency
        if latest_metrics.detection_latency > self.latency_threshold:
            self._trigger_performance_alert('latency_high',
                                          f"Detection latency {latest_metrics.detection_latency:.2f}s exceeds threshold {self.latency_threshold}s")
        
        # Check error rate
        if latest_metrics.error_rate > self.error_rate_threshold:
            self._trigger_performance_alert('error_rate_high',
                                          f"Error rate {latest_metrics.error_rate:.2%} exceeds threshold {self.error_rate_threshold:.2%}")
    
    def _trigger_performance_alert(self, alert_type: str, message: str) -> None:
        """Trigger a performance degradation alert."""
        self.logger.warning(f"Performance alert [{alert_type}]: {message}")
        
        # Publish performance alert event
        self.service.event_bus.publish(ServiceEvent(
            event_type='performance_alert',
            data={
                'alert_type': alert_type,
                'message': message,
                'metrics': self.get_current_metrics()
            },
            timestamp=datetime.now(),
            source='health_monitor'
        ))
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of all components."""
        return {
            'overall_status': self._get_overall_status(),
            'components': {
                name: {
                    'status': health.status.value,
                    'last_check': health.last_check.isoformat(),
                    'error_count': health.error_count,
                    'last_error': health.last_error
                }
                for name, health in self.components.items()
            },
            'uptime': (datetime.now() - self.start_time).total_seconds()
        }
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        if not self.metrics_history:
            return {}
        
        latest = self.metrics_history[-1]
        return {
            'cpu_usage': latest.cpu_usage,
            'memory_usage': latest.memory_usage,
            'detection_latency': latest.detection_latency,
            'alert_frequency': latest.alert_frequency,
            'error_rate': latest.error_rate,
            'uptime': latest.uptime,
            'active_sessions': latest.active_sessions
        }
    
    def _get_overall_status(self) -> str:
        """Determine overall system health status."""
        statuses = [health.status for health in self.components.values()]
        
        if any(status == ComponentStatus.FAILED for status in statuses):
            return 'critical'
        elif any(status == ComponentStatus.DEGRADED for status in statuses):
            return 'degraded'
        elif all(status in [ComponentStatus.HEALTHY, ComponentStatus.STOPPED] for status in statuses):
            return 'healthy'
        else:
            return 'starting'


class ErrorRecoveryManager:
    """
    Manages error recovery and component restart mechanisms.
    """
    
    def __init__(self, service_instance):
        self.service = service_instance
        self.logger = logging.getLogger(f"{__name__}.ErrorRecovery")
        self.recovery_attempts: Dict[str, int] = {}
        self.max_recovery_attempts = 3
        self.recovery_cooldown = 60  # seconds
        self.last_recovery_time: Dict[str, datetime] = {}
    
    def attempt_recovery(self, component: str, error: Exception) -> bool:
        """
        Attempt to recover a failed component.
        
        Args:
            component: Name of the failed component
            error: The exception that caused the failure
            
        Returns:
            True if recovery was attempted, False if max attempts exceeded
        """
        current_time = datetime.now()
        
        # Check if we're in cooldown period
        if component in self.last_recovery_time:
            time_since_last = (current_time - self.last_recovery_time[component]).total_seconds()
            if time_since_last < self.recovery_cooldown:
                self.logger.info(f"Component {component} in recovery cooldown, skipping attempt")
                return False
        
        # Check recovery attempt count
        attempts = self.recovery_attempts.get(component, 0)
        if attempts >= self.max_recovery_attempts:
            self.logger.error(f"Max recovery attempts ({self.max_recovery_attempts}) exceeded for {component}")
            return False
        
        self.logger.info(f"Attempting recovery for component {component} (attempt {attempts + 1})")
        
        try:
            success = self._recover_component(component, error)
            
            if success:
                self.logger.info(f"Successfully recovered component {component}")
                # Reset attempt count on successful recovery
                self.recovery_attempts[component] = 0
            else:
                self.recovery_attempts[component] = attempts + 1
                self.last_recovery_time[component] = current_time
                self.logger.warning(f"Recovery attempt {attempts + 1} failed for {component}")
            
            return success
            
        except Exception as recovery_error:
            self.logger.error(f"Error during recovery of {component}: {recovery_error}")
            self.recovery_attempts[component] = attempts + 1
            self.last_recovery_time[component] = current_time
            return False
    
    def _recover_component(self, component: str, error: Exception) -> bool:
        """
        Perform actual recovery for a specific component.
        
        Args:
            component: Name of the component to recover
            error: The original error
            
        Returns:
            True if recovery was successful
        """
        try:
            if component == 'cv_processing':
                return self._recover_cv_processing()
            elif component == 'event_bus':
                return self._recover_event_bus()
            elif component == 'context_analyzer':
                return self._recover_context_analyzer()
            elif component == 'alert_manager':
                return self._recover_alert_manager()
            elif component in ['gaze_detector', 'lip_detector', 'speech_analyzer', 'object_detector', 'person_detector']:
                return self._recover_detector(component)
            else:
                self.logger.warning(f"No recovery procedure defined for component {component}")
                return False
                
        except Exception as e:
            self.logger.error(f"Recovery procedure failed for {component}: {e}")
            return False
    
    def _recover_cv_processing(self) -> bool:
        """Recover CV processing components."""
        try:
            # Stop current CV processing
            self.service.stop_cv_processing()
            time.sleep(2)
            
            # Restart CV processing
            self.service.start_cv_processing()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to recover CV processing: {e}")
            return False
    
    def _recover_event_bus(self) -> bool:
        """Recover event bus."""
        try:
            # Stop current event bus
            self.service.event_bus.stop_processing()
            time.sleep(1)
            
            # Restart event bus
            self.service.event_bus.start_processing()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to recover event bus: {e}")
            return False
    
    def _recover_context_analyzer(self) -> bool:
        """Recover context analyzer."""
        try:
            # Reinitialize context analyzer
            from context_engine.analyzer import ContextCueAnalyzer
            self.service.context_analyzer = ContextCueAnalyzer(self.service.config_service)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to recover context analyzer: {e}")
            return False
    
    def _recover_alert_manager(self) -> bool:
        """Recover alert manager."""
        try:
            # Reinitialize alert manager
            from context_engine.alerts import AlertManager
            self.service.alert_manager = AlertManager(self.service.config_service)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to recover alert manager: {e}")
            return False
    
    def _recover_detector(self, detector_name: str) -> bool:
        """Recover a specific detector component."""
        try:
            from shared_utils.detection_utils import (
                GazeDetector, LipMovementDetector, SpeechAnalyzer, 
                ObjectDetector, PersonDetector
            )
            
            if detector_name == 'gaze_detector':
                self.service.gaze_detector = GazeDetector()
            elif detector_name == 'lip_detector':
                self.service.lip_detector = LipMovementDetector()
            elif detector_name == 'speech_analyzer':
                self.service.speech_analyzer = SpeechAnalyzer()
            elif detector_name == 'object_detector':
                self.service.object_detector = ObjectDetector()
            elif detector_name == 'person_detector':
                self.service.person_detector = PersonDetector()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to recover {detector_name}: {e}")
            return False
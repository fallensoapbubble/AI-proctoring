"""
Alert Manager - Alert generation, evidence collection, and notifications.

This module contains the AlertManager class that handles alert generation,
evidence packaging, notification systems, and alert deduplication logic.
"""

import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import uuid

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np

from .interfaces import AlertHandler, EvidenceCollector
from .models import (
    Alert, AlertSeverity, AnalysisResult, DetectionEvent, DetectionType,
    SystemConfiguration
)
from shared_utils.file_utils import (
    ensure_directory_exists, save_evidence_file, cleanup_old_files,
    get_file_size_mb, sanitize_filename
)
from shared_utils.validation import validate_alert_data


class EvidencePackager:
    """Handles evidence collection and packaging for alerts."""
    
    def __init__(self, evidence_dir: str = "evidence", max_file_size_mb: int = 100):
        """
        Initialize evidence packager.
        
        Args:
            evidence_dir: Base directory for evidence storage
            max_file_size_mb: Maximum file size in MB
        """
        self.evidence_dir = evidence_dir
        self.max_file_size_mb = max_file_size_mb
        self.logger = logging.getLogger(__name__)
        
        # Ensure evidence directory exists
        ensure_directory_exists(self.evidence_dir)
    
    def package_evidence(
        self,
        detection_event: DetectionEvent,
        related_events: List[DetectionEvent] = None,
        audio_data: Optional[bytes] = None
    ) -> List[str]:
        """
        Package evidence for a detection event and related events.
        
        Args:
            detection_event: Primary detection event
            related_events: List of related detection events
            audio_data: Optional audio data to include
            
        Returns:
            List of evidence file paths
        """
        evidence_files = []
        timestamp_str = detection_event.timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        try:
            # Package screenshot from primary event
            if detection_event.frame_data is not None:
                screenshot_path = self._save_screenshot(
                    detection_event.frame_data,
                    f"{timestamp_str}_{detection_event.event_type.value}_primary.jpg"
                )
                if screenshot_path:
                    evidence_files.append(screenshot_path)
            
            # Package screenshots from related events
            if related_events:
                for i, event in enumerate(related_events):
                    if event.frame_data is not None:
                        screenshot_path = self._save_screenshot(
                            event.frame_data,
                            f"{timestamp_str}_{event.event_type.value}_related_{i}.jpg"
                        )
                        if screenshot_path:
                            evidence_files.append(screenshot_path)
            
            # Package audio data if provided
            if audio_data:
                audio_path = self._save_audio_data(
                    audio_data,
                    f"{timestamp_str}_audio.wav"
                )
                if audio_path:
                    evidence_files.append(audio_path)
            
            # Package metadata
            metadata_path = self._save_metadata(
                detection_event,
                related_events or [],
                f"{timestamp_str}_metadata.json"
            )
            if metadata_path:
                evidence_files.append(metadata_path)
                
        except Exception as e:
            self.logger.error(f"Error packaging evidence: {e}")
        
        return evidence_files
    
    def _save_screenshot(self, frame_data: np.ndarray, filename: str) -> Optional[str]:
        """Save screenshot frame data to file."""
        try:
            if cv2 is None:
                self.logger.warning("OpenCV not available, cannot save screenshot")
                return None
                
            # Convert frame data to image format if needed
            if len(frame_data.shape) == 3:
                # Convert BGR to RGB for saving
                frame_rgb = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame_data
            
            # Encode as JPEG
            success, encoded_img = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                return None
            
            # Check file size
            img_bytes = encoded_img.tobytes()
            if len(img_bytes) > self.max_file_size_mb * 1024 * 1024:
                # Reduce quality if too large
                success, encoded_img = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 50])
                if not success:
                    return None
                img_bytes = encoded_img.tobytes()
            
            return save_evidence_file(img_bytes, filename, self.evidence_dir)
            
        except Exception as e:
            self.logger.error(f"Error saving screenshot {filename}: {e}")
            return None
    
    def _save_audio_data(self, audio_data: bytes, filename: str) -> Optional[str]:
        """Save audio data to file."""
        try:
            # Check file size
            if len(audio_data) > self.max_file_size_mb * 1024 * 1024:
                self.logger.warning(f"Audio data too large ({len(audio_data)} bytes), skipping")
                return None
            
            return save_evidence_file(audio_data, filename, self.evidence_dir)
            
        except Exception as e:
            self.logger.error(f"Error saving audio data {filename}: {e}")
            return None
    
    def _save_metadata(
        self,
        primary_event: DetectionEvent,
        related_events: List[DetectionEvent],
        filename: str
    ) -> Optional[str]:
        """Save event metadata to JSON file."""
        try:
            metadata = {
                'primary_event': primary_event.to_dict(),
                'related_events': [event.to_dict() for event in related_events],
                'evidence_timestamp': datetime.now().isoformat(),
                'total_events': 1 + len(related_events)
            }
            
            metadata_json = json.dumps(metadata, indent=2).encode('utf-8')
            return save_evidence_file(metadata_json, filename, self.evidence_dir)
            
        except Exception as e:
            self.logger.error(f"Error saving metadata {filename}: {e}")
            return None


class AlertDeduplicator:
    """Handles alert deduplication and suppression logic."""
    
    def __init__(self, suppression_window_seconds: int = 30):
        """
        Initialize alert deduplicator.
        
        Args:
            suppression_window_seconds: Time window for alert suppression
        """
        self.suppression_window_seconds = suppression_window_seconds
        self.recent_alerts: Dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def should_suppress_alert(self, alert: Alert) -> bool:
        """
        Check if alert should be suppressed due to recent similar alerts.
        
        Args:
            alert: Alert to check
            
        Returns:
            True if alert should be suppressed, False otherwise
        """
        with self.lock:
            # Create suppression key based on detection types and session
            suppression_key = self._create_suppression_key(alert)
            
            # Clean old alerts from the deque
            self._clean_old_alerts(suppression_key)
            
            # Check if we have recent similar alerts
            recent_alerts = self.recent_alerts[suppression_key]
            
            if len(recent_alerts) > 0:
                # Check if we should suppress based on frequency
                if self._should_suppress_by_frequency(recent_alerts, alert):
                    self.logger.info(f"Suppressing alert {alert.alert_id} due to frequency")
                    return True
            
            # Add this alert to recent alerts
            recent_alerts.append({
                'timestamp': alert.timestamp,
                'severity': alert.severity,
                'alert_id': alert.alert_id
            })
            
            return False
    
    def _create_suppression_key(self, alert: Alert) -> str:
        """Create a key for alert suppression grouping."""
        # Group by primary detection type and session
        key_parts = [
            alert.primary_detection.value,
            alert.session_id or "no_session"
        ]
        
        # Add contributing detections for more specific grouping
        if alert.contributing_detections:
            contributing_sorted = sorted([det.value for det in alert.contributing_detections])
            key_parts.extend(contributing_sorted)
        
        return "|".join(key_parts)
    
    def _clean_old_alerts(self, suppression_key: str) -> None:
        """Remove alerts older than suppression window."""
        cutoff_time = datetime.now() - timedelta(seconds=self.suppression_window_seconds)
        recent_alerts = self.recent_alerts[suppression_key]
        
        while recent_alerts and recent_alerts[0]['timestamp'] < cutoff_time:
            recent_alerts.popleft()
    
    def _should_suppress_by_frequency(self, recent_alerts: deque, new_alert: Alert) -> bool:
        """Check if alert should be suppressed based on frequency."""
        # Suppress if we have more than 3 similar alerts in the window
        if len(recent_alerts) >= 3:
            return True
        
        # Suppress if we have a high severity alert very recently (within 10 seconds)
        if new_alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            recent_cutoff = datetime.now() - timedelta(seconds=10)
            for alert_info in recent_alerts:
                if (alert_info['timestamp'] > recent_cutoff and 
                    alert_info['severity'] in ['high', 'critical']):
                    return True
        
        return False
    
    def get_suppression_stats(self) -> Dict[str, Any]:
        """Get statistics about alert suppression."""
        with self.lock:
            stats = {
                'active_suppression_keys': len(self.recent_alerts),
                'total_recent_alerts': sum(len(alerts) for alerts in self.recent_alerts.values()),
                'suppression_window_seconds': self.suppression_window_seconds
            }
            
            # Add breakdown by detection type
            detection_counts = defaultdict(int)
            for key, alerts in self.recent_alerts.items():
                primary_detection = key.split('|')[0]
                detection_counts[primary_detection] += len(alerts)
            
            stats['alerts_by_detection_type'] = dict(detection_counts)
            return stats


class NotificationSystem:
    """Handles real-time notifications for administrators."""
    
    def __init__(self):
        """Initialize notification system."""
        self.subscribers: Dict[str, List[callable]] = {
            'email': [],
            'webhook': [],
            'websocket': [],
            'log': []
        }
        self.notification_queue = deque()
        self.queue_lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Start notification worker thread
        self.worker_thread = threading.Thread(target=self._notification_worker, daemon=True)
        self.worker_running = True
        self.worker_thread.start()
    
    def subscribe(self, notification_type: str, callback: callable) -> bool:
        """
        Subscribe to notifications of a specific type.
        
        Args:
            notification_type: Type of notification ('email', 'webhook', 'websocket', 'log')
            callback: Function to call when notification is sent
            
        Returns:
            True if subscription successful, False otherwise
        """
        if notification_type not in self.subscribers:
            return False
        
        self.subscribers[notification_type].append(callback)
        self.logger.info(f"Added subscriber for {notification_type} notifications")
        return True
    
    def unsubscribe(self, notification_type: str, callback: callable) -> bool:
        """
        Unsubscribe from notifications.
        
        Args:
            notification_type: Type of notification
            callback: Callback function to remove
            
        Returns:
            True if unsubscription successful, False otherwise
        """
        if notification_type not in self.subscribers:
            return False
        
        if callback in self.subscribers[notification_type]:
            self.subscribers[notification_type].remove(callback)
            self.logger.info(f"Removed subscriber for {notification_type} notifications")
            return True
        
        return False
    
    def send_notification(self, alert: Alert, notification_types: List[str] = None) -> None:
        """
        Send notification for an alert.
        
        Args:
            alert: Alert to notify about
            notification_types: Types of notifications to send (default: all)
        """
        if notification_types is None:
            notification_types = list(self.subscribers.keys())
        
        notification = {
            'alert': alert,
            'timestamp': datetime.now(),
            'types': notification_types
        }
        
        with self.queue_lock:
            self.notification_queue.append(notification)
    
    def _notification_worker(self) -> None:
        """Worker thread that processes notification queue."""
        while self.worker_running:
            try:
                with self.queue_lock:
                    if self.notification_queue:
                        notification = self.notification_queue.popleft()
                    else:
                        notification = None
                
                if notification:
                    self._process_notification(notification)
                else:
                    time.sleep(0.1)  # Brief sleep when queue is empty
                    
            except Exception as e:
                self.logger.error(f"Error in notification worker: {e}")
                time.sleep(1)  # Longer sleep on error
    
    def _process_notification(self, notification: Dict) -> None:
        """Process a single notification."""
        alert = notification['alert']
        types = notification['types']
        
        for notification_type in types:
            if notification_type in self.subscribers:
                for callback in self.subscribers[notification_type]:
                    try:
                        callback(alert, notification_type)
                    except Exception as e:
                        self.logger.error(f"Error in {notification_type} notification callback: {e}")
    
    def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification system statistics."""
        with self.queue_lock:
            queue_size = len(self.notification_queue)
        
        subscriber_counts = {
            notification_type: len(callbacks)
            for notification_type, callbacks in self.subscribers.items()
        }
        
        return {
            'queue_size': queue_size,
            'subscriber_counts': subscriber_counts,
            'worker_running': self.worker_running
        }
    
    def shutdown(self) -> None:
        """Shutdown the notification system."""
        self.worker_running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)


class AlertHistoryTracker:
    """Tracks alert history and performs pattern analysis."""
    
    def __init__(self, max_history_size: int = 10000):
        """
        Initialize alert history tracker.
        
        Args:
            max_history_size: Maximum number of alerts to keep in memory
        """
        self.max_history_size = max_history_size
        self.alert_history: List[Alert] = []
        self.history_lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def add_alert(self, alert: Alert) -> None:
        """Add alert to history tracking."""
        with self.history_lock:
            self.alert_history.append(alert)
            
            # Maintain maximum history size
            if len(self.alert_history) > self.max_history_size:
                self.alert_history = self.alert_history[-self.max_history_size:]
    
    def get_alert_patterns(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Analyze alert patterns within a time window.
        
        Args:
            time_window_hours: Time window for pattern analysis
            
        Returns:
            Dictionary containing pattern analysis results
        """
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        with self.history_lock:
            recent_alerts = [
                alert for alert in self.alert_history
                if alert.timestamp > cutoff_time
            ]
        
        if not recent_alerts:
            return {'total_alerts': 0, 'patterns': {}}
        
        patterns = {
            'total_alerts': len(recent_alerts),
            'severity_distribution': self._analyze_severity_distribution(recent_alerts),
            'detection_type_frequency': self._analyze_detection_frequency(recent_alerts),
            'temporal_patterns': self._analyze_temporal_patterns(recent_alerts),
            'session_patterns': self._analyze_session_patterns(recent_alerts),
            'correlation_patterns': self._analyze_correlation_patterns(recent_alerts)
        }
        
        return patterns
    
    def _analyze_severity_distribution(self, alerts: List[Alert]) -> Dict[str, int]:
        """Analyze distribution of alert severities."""
        severity_counts = defaultdict(int)
        for alert in alerts:
            severity_counts[alert.severity.value] += 1
        return dict(severity_counts)
    
    def _analyze_detection_frequency(self, alerts: List[Alert]) -> Dict[str, int]:
        """Analyze frequency of different detection types."""
        detection_counts = defaultdict(int)
        for alert in alerts:
            detection_counts[alert.primary_detection.value] += 1
            for contributing in alert.contributing_detections:
                detection_counts[f"{contributing.value}_contributing"] += 1
        return dict(detection_counts)
    
    def _analyze_temporal_patterns(self, alerts: List[Alert]) -> Dict[str, Any]:
        """Analyze temporal patterns in alerts."""
        if not alerts:
            return {}
        
        # Group alerts by hour of day
        hourly_counts = defaultdict(int)
        for alert in alerts:
            hour = alert.timestamp.hour
            hourly_counts[hour] += 1
        
        # Find peak hours
        if hourly_counts:
            peak_hour = max(hourly_counts.items(), key=lambda x: x[1])
            avg_per_hour = sum(hourly_counts.values()) / len(hourly_counts)
        else:
            peak_hour = (0, 0)
            avg_per_hour = 0
        
        return {
            'hourly_distribution': dict(hourly_counts),
            'peak_hour': peak_hour[0],
            'peak_hour_count': peak_hour[1],
            'average_per_hour': avg_per_hour
        }
    
    def _analyze_session_patterns(self, alerts: List[Alert]) -> Dict[str, Any]:
        """Analyze patterns by session."""
        session_counts = defaultdict(int)
        session_severities = defaultdict(list)
        
        for alert in alerts:
            if alert.session_id:
                session_counts[alert.session_id] += 1
                session_severities[alert.session_id].append(alert.severity.value)
        
        # Find sessions with most alerts
        if session_counts:
            top_session = max(session_counts.items(), key=lambda x: x[1])
            avg_per_session = sum(session_counts.values()) / len(session_counts)
        else:
            top_session = ("", 0)
            avg_per_session = 0
        
        return {
            'unique_sessions': len(session_counts),
            'top_session_id': top_session[0],
            'top_session_alerts': top_session[1],
            'average_alerts_per_session': avg_per_session,
            'sessions_with_high_severity': len([
                session_id for session_id, severities in session_severities.items()
                if 'high' in severities or 'critical' in severities
            ])
        }
    
    def _analyze_correlation_patterns(self, alerts: List[Alert]) -> Dict[str, Any]:
        """Analyze correlation patterns between detection types."""
        correlation_pairs = defaultdict(int)
        multi_modal_count = 0
        
        for alert in alerts:
            if alert.contributing_detections:
                multi_modal_count += 1
                primary = alert.primary_detection.value
                
                for contributing in alert.contributing_detections:
                    pair = tuple(sorted([primary, contributing.value]))
                    correlation_pairs[pair] += 1
        
        # Find most common correlation pairs
        if correlation_pairs:
            top_correlation = max(correlation_pairs.items(), key=lambda x: x[1])
        else:
            top_correlation = ((), 0)
        
        return {
            'multi_modal_alerts': multi_modal_count,
            'single_modal_alerts': len(alerts) - multi_modal_count,
            'correlation_pairs': dict(correlation_pairs),
            'most_common_correlation': {
                'pair': list(top_correlation[0]) if top_correlation[0] else [],
                'count': top_correlation[1]
            }
        }
    
    def get_alert_history(self, limit: Optional[int] = None) -> List[Alert]:
        """Get alert history with optional limit."""
        with self.history_lock:
            if limit:
                return self.alert_history[-limit:]
            return self.alert_history.copy()
    
    def clear_history(self) -> int:
        """Clear alert history and return number of alerts cleared."""
        with self.history_lock:
            count = len(self.alert_history)
            self.alert_history.clear()
            return count


class AlertManager(AlertHandler, EvidenceCollector):
    """
    Manages alert generation, evidence collection, and notifications.
    
    This class handles the complete alert lifecycle from generation through
    notification and storage, including evidence packaging and deduplication.
    """
    
    def __init__(
        self,
        config: SystemConfiguration,
        evidence_dir: str = "evidence",
        alert_log_file: str = "alerts.log"
    ):
        """
        Initialize AlertManager.
        
        Args:
            config: System configuration
            evidence_dir: Directory for evidence storage
            alert_log_file: File for structured alert logging
        """
        self.config = config
        self.evidence_dir = evidence_dir
        self.alert_log_file = alert_log_file
        
        # Initialize components
        self.evidence_packager = EvidencePackager(
            evidence_dir, 
            config.max_evidence_file_size_mb
        )
        self.deduplicator = AlertDeduplicator()
        self.notification_system = NotificationSystem()
        self.history_tracker = AlertHistoryTracker()
        
        # Notification callbacks (legacy support)
        self.notification_callbacks: List[callable] = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self._setup_alert_logging()
        
        # Ensure evidence directory exists
        ensure_directory_exists(self.evidence_dir)
        
        # Setup default notification handlers
        self._setup_default_notifications()
    
    def generate_alert(
        self,
        analysis_result: AnalysisResult,
        session_id: Optional[str] = None,
        student_id: Optional[str] = None,
        audio_data: Optional[bytes] = None
    ) -> Optional[Alert]:
        """
        Generate an alert from analysis result.
        
        Args:
            analysis_result: Result of contextual analysis
            session_id: Optional session identifier
            student_id: Optional student identifier
            audio_data: Optional audio evidence
            
        Returns:
            Generated Alert object, or None if alert was suppressed
        """
        try:
            # Determine alert severity based on analysis
            severity = self._determine_alert_severity(analysis_result)
            
            # Create alert object
            alert = Alert(
                timestamp=datetime.now(),
                severity=severity,
                primary_detection=analysis_result.primary_event.event_type,
                contributing_detections=[
                    event.event_type for event in analysis_result.correlated_events
                ],
                confidence_breakdown=self._create_confidence_breakdown(analysis_result),
                contextual_reasoning=analysis_result.reasoning,
                session_id=session_id,
                student_id=student_id
            )
            
            # Check if alert should be suppressed
            if self.deduplicator.should_suppress_alert(alert):
                self.logger.info(f"Alert suppressed: {alert.alert_id}")
                return None
            
            # Package evidence
            evidence_files = self.evidence_packager.package_evidence(
                analysis_result.primary_event,
                analysis_result.correlated_events,
                audio_data
            )
            alert.evidence_files = evidence_files
            
            # Validate alert data
            is_valid, errors = alert.validate_alert()
            if not is_valid:
                self.logger.error(f"Invalid alert data: {errors}")
                return None
            
            # Store alert in history tracker
            self.history_tracker.add_alert(alert)
            
            # Log structured alert information
            self._log_alert(alert)
            
            # Send notifications through new system
            self.notification_system.send_notification(alert)
            
            # Send legacy notifications
            self._send_legacy_notifications(alert)
            
            self.logger.info(f"Alert generated: {alert.alert_id} (severity: {severity.value})")
            return alert
            
        except Exception as e:
            self.logger.error(f"Error generating alert: {e}")
            return None
    
    def _determine_alert_severity(self, analysis_result: AnalysisResult) -> AlertSeverity:
        """Determine alert severity based on analysis result."""
        confidence = analysis_result.confidence_score
        num_correlated = len(analysis_result.correlated_events)
        primary_type = analysis_result.primary_event.event_type
        
        # Critical severity for high-confidence multi-modal detections
        if confidence >= 0.9 and num_correlated >= 2:
            return AlertSeverity.CRITICAL
        
        # High severity for serious single detections or moderate multi-modal
        if (confidence >= 0.8 and primary_type in [
            DetectionType.MULTIPLE_PEOPLE, 
            DetectionType.FACE_SPOOF,
            DetectionType.MOBILE_DETECTED
        ]) or (confidence >= 0.7 and num_correlated >= 1):
            return AlertSeverity.HIGH
        
        # Medium severity for moderate confidence detections
        if confidence >= 0.6:
            return AlertSeverity.MEDIUM
        
        # Low severity for everything else
        return AlertSeverity.LOW
    
    def _create_confidence_breakdown(self, analysis_result: AnalysisResult) -> Dict[str, float]:
        """Create detailed confidence breakdown for alert."""
        breakdown = {
            'overall_confidence': analysis_result.confidence_score,
            'primary_event_confidence': analysis_result.primary_event.confidence
        }
        
        # Add correlated event confidences
        for i, event in enumerate(analysis_result.correlated_events):
            breakdown[f'correlated_event_{i}_confidence'] = event.confidence
        
        # Add contextual factors
        breakdown.update(analysis_result.contextual_factors)
        
        return breakdown
    
    def _setup_default_notifications(self) -> None:
        """Setup default notification handlers."""
        # Log notification handler
        def log_notification_handler(alert: Alert, notification_type: str) -> None:
            self.logger.info(
                f"ALERT NOTIFICATION [{notification_type.upper()}]: "
                f"ID={alert.alert_id}, Severity={alert.severity.value}, "
                f"Type={alert.primary_detection.value}, Session={alert.session_id}"
            )
        
        self.notification_system.subscribe('log', log_notification_handler)
        
        # Console notification handler for high severity alerts
        def console_notification_handler(alert: Alert, notification_type: str) -> None:
            if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                print(f"\n🚨 HIGH PRIORITY ALERT 🚨")
                print(f"Alert ID: {alert.alert_id}")
                print(f"Severity: {alert.severity.value.upper()}")
                print(f"Detection: {alert.primary_detection.value}")
                print(f"Session: {alert.session_id}")
                print(f"Reasoning: {alert.contextual_reasoning}")
                print(f"Evidence Files: {len(alert.evidence_files)}")
                print("=" * 50)
        
        self.notification_system.subscribe('log', console_notification_handler)
    
    def _log_alert(self, alert: Alert) -> None:
        """Log structured alert information."""
        try:
            alert_log_entry = {
                'timestamp': alert.timestamp.isoformat(),
                'alert_id': alert.alert_id,
                'severity': alert.severity.value,
                'primary_detection': alert.primary_detection.value,
                'contributing_detections': [det.value for det in alert.contributing_detections],
                'confidence_breakdown': alert.confidence_breakdown,
                'evidence_file_count': len(alert.evidence_files),
                'session_id': alert.session_id,
                'student_id': alert.student_id,
                'contextual_reasoning': alert.contextual_reasoning
            }
            
            # Write to structured log file
            log_line = json.dumps(alert_log_entry) + '\n'
            with open(self.alert_log_file, 'a', encoding='utf-8') as f:
                f.write(log_line)
                
        except Exception as e:
            self.logger.error(f"Error logging alert: {e}")
    
    def _send_legacy_notifications(self, alert: Alert) -> None:
        """Send notifications using legacy callback system."""
        for callback in self.notification_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in legacy notification callback: {e}")
    
    def _setup_alert_logging(self) -> None:
        """Setup structured logging for alerts."""
        # Configure logger for this module
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    # AlertHandler interface implementation
    def handle_alert(self, alert: Alert) -> bool:
        """Handle a pre-generated alert."""
        try:
            # Store and log the alert
            self.history_tracker.add_alert(alert)
            self._log_alert(alert)
            self.notification_system.send_notification(alert)
            self._send_legacy_notifications(alert)
            return True
        except Exception as e:
            self.logger.error(f"Error handling alert: {e}")
            return False
    
    def get_alert_history(self, limit: Optional[int] = None) -> List[Alert]:
        """Get alert history."""
        return self.history_tracker.get_alert_history(limit)
    
    # EvidenceCollector interface implementation
    def collect_evidence(self, event: DetectionEvent) -> List[str]:
        """Collect evidence for a detection event."""
        return self.evidence_packager.package_evidence(event)
    
    def store_evidence(self, evidence_data: bytes, filename: str) -> str:
        """Store evidence data to persistent storage."""
        safe_filename = sanitize_filename(filename)
        return save_evidence_file(evidence_data, safe_filename, self.evidence_dir)
    
    def cleanup_old_evidence(self, retention_days: int) -> int:
        """Clean up evidence older than retention period."""
        files_deleted, _ = cleanup_old_files(
            self.evidence_dir,
            retention_days,
            "*",
            dry_run=False
        )
        self.logger.info(f"Cleaned up {files_deleted} old evidence files")
        return files_deleted
    
    # Additional utility methods
    def add_notification_callback(self, callback: callable) -> None:
        """Add a notification callback function."""
        self.notification_callbacks.append(callback)
    
    def remove_notification_callback(self, callback: callable) -> None:
        """Remove a notification callback function."""
        if callback in self.notification_callbacks:
            self.notification_callbacks.remove(callback)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get comprehensive alert statistics and metrics."""
        alert_history = self.history_tracker.get_alert_history()
        
        if not alert_history:
            return {
                'total_alerts': 0,
                'notification_stats': self.notification_system.get_notification_stats(),
                'suppression_stats': self.deduplicator.get_suppression_stats()
            }
        
        # Basic statistics
        total_alerts = len(alert_history)
        severity_counts = defaultdict(int)
        detection_counts = defaultdict(int)
        
        for alert in alert_history:
            severity_counts[alert.severity.value] += 1
            detection_counts[alert.primary_detection.value] += 1
        
        # Recent alerts (last hour)
        recent_cutoff = datetime.now() - timedelta(hours=1)
        recent_alerts = [
            alert for alert in alert_history 
            if alert.timestamp > recent_cutoff
        ]
        
        return {
            'total_alerts': total_alerts,
            'recent_alerts_1h': len(recent_alerts),
            'severity_breakdown': dict(severity_counts),
            'detection_type_breakdown': dict(detection_counts),
            'suppression_stats': self.deduplicator.get_suppression_stats(),
            'notification_stats': self.notification_system.get_notification_stats(),
            'evidence_directory_size_mb': self._get_evidence_directory_size(),
            'pattern_analysis': self.history_tracker.get_alert_patterns()
        }
    
    def _get_evidence_directory_size(self) -> float:
        """Get total size of evidence directory in MB."""
        try:
            from shared_utils.file_utils import get_directory_size_mb
            return get_directory_size_mb(self.evidence_dir)
        except Exception:
            return 0.0
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Mark an alert as resolved."""
        alert_history = self.history_tracker.get_alert_history()
        for alert in alert_history:
            if alert.alert_id == alert_id:
                alert.resolved = True
                self.logger.info(f"Alert {alert_id} resolved by {resolved_by}")
                return True
        return False
    
    # New notification system methods
    def subscribe_to_notifications(self, notification_type: str, callback: callable) -> bool:
        """
        Subscribe to real-time notifications.
        
        Args:
            notification_type: Type of notification ('email', 'webhook', 'websocket', 'log')
            callback: Function to call when notification is sent
            
        Returns:
            True if subscription successful, False otherwise
        """
        return self.notification_system.subscribe(notification_type, callback)
    
    def unsubscribe_from_notifications(self, notification_type: str, callback: callable) -> bool:
        """
        Unsubscribe from notifications.
        
        Args:
            notification_type: Type of notification
            callback: Callback function to remove
            
        Returns:
            True if unsubscription successful, False otherwise
        """
        return self.notification_system.unsubscribe(notification_type, callback)
    
    def get_alert_patterns(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Get alert pattern analysis for specified time window.
        
        Args:
            time_window_hours: Time window for pattern analysis
            
        Returns:
            Dictionary containing pattern analysis results
        """
        return self.history_tracker.get_alert_patterns(time_window_hours)
    
    def clear_alert_history(self) -> int:
        """
        Clear alert history and return number of alerts cleared.
        
        Returns:
            Number of alerts that were cleared
        """
        return self.history_tracker.clear_history()
    
    def shutdown(self) -> None:
        """Shutdown the AlertManager and all its components."""
        self.logger.info("Shutting down AlertManager")
        self.notification_system.shutdown()
        
        # Cleanup old evidence files
        try:
            self.cleanup_old_evidence(self.config.evidence_retention_days)
        except Exception as e:
            self.logger.error(f"Error during evidence cleanup: {e}")
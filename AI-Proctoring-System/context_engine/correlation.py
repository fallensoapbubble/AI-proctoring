"""
Correlation Manager - Temporal analysis and event correlation.

This module contains the CorrelationManager class that handles
sliding time window analysis, event sequence pattern matching,
and temporal relationship scoring between detection events.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional, Tuple
from collections import deque, defaultdict
import logging
from dataclasses import dataclass
import math

from .models import DetectionEvent, DetectionType, CorrelatedEvent, SystemConfiguration
from .interfaces import EventProcessor


logger = logging.getLogger(__name__)


@dataclass
class EventPattern:
    """Represents a pattern of events that commonly occur together."""
    event_types: List[DetectionType]
    max_time_span_seconds: float
    confidence_multiplier: float
    description: str


@dataclass
class TimeWindow:
    """Represents a sliding time window for event analysis."""
    start_time: datetime
    end_time: datetime
    events: List[DetectionEvent]
    
    def contains_time(self, timestamp: datetime) -> bool:
        """Check if timestamp falls within this window."""
        return self.start_time <= timestamp <= self.end_time
    
    def get_duration_seconds(self) -> float:
        """Get the duration of this window in seconds."""
        return (self.end_time - self.start_time).total_seconds()


@dataclass
class EventCluster:
    """Represents a cluster of related detection events."""
    cluster_id: str
    events: List[DetectionEvent]
    cluster_score: float
    dominant_type: DetectionType
    time_span_seconds: float
    
    def get_event_types(self) -> Set[DetectionType]:
        """Get all unique event types in this cluster."""
        return {event.event_type for event in self.events}
    
    def get_average_confidence(self) -> float:
        """Get average confidence score of events in cluster."""
        if not self.events:
            return 0.0
        return sum(event.confidence for event in self.events) / len(self.events)


@dataclass
class SuppressionRule:
    """Defines conditions for suppressing alerts based on contradictory signals."""
    suppressing_types: List[DetectionType]
    suppressed_types: List[DetectionType]
    time_window_seconds: float
    confidence_threshold: float
    description: str


class CorrelationManager(EventProcessor):
    """
    Manages temporal correlation analysis and event pattern matching.
    
    This class implements sliding time window analysis to identify
    related detection events and calculate correlation scores based
    on temporal proximity and event type relationships.
    """
    
    def __init__(self, config: SystemConfiguration):
        """
        Initialize the CorrelationManager.
        
        Args:
            config: System configuration containing correlation settings
        """
        self.config = config
        self.event_buffer: deque[DetectionEvent] = deque(maxlen=config.max_correlation_events * 2)
        self.correlation_patterns = self._initialize_patterns()
        self.temporal_weights = self._initialize_temporal_weights()
        self.contextual_weights = self._initialize_contextual_weights()
        self.suppression_rules = self._initialize_suppression_rules()
        self.event_clusters: List[EventCluster] = []
        
        logger.info(f"CorrelationManager initialized with {len(self.correlation_patterns)} patterns, "
                   f"{len(self.suppression_rules)} suppression rules")
    
    def _initialize_patterns(self) -> List[EventPattern]:
        """Initialize common cheating behavior patterns."""
        return [
            EventPattern(
                event_types=[DetectionType.GAZE_AWAY, DetectionType.SUSPICIOUS_SPEECH],
                max_time_span_seconds=3.0,
                confidence_multiplier=1.5,
                description="Looking away while speaking - potential communication"
            ),
            EventPattern(
                event_types=[DetectionType.LIP_MOVEMENT, DetectionType.AUDIO_ANOMALY],
                max_time_span_seconds=2.0,
                confidence_multiplier=1.3,
                description="Lip movement with audio anomaly - potential whispering"
            ),
            EventPattern(
                event_types=[DetectionType.MULTIPLE_PEOPLE, DetectionType.MOBILE_DETECTED],
                max_time_span_seconds=5.0,
                confidence_multiplier=2.0,
                description="Multiple people with mobile device - high risk scenario"
            ),
            EventPattern(
                event_types=[DetectionType.GAZE_AWAY, DetectionType.HEAD_POSE_SUSPICIOUS],
                max_time_span_seconds=2.5,
                confidence_multiplier=1.2,
                description="Coordinated head movement - looking at external source"
            ),
            EventPattern(
                event_types=[DetectionType.FACE_SPOOF, DetectionType.MULTIPLE_PEOPLE],
                max_time_span_seconds=4.0,
                confidence_multiplier=2.5,
                description="Face spoofing with multiple people - identity substitution"
            )
        ]
    
    def _initialize_temporal_weights(self) -> Dict[float, float]:
        """Initialize temporal proximity weights for correlation scoring."""
        return {
            0.5: 1.0,   # Events within 0.5 seconds get full weight
            1.0: 0.9,   # Events within 1 second get 90% weight
            2.0: 0.7,   # Events within 2 seconds get 70% weight
            3.0: 0.5,   # Events within 3 seconds get 50% weight
            5.0: 0.3,   # Events within 5 seconds get 30% weight
        }
    
    def _initialize_contextual_weights(self) -> Dict[Tuple[DetectionType, DetectionType], float]:
        """Initialize contextual weights for different detection type combinations."""
        return {
            # High-risk combinations
            (DetectionType.MULTIPLE_PEOPLE, DetectionType.MOBILE_DETECTED): 2.5,
            (DetectionType.FACE_SPOOF, DetectionType.MULTIPLE_PEOPLE): 2.3,
            (DetectionType.GAZE_AWAY, DetectionType.SUSPICIOUS_SPEECH): 1.8,
            (DetectionType.LIP_MOVEMENT, DetectionType.AUDIO_ANOMALY): 1.6,
            
            # Medium-risk combinations
            (DetectionType.GAZE_AWAY, DetectionType.HEAD_POSE_SUSPICIOUS): 1.4,
            (DetectionType.SUSPICIOUS_SPEECH, DetectionType.AUDIO_ANOMALY): 1.3,
            (DetectionType.MOBILE_DETECTED, DetectionType.GAZE_AWAY): 1.2,
            
            # Lower-risk but still relevant combinations
            (DetectionType.LIP_MOVEMENT, DetectionType.HEAD_POSE_SUSPICIOUS): 1.1,
            (DetectionType.GAZE_AWAY, DetectionType.LIP_MOVEMENT): 1.0,
        }
    
    def _initialize_suppression_rules(self) -> List[SuppressionRule]:
        """Initialize rules for suppressing false positives based on contradictory signals."""
        return [
            SuppressionRule(
                suppressing_types=[DetectionType.SUSPICIOUS_SPEECH],
                suppressed_types=[DetectionType.LIP_MOVEMENT],
                time_window_seconds=1.0,
                confidence_threshold=0.8,
                description="High-confidence speech detection suppresses lip movement false positives"
            ),
            SuppressionRule(
                suppressing_types=[DetectionType.FACE_SPOOF],
                suppressed_types=[DetectionType.GAZE_AWAY, DetectionType.HEAD_POSE_SUSPICIOUS],
                time_window_seconds=2.0,
                confidence_threshold=0.9,
                description="Face spoofing detection suppresses gaze-based alerts"
            ),
            SuppressionRule(
                suppressing_types=[DetectionType.MULTIPLE_PEOPLE],
                suppressed_types=[DetectionType.GAZE_AWAY],
                time_window_seconds=3.0,
                confidence_threshold=0.7,
                description="Multiple people detection may explain gaze deviation"
            )
        ]
    
    def process_event(self, event: DetectionEvent) -> Optional[CorrelatedEvent]:
        """
        Process a single detection event and find correlations.
        
        Args:
            event: The detection event to process
            
        Returns:
            CorrelatedEvent if correlations found, None otherwise
        """
        # Add event to buffer
        self.event_buffer.append(event)
        
        # Create time window for analysis
        window = self._create_time_window(event.timestamp)
        
        # Find correlated events within the window
        correlated_events = self._find_correlated_events(event, window)
        
        if correlated_events:
            # Calculate correlation score
            correlation_score = self._calculate_correlation_score(event, correlated_events)
            
            # Calculate time span
            all_events = [event] + correlated_events
            time_span = self._calculate_time_span(all_events)
            
            logger.debug(f"Found correlation for {event.event_type.value} with "
                        f"{len(correlated_events)} related events, score: {correlation_score:.3f}")
            
            return CorrelatedEvent(
                primary_event=event,
                related_events=correlated_events,
                correlation_score=correlation_score,
                time_span_seconds=time_span
            )
        
        return None
    
    def process_batch(self, events: List[DetectionEvent]) -> List[CorrelatedEvent]:
        """
        Process a batch of detection events.
        
        Args:
            events: List of detection events to process
            
        Returns:
            List of correlated events found
        """
        correlated_events = []
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        for event in sorted_events:
            correlation = self.process_event(event)
            if correlation:
                correlated_events.append(correlation)
        
        logger.info(f"Processed batch of {len(events)} events, found {len(correlated_events)} correlations")
        return correlated_events
    
    def _create_time_window(self, center_time: datetime) -> TimeWindow:
        """
        Create a sliding time window centered on the given time.
        
        Args:
            center_time: Center timestamp for the window
            
        Returns:
            TimeWindow object containing relevant events
        """
        window_size = timedelta(seconds=self.config.correlation_window_seconds)
        half_window = window_size / 2
        
        start_time = center_time - half_window
        end_time = center_time + half_window
        
        # Filter events within the time window
        window_events = [
            event for event in self.event_buffer
            if start_time <= event.timestamp <= end_time
        ]
        
        return TimeWindow(
            start_time=start_time,
            end_time=end_time,
            events=window_events
        )
    
    def _find_correlated_events(self, primary_event: DetectionEvent, window: TimeWindow) -> List[DetectionEvent]:
        """
        Find events correlated with the primary event within the time window.
        
        Args:
            primary_event: The primary event to find correlations for
            window: Time window to search within
            
        Returns:
            List of correlated events
        """
        correlated = []
        
        for event in window.events:
            # Skip the primary event itself
            if event.event_id == primary_event.event_id:
                continue
            
            # Check if events match any known patterns
            if self._events_match_pattern(primary_event, event):
                correlated.append(event)
            
            # Check for temporal proximity correlation
            elif self._events_temporally_correlated(primary_event, event):
                correlated.append(event)
        
        return correlated
    
    def _events_match_pattern(self, event1: DetectionEvent, event2: DetectionEvent) -> bool:
        """
        Check if two events match any known correlation pattern.
        
        Args:
            event1: First event
            event2: Second event
            
        Returns:
            True if events match a pattern, False otherwise
        """
        event_types = {event1.event_type, event2.event_type}
        
        for pattern in self.correlation_patterns:
            pattern_types = set(pattern.event_types)
            if event_types.issubset(pattern_types) or pattern_types.issubset(event_types):
                # Check if events occur within pattern time span
                time_diff = abs((event1.timestamp - event2.timestamp).total_seconds())
                if time_diff <= pattern.max_time_span_seconds:
                    return True
        
        return False
    
    def _events_temporally_correlated(self, event1: DetectionEvent, event2: DetectionEvent) -> bool:
        """
        Check if two events are temporally correlated based on proximity.
        
        Args:
            event1: First event
            event2: Second event
            
        Returns:
            True if events are temporally correlated, False otherwise
        """
        time_diff = abs((event1.timestamp - event2.timestamp).total_seconds())
        
        # Events are correlated if they occur within the correlation window
        # and have sufficient confidence scores
        if time_diff <= self.config.correlation_window_seconds:
            min_confidence = min(event1.confidence, event2.confidence)
            # Require higher confidence for temporal-only correlations
            return min_confidence >= 0.6
        
        return False
    
    def _calculate_correlation_score(self, primary_event: DetectionEvent, 
                                   correlated_events: List[DetectionEvent]) -> float:
        """
        Calculate correlation score based on event relationships and timing.
        
        Args:
            primary_event: The primary event
            correlated_events: List of correlated events
            
        Returns:
            Correlation score between 0.0 and 1.0
        """
        if not correlated_events:
            return 0.0
        
        total_score = 0.0
        max_possible_score = 0.0
        
        for event in correlated_events:
            # Base score from event confidence
            base_score = (primary_event.confidence + event.confidence) / 2
            
            # Apply temporal weight
            time_diff = abs((primary_event.timestamp - event.timestamp).total_seconds())
            temporal_weight = self._get_temporal_weight(time_diff)
            
            # Apply pattern multiplier if events match a pattern
            pattern_multiplier = self._get_pattern_multiplier(primary_event, event)
            
            # Calculate weighted score
            weighted_score = base_score * temporal_weight * pattern_multiplier
            total_score += weighted_score
            max_possible_score += pattern_multiplier  # Maximum possible score for this pair
        
        # Normalize score to 0-1 range
        if max_possible_score > 0:
            normalized_score = min(total_score / max_possible_score, 1.0)
        else:
            normalized_score = total_score / len(correlated_events)
        
        return normalized_score
    
    def _get_temporal_weight(self, time_diff_seconds: float) -> float:
        """
        Get temporal weight based on time difference between events.
        
        Args:
            time_diff_seconds: Time difference in seconds
            
        Returns:
            Temporal weight between 0.0 and 1.0
        """
        for threshold, weight in sorted(self.temporal_weights.items()):
            if time_diff_seconds <= threshold:
                return weight
        
        # Return minimum weight for events outside all thresholds
        return 0.1
    
    def _get_pattern_multiplier(self, event1: DetectionEvent, event2: DetectionEvent) -> float:
        """
        Get pattern multiplier if events match a known pattern.
        
        Args:
            event1: First event
            event2: Second event
            
        Returns:
            Pattern multiplier (1.0 if no pattern match)
        """
        event_types = {event1.event_type, event2.event_type}
        
        for pattern in self.correlation_patterns:
            pattern_types = set(pattern.event_types)
            if event_types.issubset(pattern_types) or pattern_types.issubset(event_types):
                time_diff = abs((event1.timestamp - event2.timestamp).total_seconds())
                if time_diff <= pattern.max_time_span_seconds:
                    return pattern.confidence_multiplier
        
        return 1.0
    
    def _calculate_time_span(self, events: List[DetectionEvent]) -> float:
        """
        Calculate the time span covered by a list of events.
        
        Args:
            events: List of detection events
            
        Returns:
            Time span in seconds
        """
        if len(events) < 2:
            return 0.0
        
        timestamps = [event.timestamp for event in events]
        min_time = min(timestamps)
        max_time = max(timestamps)
        
        return (max_time - min_time).total_seconds()
    
    def get_pattern_statistics(self) -> Dict[str, int]:
        """
        Get statistics about pattern matching from recent events.
        
        Returns:
            Dictionary with pattern match counts
        """
        pattern_counts = defaultdict(int)
        
        # Analyze recent events in buffer
        events_list = list(self.event_buffer)
        for i, event1 in enumerate(events_list):
            for j, event2 in enumerate(events_list[i+1:], i+1):
                if self._events_match_pattern(event1, event2):
                    # Find which pattern matched
                    event_types = {event1.event_type, event2.event_type}
                    for pattern in self.correlation_patterns:
                        pattern_types = set(pattern.event_types)
                        if event_types.issubset(pattern_types) or pattern_types.issubset(event_types):
                            pattern_counts[pattern.description] += 1
                            break
        
        return dict(pattern_counts)
    
    def clear_buffer(self) -> None:
        """Clear the event buffer."""
        self.event_buffer.clear()
        logger.info("Event buffer cleared")
    
    def get_buffer_size(self) -> int:
        """Get current buffer size."""
        return len(self.event_buffer)
    
    # Advanced correlation features for task 4.2
    
    def cluster_events(self, events: List[DetectionEvent], max_clusters: int = 5) -> List[EventCluster]:
        """
        Cluster related detection events using temporal and type-based similarity.
        
        Args:
            events: List of events to cluster
            max_clusters: Maximum number of clusters to create
            
        Returns:
            List of event clusters
        """
        if not events:
            return []
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        clusters = []
        
        for event in sorted_events:
            # Try to add event to existing cluster
            added_to_cluster = False
            
            for cluster in clusters:
                if self._should_add_to_cluster(event, cluster):
                    cluster.events.append(event)
                    cluster.cluster_score = self._calculate_cluster_score(cluster.events)
                    cluster.time_span_seconds = self._calculate_time_span(cluster.events)
                    added_to_cluster = True
                    break
            
            # Create new cluster if event doesn't fit existing ones
            if not added_to_cluster and len(clusters) < max_clusters:
                cluster_id = f"cluster_{len(clusters) + 1}_{event.timestamp.strftime('%H%M%S')}"
                new_cluster = EventCluster(
                    cluster_id=cluster_id,
                    events=[event],
                    cluster_score=event.confidence,
                    dominant_type=event.event_type,
                    time_span_seconds=0.0
                )
                clusters.append(new_cluster)
        
        # Update dominant types for multi-event clusters
        for cluster in clusters:
            if len(cluster.events) > 1:
                cluster.dominant_type = self._find_dominant_event_type(cluster.events)
        
        logger.debug(f"Created {len(clusters)} event clusters from {len(events)} events")
        return clusters
    
    def _should_add_to_cluster(self, event: DetectionEvent, cluster: EventCluster) -> bool:
        """
        Determine if an event should be added to an existing cluster.
        
        Args:
            event: Event to potentially add
            cluster: Existing cluster
            
        Returns:
            True if event should be added to cluster
        """
        if not cluster.events:
            return True
        
        # Check temporal proximity to cluster events
        for cluster_event in cluster.events:
            time_diff = abs((event.timestamp - cluster_event.timestamp).total_seconds())
            if time_diff <= self.config.correlation_window_seconds:
                # Check if event types are compatible
                if self._are_event_types_compatible(event.event_type, cluster_event.event_type):
                    return True
        
        return False
    
    def _are_event_types_compatible(self, type1: DetectionType, type2: DetectionType) -> bool:
        """
        Check if two event types are compatible for clustering.
        
        Args:
            type1: First event type
            type2: Second event type
            
        Returns:
            True if types are compatible
        """
        # Same types are always compatible
        if type1 == type2:
            return True
        
        # Check if types have contextual weight (indicating relationship)
        type_pair = (type1, type2)
        reverse_pair = (type2, type1)
        
        return (type_pair in self.contextual_weights or 
                reverse_pair in self.contextual_weights)
    
    def _calculate_cluster_score(self, events: List[DetectionEvent]) -> float:
        """
        Calculate overall score for an event cluster.
        
        Args:
            events: Events in the cluster
            
        Returns:
            Cluster score between 0.0 and 1.0
        """
        if not events:
            return 0.0
        
        # Base score from average confidence
        avg_confidence = sum(event.confidence for event in events) / len(events)
        
        # Bonus for multiple related event types
        unique_types = len(set(event.event_type for event in events))
        type_diversity_bonus = min(unique_types * 0.1, 0.3)
        
        # Bonus for contextual relationships
        contextual_bonus = 0.0
        for i, event1 in enumerate(events):
            for event2 in events[i+1:]:
                contextual_bonus += self._get_contextual_weight(event1.event_type, event2.event_type)
        
        contextual_bonus = min(contextual_bonus / len(events), 0.4)
        
        total_score = avg_confidence + type_diversity_bonus + contextual_bonus
        return min(total_score, 1.0)
    
    def _find_dominant_event_type(self, events: List[DetectionEvent]) -> DetectionType:
        """
        Find the dominant event type in a cluster based on frequency and confidence.
        
        Args:
            events: Events in the cluster
            
        Returns:
            Dominant event type
        """
        type_scores = defaultdict(float)
        
        for event in events:
            # Score based on confidence and frequency
            type_scores[event.event_type] += event.confidence
        
        # Return type with highest total score
        return max(type_scores.items(), key=lambda x: x[1])[0]
    
    def _get_contextual_weight(self, type1: DetectionType, type2: DetectionType) -> float:
        """
        Get contextual weight for a pair of detection types.
        
        Args:
            type1: First detection type
            type2: Second detection type
            
        Returns:
            Contextual weight (1.0 if no specific weight defined)
        """
        type_pair = (type1, type2)
        reverse_pair = (type2, type1)
        
        return self.contextual_weights.get(type_pair, 
               self.contextual_weights.get(reverse_pair, 1.0))
    
    def apply_suppression_logic(self, events: List[DetectionEvent]) -> List[DetectionEvent]:
        """
        Apply suppression rules to filter out contradictory or isolated signals.
        
        Args:
            events: List of detection events to filter
            
        Returns:
            Filtered list of events after applying suppression rules
        """
        if not events:
            return events
        
        suppressed_events = set()
        
        for rule in self.suppression_rules:
            # Find suppressing events (high-confidence events that can suppress others)
            suppressing_events = [
                event for event in events
                if (event.event_type in rule.suppressing_types and 
                    event.confidence >= rule.confidence_threshold)
            ]
            
            if not suppressing_events:
                continue
            
            # Find events to potentially suppress
            for suppressing_event in suppressing_events:
                for event in events:
                    if (event.event_type in rule.suppressed_types and 
                        event.event_id != suppressing_event.event_id):
                        
                        # Check if events are within suppression time window
                        time_diff = abs((event.timestamp - suppressing_event.timestamp).total_seconds())
                        if time_diff <= rule.time_window_seconds:
                            suppressed_events.add(event.event_id)
                            logger.debug(f"Suppressed {event.event_type.value} due to rule: {rule.description}")
        
        # Return events that weren't suppressed
        filtered_events = [event for event in events if event.event_id not in suppressed_events]
        
        if len(filtered_events) < len(events):
            logger.info(f"Suppressed {len(events) - len(filtered_events)} events using suppression rules")
        
        return filtered_events
    
    def detect_isolated_signals(self, events: List[DetectionEvent], 
                               isolation_threshold_seconds: float = 10.0) -> List[DetectionEvent]:
        """
        Detect isolated signals that occur without supporting evidence.
        
        Args:
            events: List of events to analyze
            isolation_threshold_seconds: Time window to check for supporting events
            
        Returns:
            List of isolated events
        """
        isolated_events = []
        
        for event in events:
            # Count supporting events within the isolation threshold
            supporting_count = 0
            
            for other_event in events:
                if other_event.event_id == event.event_id:
                    continue
                
                time_diff = abs((event.timestamp - other_event.timestamp).total_seconds())
                if time_diff <= isolation_threshold_seconds:
                    # Check if this is a supporting event (different type or same type with good confidence)
                    if (other_event.event_type != event.event_type or 
                        other_event.confidence >= 0.7):
                        supporting_count += 1
            
            # Event is isolated if it has no supporting events
            if supporting_count == 0:
                isolated_events.append(event)
        
        if isolated_events:
            logger.debug(f"Detected {len(isolated_events)} isolated signals")
        
        return isolated_events
    
    def calculate_contextual_confidence(self, primary_event: DetectionEvent, 
                                      related_events: List[DetectionEvent]) -> float:
        """
        Calculate contextual confidence score considering event relationships.
        
        Args:
            primary_event: The primary event
            related_events: Related events for context
            
        Returns:
            Contextual confidence score
        """
        if not related_events:
            return primary_event.confidence
        
        # Start with primary event confidence
        base_confidence = primary_event.confidence
        
        # Add contextual bonuses from related events
        contextual_bonus = 0.0
        
        for related_event in related_events:
            # Get contextual weight for this relationship
            weight = self._get_contextual_weight(primary_event.event_type, related_event.event_type)
            
            # Calculate temporal decay
            time_diff = abs((primary_event.timestamp - related_event.timestamp).total_seconds())
            temporal_weight = self._get_temporal_weight(time_diff)
            
            # Add weighted contribution
            contribution = related_event.confidence * weight * temporal_weight * 0.2
            contextual_bonus += contribution
        
        # Normalize contextual bonus
        contextual_bonus = min(contextual_bonus, 0.3)
        
        # Calculate final contextual confidence
        contextual_confidence = min(base_confidence + contextual_bonus, 1.0)
        
        return contextual_confidence
    
    def update_event_clusters(self, new_events: List[DetectionEvent]) -> None:
        """
        Update existing event clusters with new events.
        
        Args:
            new_events: New events to potentially add to clusters
        """
        if not new_events:
            return
        
        # Remove old clusters (older than correlation window)
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(seconds=self.config.correlation_window_seconds * 2)
        
        self.event_clusters = [
            cluster for cluster in self.event_clusters
            if any(event.timestamp > cutoff_time for event in cluster.events)
        ]
        
        # Try to add new events to existing clusters
        for event in new_events:
            added_to_existing = False
            
            for cluster in self.event_clusters:
                if self._should_add_to_cluster(event, cluster):
                    cluster.events.append(event)
                    cluster.cluster_score = self._calculate_cluster_score(cluster.events)
                    cluster.time_span_seconds = self._calculate_time_span(cluster.events)
                    added_to_existing = True
                    break
            
            # Create new cluster if needed
            if not added_to_existing:
                cluster_id = f"cluster_{len(self.event_clusters) + 1}_{event.timestamp.strftime('%H%M%S')}"
                new_cluster = EventCluster(
                    cluster_id=cluster_id,
                    events=[event],
                    cluster_score=event.confidence,
                    dominant_type=event.event_type,
                    time_span_seconds=0.0
                )
                self.event_clusters.append(new_cluster)
        
        logger.debug(f"Updated clusters: {len(self.event_clusters)} active clusters")
    
    def get_active_clusters(self) -> List[EventCluster]:
        """
        Get currently active event clusters.
        
        Returns:
            List of active event clusters
        """
        return self.event_clusters.copy()
    
    def get_suppression_statistics(self) -> Dict[str, int]:
        """
        Get statistics about suppression rule applications.
        
        Returns:
            Dictionary with suppression rule usage counts
        """
        # This would be implemented with actual tracking in a production system
        # For now, return placeholder statistics
        return {
            rule.description: 0 for rule in self.suppression_rules
        }
"""
Context Cue Analyzer - Core intelligence engine for multi-modal analysis.

This module contains the main ContextCueAnalyzer class that processes
and correlates detection signals across time and modality to generate
intelligent alerts with reduced false positives.
"""

import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import deque
import logging

from .models import (
    DetectionEvent, AnalysisResult, DetectionType, AlertRecommendation,
    SystemConfiguration, CorrelatedEvent
)
from .interfaces import EventProcessor


logger = logging.getLogger(__name__)


class ContextCueAnalyzer(EventProcessor):
    """
    Core intelligence engine that processes and correlates detection signals
    to generate context-aware alerts with reduced false positives.
    
    The analyzer maintains a sliding time window of recent events and applies
    temporal correlation analysis, confidence scoring, and contextual reasoning
    to determine when alerts should be triggered.
    """
    
    def __init__(self, config: SystemConfiguration):
        """
        Initialize the ContextCueAnalyzer.
        
        Args:
            config: System configuration containing thresholds and settings
        """
        self.config = config
        self.event_buffer: deque[DetectionEvent] = deque(maxlen=config.max_correlation_events)
        self.last_cleanup = datetime.now()
        
        # Performance tracking
        self.processing_times: List[float] = []
        self.events_processed = 0
        
        logger.info(f"ContextCueAnalyzer initialized with {config.correlation_window_seconds}s window")
    
    def process_event(self, event: DetectionEvent) -> Optional[AnalysisResult]:
        """
        Process a single detection event with temporal correlation analysis.
        
        Args:
            event: The detection event to process
            
        Returns:
            Analysis result if event warrants analysis, None otherwise
        """
        start_time = time.time()
        
        try:
            # Add event to buffer
            self.event_buffer.append(event)
            self.events_processed += 1
            
            # Clean up old events periodically
            if (datetime.now() - self.last_cleanup).seconds > 10:
                self._cleanup_old_events()
            
            # Check if event meets minimum confidence threshold
            threshold = self.config.get_threshold(event.event_type)
            if event.confidence < threshold:
                logger.debug(f"Event {event.event_type} below threshold: {event.confidence} < {threshold}")
                return None
            
            # Find correlated events within time window
            correlated_events = self._find_correlated_events(event)
            
            # Calculate confidence score based on correlations
            confidence_score = self._calculate_confidence_score(event, correlated_events)
            
            # Determine recommendation based on analysis
            recommendation = self._determine_recommendation(event, correlated_events, confidence_score)
            
            # Generate contextual reasoning
            reasoning = self._generate_reasoning(event, correlated_events, confidence_score)
            
            # Create analysis result
            result = AnalysisResult(
                primary_event=event,
                correlated_events=correlated_events,
                confidence_score=confidence_score,
                contextual_factors=self._extract_contextual_factors(event, correlated_events),
                recommendation=recommendation,
                reasoning=reasoning
            )
            
            # Track processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            self.processing_times.append(processing_time)
            
            logger.info(f"Processed {event.event_type} event: confidence={confidence_score:.3f}, "
                       f"recommendation={recommendation.value}, correlated_events={len(correlated_events)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing event {event.event_id}: {e}")
            return None
    
    def process_batch(self, events: List[DetectionEvent]) -> List[AnalysisResult]:
        """
        Process a batch of detection events.
        
        Args:
            events: List of detection events to process
            
        Returns:
            List of analysis results
        """
        results = []
        for event in events:
            result = self.process_event(event)
            if result:
                results.append(result)
        return results
    
    def _find_correlated_events(self, primary_event: DetectionEvent) -> List[DetectionEvent]:
        """
        Find events that are temporally correlated with the primary event.
        
        Args:
            primary_event: The event to find correlations for
            
        Returns:
            List of correlated events within the time window
        """
        correlated = []
        time_window = timedelta(seconds=self.config.correlation_window_seconds)
        
        for event in self.event_buffer:
            # Skip the primary event itself
            if event.event_id == primary_event.event_id:
                continue
            
            # Check if event is within time window
            time_diff = abs(primary_event.timestamp - event.timestamp)
            if time_diff <= time_window:
                correlated.append(event)
        
        # Sort by temporal proximity to primary event
        correlated.sort(key=lambda e: abs(primary_event.timestamp - e.timestamp))
        
        return correlated
    
    def _calculate_confidence_score(self, primary_event: DetectionEvent, 
                                  correlated_events: List[DetectionEvent]) -> float:
        """
        Calculate overall confidence score based on primary event and correlations.
        
        Args:
            primary_event: The primary detection event
            correlated_events: List of temporally correlated events
            
        Returns:
            Combined confidence score between 0.0 and 1.0
        """
        # Start with primary event confidence
        base_confidence = primary_event.confidence
        
        if not correlated_events:
            return base_confidence
        
        # Apply contextual relationship analysis
        correlation_boost = self._analyze_contextual_relationships(primary_event, correlated_events)
        
        # Apply false positive suppression
        suppression_factor = self._calculate_suppression_factor(primary_event, correlated_events)
        
        # General correlation boost based on number of correlated events
        general_boost = min(len(correlated_events) * 0.1, 0.2)
        correlation_boost += general_boost
        
        # Apply weighted average of correlated event confidences
        if correlated_events:
            avg_correlated_confidence = sum(e.confidence for e in correlated_events) / len(correlated_events)
            correlation_boost += (avg_correlated_confidence - 0.5) * 0.1
        
        # Calculate final confidence with boost and suppression
        final_confidence = min(base_confidence + correlation_boost - suppression_factor, 1.0)
        final_confidence = max(final_confidence, 0.0)  # Ensure non-negative
        
        return final_confidence
    
    def _analyze_contextual_relationships(self, primary_event: DetectionEvent,
                                        correlated_events: List[DetectionEvent]) -> float:
        """
        Analyze contextual relationships between different detection types.
        
        Args:
            primary_event: The primary detection event
            correlated_events: List of correlated events
            
        Returns:
            Confidence boost based on contextual relationships
        """
        boost = 0.0
        event_types = {event.event_type for event in correlated_events}
        
        # Strong positive correlations (mutually reinforcing)
        strong_correlations = [
            # Communication indicators
            (DetectionType.GAZE_AWAY, DetectionType.SUSPICIOUS_SPEECH, 0.3),
            (DetectionType.LIP_MOVEMENT, DetectionType.AUDIO_ANOMALY, 0.25),
            (DetectionType.LIP_MOVEMENT, DetectionType.SUSPICIOUS_SPEECH, 0.2),
            
            # Collaboration indicators
            (DetectionType.MULTIPLE_PEOPLE, DetectionType.MOBILE_DETECTED, 0.4),
            (DetectionType.MULTIPLE_PEOPLE, DetectionType.SUSPICIOUS_SPEECH, 0.3),
            
            # Impersonation/fraud indicators
            (DetectionType.FACE_SPOOF, DetectionType.MULTIPLE_PEOPLE, 0.35),
            (DetectionType.FACE_SPOOF, DetectionType.MOBILE_DETECTED, 0.25),
            
            # Distraction/assistance indicators
            (DetectionType.GAZE_AWAY, DetectionType.MOBILE_DETECTED, 0.2),
            (DetectionType.HEAD_POSE_SUSPICIOUS, DetectionType.GAZE_AWAY, 0.15),
        ]
        
        # Check for strong correlations
        for primary_type, correlated_type, boost_value in strong_correlations:
            if (primary_event.event_type == primary_type and correlated_type in event_types) or \
               (primary_event.event_type == correlated_type and primary_type in event_types):
                boost += boost_value
                logger.debug(f"Strong correlation found: {primary_type.value} + {correlated_type.value} (+{boost_value})")
        
        # Multi-modal reinforcement (3+ different detection types)
        unique_types = len(set([primary_event.event_type] + [e.event_type for e in correlated_events]))
        if unique_types >= 3:
            multimodal_boost = min(unique_types * 0.05, 0.15)
            boost += multimodal_boost
            logger.debug(f"Multi-modal reinforcement: {unique_types} types (+{multimodal_boost})")
        
        # Temporal clustering bonus (events very close in time)
        if correlated_events:
            timestamps = [e.timestamp for e in correlated_events] + [primary_event.timestamp]
            time_span = (max(timestamps) - min(timestamps)).total_seconds()
            if time_span <= 2.0:  # Events within 2 seconds
                temporal_boost = 0.1
                boost += temporal_boost
                logger.debug(f"Temporal clustering bonus: {time_span}s span (+{temporal_boost})")
        
        return boost
    
    def _calculate_suppression_factor(self, primary_event: DetectionEvent,
                                    correlated_events: List[DetectionEvent]) -> float:
        """
        Calculate false positive suppression based on signal contradictions.
        
        Args:
            primary_event: The primary detection event
            correlated_events: List of correlated events
            
        Returns:
            Suppression factor to reduce confidence (0.0 to 1.0)
        """
        suppression = 0.0
        event_types = {event.event_type for event in correlated_events}
        
        # Contradictory signal patterns that suggest false positives
        contradictions = [
            # Face spoof detected but no other suspicious activity
            (DetectionType.FACE_SPOOF, {DetectionType.MULTIPLE_PEOPLE, DetectionType.MOBILE_DETECTED, 
                                       DetectionType.SUSPICIOUS_SPEECH}, 0.2),
            
            # Gaze away without any communication indicators (might be natural)
            (DetectionType.GAZE_AWAY, {DetectionType.LIP_MOVEMENT, DetectionType.SUSPICIOUS_SPEECH, 
                                      DetectionType.AUDIO_ANOMALY}, 0.15),
            
            # Lip movement without audio (might be natural mouth movement)
            (DetectionType.LIP_MOVEMENT, {DetectionType.AUDIO_ANOMALY, DetectionType.SUSPICIOUS_SPEECH}, 0.1),
        ]
        
        # Check for contradictory patterns
        for primary_type, supporting_types, suppression_value in contradictions:
            if primary_event.event_type == primary_type:
                if not any(supporting_type in event_types for supporting_type in supporting_types):
                    suppression += suppression_value
                    logger.debug(f"Contradiction detected: {primary_type.value} without supporting signals (-{suppression_value})")
        
        # Isolated event suppression (single event with low base confidence)
        if not correlated_events and primary_event.confidence < 0.7:
            isolation_suppression = 0.1
            suppression += isolation_suppression
            logger.debug(f"Isolated low-confidence event suppression (-{isolation_suppression})")
        
        # Time-based suppression for very old correlations
        if correlated_events:
            timestamps = [e.timestamp for e in correlated_events] + [primary_event.timestamp]
            time_span = (max(timestamps) - min(timestamps)).total_seconds()
            if time_span > self.config.correlation_window_seconds * 0.8:  # Near window limit
                temporal_suppression = 0.05
                suppression += temporal_suppression
                logger.debug(f"Temporal suppression for old correlations (-{temporal_suppression})")
        
        return min(suppression, 0.5)  # Cap suppression at 0.5
    
    def _determine_recommendation(self, primary_event: DetectionEvent,
                                correlated_events: List[DetectionEvent],
                                confidence_score: float) -> AlertRecommendation:
        """
        Determine the recommended action based on analysis results with configurable sensitivity.
        
        Args:
            primary_event: The primary detection event
            correlated_events: List of correlated events
            confidence_score: Calculated confidence score
            
        Returns:
            Recommended action for this analysis
        """
        # Apply contextual decision rules
        context_factors = self._evaluate_context_factors(primary_event, correlated_events, confidence_score)
        
        # Critical events that always warrant immediate attention
        if context_factors['is_critical']:
            return AlertRecommendation.IMMEDIATE_INTERVENTION
        
        # High-priority scenarios
        if context_factors['is_high_priority']:
            return AlertRecommendation.ALERT_HIGH
        
        # Apply configurable thresholds based on deployment mode
        thresholds = self._get_alert_thresholds()
        
        if confidence_score > thresholds['high']:
            return AlertRecommendation.ALERT_HIGH
        elif confidence_score > thresholds['medium']:
            return AlertRecommendation.ALERT_MEDIUM
        elif confidence_score > thresholds['low']:
            return AlertRecommendation.ALERT_LOW
        elif confidence_score > thresholds['monitor']:
            return AlertRecommendation.MONITOR
        else:
            return AlertRecommendation.IGNORE
    
    def _evaluate_context_factors(self, primary_event: DetectionEvent,
                                correlated_events: List[DetectionEvent],
                                confidence_score: float) -> Dict[str, bool]:
        """
        Evaluate contextual factors that influence alert decisions.
        
        Args:
            primary_event: The primary detection event
            correlated_events: List of correlated events
            confidence_score: Calculated confidence score
            
        Returns:
            Dictionary of boolean context factors
        """
        event_types = {event.event_type for event in correlated_events}
        
        # Critical scenarios requiring immediate intervention
        is_critical = (
            # High-confidence face spoofing
            (primary_event.event_type == DetectionType.FACE_SPOOF and confidence_score > 0.8) or
            
            # Multiple people with high confidence
            (primary_event.event_type == DetectionType.MULTIPLE_PEOPLE and confidence_score > 0.85) or
            
            # Multiple people + mobile device combination
            (primary_event.event_type == DetectionType.MULTIPLE_PEOPLE and 
             DetectionType.MOBILE_DETECTED in event_types) or
            
            # Face spoof + multiple people (impersonation attempt)
            (primary_event.event_type == DetectionType.FACE_SPOOF and 
             DetectionType.MULTIPLE_PEOPLE in event_types)
        )
        
        # High-priority scenarios
        is_high_priority = (
            # High confidence with multiple correlations
            (confidence_score > 0.85 and len(correlated_events) >= 2) or
            
            # Communication pattern with high confidence
            (confidence_score > 0.8 and 
             ((primary_event.event_type == DetectionType.GAZE_AWAY and 
               DetectionType.SUSPICIOUS_SPEECH in event_types) or
              (primary_event.event_type == DetectionType.LIP_MOVEMENT and 
               DetectionType.AUDIO_ANOMALY in event_types))) or
            
            # Mobile device with communication indicators
            (primary_event.event_type == DetectionType.MOBILE_DETECTED and
             (DetectionType.SUSPICIOUS_SPEECH in event_types or 
              DetectionType.LIP_MOVEMENT in event_types))
        )
        
        return {
            'is_critical': is_critical,
            'is_high_priority': is_high_priority
        }
    
    def _get_alert_thresholds(self) -> Dict[str, float]:
        """
        Get alert thresholds based on deployment mode and configuration.
        
        Returns:
            Dictionary of threshold values for different alert levels
        """
        # Base thresholds
        base_thresholds = {
            'high': 0.75,
            'medium': 0.65,
            'low': 0.55,
            'monitor': 0.45
        }
        
        # Adjust thresholds based on deployment mode
        if self.config.deployment_mode.value == 'development':
            # More sensitive in development for testing
            return {
                'high': base_thresholds['high'] - 0.1,
                'medium': base_thresholds['medium'] - 0.1,
                'low': base_thresholds['low'] - 0.1,
                'monitor': base_thresholds['monitor'] - 0.1
            }
        elif self.config.deployment_mode.value == 'production':
            # Less sensitive in production to reduce false positives
            return {
                'high': base_thresholds['high'] + 0.05,
                'medium': base_thresholds['medium'] + 0.05,
                'low': base_thresholds['low'] + 0.05,
                'monitor': base_thresholds['monitor'] + 0.05
            }
        else:
            return base_thresholds
    
    def _generate_reasoning(self, primary_event: DetectionEvent,
                          correlated_events: List[DetectionEvent],
                          confidence_score: float) -> str:
        """
        Generate comprehensive human-readable reasoning for the analysis decision.
        
        Args:
            primary_event: The primary detection event
            correlated_events: List of correlated events
            confidence_score: Calculated confidence score
            
        Returns:
            Detailed textual explanation of the analysis reasoning
        """
        reasoning_parts = []
        
        # Primary event description with context
        reasoning_parts.append(f"Primary detection: {primary_event.event_type.value} "
                             f"(confidence: {primary_event.confidence:.2f}, source: {primary_event.source})")
        
        if correlated_events:
            # Describe correlations with timing
            event_descriptions = []
            for event in correlated_events:
                time_diff = abs((primary_event.timestamp - event.timestamp).total_seconds())
                event_descriptions.append(f"{event.event_type.value} ({event.confidence:.2f}, {time_diff:.1f}s)")
            
            reasoning_parts.append(f"Correlated with {len(correlated_events)} events: "
                                 f"{'; '.join(event_descriptions)}")
            
            # Analyze and explain specific patterns
            pattern_explanations = self._analyze_behavioral_patterns(primary_event, correlated_events)
            reasoning_parts.extend(pattern_explanations)
            
            # Temporal analysis
            timestamps = [e.timestamp for e in correlated_events] + [primary_event.timestamp]
            time_span = (max(timestamps) - min(timestamps)).total_seconds()
            reasoning_parts.append(f"Event cluster spans {time_span:.1f} seconds")
            
        else:
            reasoning_parts.append("No temporal correlations found - isolated event")
            
            # Explain why isolated event might be significant or not
            if primary_event.confidence > 0.8:
                reasoning_parts.append("High individual confidence despite isolation")
            elif primary_event.confidence < 0.6:
                reasoning_parts.append("Low individual confidence and isolated - likely false positive")
        
        # Contextual factors explanation
        context_explanation = self._explain_contextual_factors(primary_event, correlated_events)
        if context_explanation:
            reasoning_parts.append(context_explanation)
        
        # Final assessment with confidence change explanation
        confidence_change = confidence_score - primary_event.confidence
        if abs(confidence_change) > 0.05:
            change_direction = "increased" if confidence_change > 0 else "decreased"
            reasoning_parts.append(f"Confidence {change_direction} by {abs(confidence_change):.2f} "
                                 f"due to contextual analysis")
        
        reasoning_parts.append(f"Final confidence: {confidence_score:.2f}")
        
        return ". ".join(reasoning_parts)
    
    def _analyze_behavioral_patterns(self, primary_event: DetectionEvent,
                                   correlated_events: List[DetectionEvent]) -> List[str]:
        """
        Analyze and explain behavioral patterns in the event sequence.
        
        Args:
            primary_event: The primary detection event
            correlated_events: List of correlated events
            
        Returns:
            List of pattern explanation strings
        """
        explanations = []
        event_type_set = {e.event_type for e in correlated_events}
        
        # Communication patterns
        if (primary_event.event_type == DetectionType.GAZE_AWAY and 
            DetectionType.SUSPICIOUS_SPEECH in event_type_set):
            explanations.append("COMMUNICATION PATTERN: Looking away while speaking suggests external communication")
        
        if (primary_event.event_type == DetectionType.LIP_MOVEMENT and 
            DetectionType.AUDIO_ANOMALY in event_type_set):
            explanations.append("SPEAKING PATTERN: Lip movement synchronized with audio anomaly confirms verbal communication")
        
        if (primary_event.event_type == DetectionType.LIP_MOVEMENT and 
            DetectionType.SUSPICIOUS_SPEECH in event_type_set):
            explanations.append("VERBAL COMMUNICATION: Lip movement with suspicious speech indicates active conversation")
        
        # Collaboration patterns
        if (primary_event.event_type == DetectionType.MULTIPLE_PEOPLE and 
            DetectionType.MOBILE_DETECTED in event_type_set):
            explanations.append("COLLABORATION PATTERN: Multiple people with mobile device suggests coordinated cheating")
        
        if (primary_event.event_type == DetectionType.MULTIPLE_PEOPLE and 
            DetectionType.SUSPICIOUS_SPEECH in event_type_set):
            explanations.append("GROUP COMMUNICATION: Multiple people with speech indicates discussion or assistance")
        
        # Impersonation/fraud patterns
        if (primary_event.event_type == DetectionType.FACE_SPOOF and 
            DetectionType.MULTIPLE_PEOPLE in event_type_set):
            explanations.append("IMPERSONATION ATTEMPT: Face spoofing with multiple people suggests identity fraud")
        
        if (primary_event.event_type == DetectionType.FACE_SPOOF and 
            DetectionType.MOBILE_DETECTED in event_type_set):
            explanations.append("DIGITAL FRAUD: Face spoofing with mobile device indicates sophisticated cheating attempt")
        
        # Technology-assisted cheating
        if (primary_event.event_type == DetectionType.MOBILE_DETECTED and
            (DetectionType.GAZE_AWAY in event_type_set or DetectionType.SUSPICIOUS_SPEECH in event_type_set)):
            explanations.append("TECHNOLOGY ASSISTANCE: Mobile device with communication indicators suggests external help")
        
        # Distraction patterns
        if (primary_event.event_type == DetectionType.GAZE_AWAY and 
            DetectionType.HEAD_POSE_SUSPICIOUS in event_type_set):
            explanations.append("ATTENTION DIVERSION: Gaze and head movement suggest looking at unauthorized materials")
        
        return explanations
    
    def _explain_contextual_factors(self, primary_event: DetectionEvent,
                                  correlated_events: List[DetectionEvent]) -> str:
        """
        Explain the contextual factors that influenced the analysis.
        
        Args:
            primary_event: The primary detection event
            correlated_events: List of correlated events
            
        Returns:
            Explanation of contextual factors
        """
        factors = []
        
        # Multi-modal analysis
        unique_types = len(set([primary_event.event_type] + [e.event_type for e in correlated_events]))
        if unique_types >= 3:
            factors.append(f"Multi-modal detection ({unique_types} different signal types)")
        
        # Temporal clustering
        if correlated_events:
            timestamps = [e.timestamp for e in correlated_events] + [primary_event.timestamp]
            time_span = (max(timestamps) - min(timestamps)).total_seconds()
            if time_span <= 2.0:
                factors.append("Rapid event sequence (high temporal correlation)")
            elif time_span > 4.0:
                factors.append("Extended event sequence (sustained suspicious activity)")
        
        # Confidence consistency
        if correlated_events:
            confidences = [e.confidence for e in correlated_events] + [primary_event.confidence]
            avg_confidence = sum(confidences) / len(confidences)
            if avg_confidence > 0.8:
                factors.append("Consistently high confidence across all detections")
            elif max(confidences) - min(confidences) > 0.3:
                factors.append("Variable confidence levels across detections")
        
        if factors:
            return "Contextual factors: " + "; ".join(factors)
        return ""
    
    def _extract_contextual_factors(self, primary_event: DetectionEvent,
                                  correlated_events: List[DetectionEvent]) -> Dict[str, float]:
        """
        Extract contextual factors that influenced the analysis.
        
        Args:
            primary_event: The primary detection event
            correlated_events: List of correlated events
            
        Returns:
            Dictionary of contextual factors and their weights
        """
        factors = {
            'primary_confidence': primary_event.confidence,
            'correlation_count': len(correlated_events),
            'time_span': 0.0,
            'event_diversity': 0.0
        }
        
        if correlated_events:
            # Calculate time span of correlated events
            timestamps = [e.timestamp for e in correlated_events] + [primary_event.timestamp]
            time_span = (max(timestamps) - min(timestamps)).total_seconds()
            factors['time_span'] = time_span
            
            # Calculate event type diversity
            unique_types = len(set(e.event_type for e in correlated_events + [primary_event]))
            factors['event_diversity'] = unique_types / (len(correlated_events) + 1)
            
            # Average confidence of correlated events
            factors['avg_correlated_confidence'] = sum(e.confidence for e in correlated_events) / len(correlated_events)
        
        return factors
    
    def _cleanup_old_events(self) -> None:
        """Remove events older than the correlation window from the buffer."""
        cutoff_time = datetime.now() - timedelta(seconds=self.config.correlation_window_seconds * 2)
        
        # Convert deque to list for filtering, then back to deque
        recent_events = [event for event in self.event_buffer if event.timestamp > cutoff_time]
        self.event_buffer.clear()
        self.event_buffer.extend(recent_events)
        
        self.last_cleanup = datetime.now()
        logger.debug(f"Cleaned up old events, {len(recent_events)} events remaining")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics for the analyzer.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.processing_times:
            return {'avg_processing_time_ms': 0.0, 'events_processed': 0}
        
        return {
            'avg_processing_time_ms': sum(self.processing_times) / len(self.processing_times),
            'max_processing_time_ms': max(self.processing_times),
            'min_processing_time_ms': min(self.processing_times),
            'events_processed': self.events_processed,
            'buffer_size': len(self.event_buffer)
        }
    
    def reset_metrics(self) -> None:
        """Reset performance tracking metrics."""
        self.processing_times.clear()
        self.events_processed = 0
        logger.info("Performance metrics reset")
    
    def should_trigger_alert(self, analysis: AnalysisResult) -> bool:
        """
        Determine if an analysis result should trigger an alert.
        
        Args:
            analysis: The analysis result to evaluate
            
        Returns:
            True if an alert should be triggered, False otherwise
        """
        alert_recommendations = {
            AlertRecommendation.ALERT_LOW,
            AlertRecommendation.ALERT_MEDIUM,
            AlertRecommendation.ALERT_HIGH,
            AlertRecommendation.IMMEDIATE_INTERVENTION
        }
        
        return analysis.recommendation in alert_recommendations
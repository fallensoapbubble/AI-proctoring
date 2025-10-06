"""
Speech analysis component for detecting suspicious audio content.
"""

import threading
import time
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
import queue
import logging

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

from context_engine.interfaces import DetectionSource
from context_engine.models import DetectionEvent, DetectionType
from shared_utils.detection_utils import normalize_confidence


class SpeechAnalyzer(DetectionSource):
    """
    Analyzes audio input for suspicious speech patterns and keywords.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the speech analyzer."""
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.SpeechAnalyzer")
        
        # Speech recognition setup
        self.recognizer = None
        self.microphone = None
        self.is_running = False
        self.listening_thread = None
        
        # Detection parameters
        self.cheating_keywords = set(self.config.get('cheating_keywords', [
            "answer", "solution", "help", "cheat", "google", "search", 
            "tell me", "what is", "how do", "give me", "show me"
        ]))
        
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        self.listen_timeout = self.config.get('listen_timeout', 5)
        self.phrase_timeout = self.config.get('phrase_timeout', 1)
        
        # Event queue for thread-safe communication
        self.event_queue = queue.Queue()
        
        # Statistics tracking
        self.total_phrases_detected = 0
        self.suspicious_phrases_detected = 0
        
        if SPEECH_RECOGNITION_AVAILABLE:
            self.recognizer = sr.Recognizer()
        else:
            self.logger.warning("Speech recognition not available - install speech_recognition package")
    
    def get_source_name(self) -> str:
        """Return the unique name of this detection source."""
        return "speech_analyzer"
    
    def is_available(self) -> bool:
        """Check if microphone and speech recognition are available."""
        if not SPEECH_RECOGNITION_AVAILABLE:
            return False
        
        try:
            with sr.Microphone() as source:
                pass
            return True
        except Exception:
            return False
    
    def start_detection(self) -> None:
        """Start speech recognition in a background thread."""
        if self.is_running:
            return
        
        try:
            if not SPEECH_RECOGNITION_AVAILABLE:
                self.logger.warning("Speech recognition not available, starting in mock mode")
                self.is_running = True
                return
            
            self.microphone = sr.Microphone()
            self.is_running = True
            
            # Start listening thread
            self.listening_thread = threading.Thread(
                target=self._listen_continuously,
                daemon=True
            )
            self.listening_thread.start()
            self.logger.info("SpeechAnalyzer started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start SpeechAnalyzer: {e}")
            self.is_running = False
            raise
    
    def stop_detection(self) -> None:
        """Stop speech recognition and cleanup resources."""
        try:
            self.is_running = False
            
            if self.listening_thread and self.listening_thread.is_alive():
                self.listening_thread.join(timeout=2)
            
            self.microphone = None
            self.listening_thread = None
            
            # Clear event queue
            while not self.event_queue.empty():
                try:
                    self.event_queue.get_nowait()
                except queue.Empty:
                    break
            
        except Exception:
            pass
    
    def get_health_status(self) -> Dict[str, Any]:
        """Return health status information for monitoring."""
        return {
            'source_name': self.get_source_name(),
            'is_running': self.is_running,
            'is_available': self.is_available(),
            'microphone_initialized': self.microphone is not None,
            'listening_thread_alive': self.listening_thread.is_alive() if self.listening_thread else False,
            'event_queue_size': self.event_queue.qsize(),
            'total_phrases_detected': self.total_phrases_detected,
            'suspicious_phrases_detected': self.suspicious_phrases_detected,
            'config': {
                'keywords_count': len(self.cheating_keywords),
                'confidence_threshold': self.confidence_threshold,
                'listen_timeout': self.listen_timeout
            }
        }
    
    def get_pending_events(self) -> List[DetectionEvent]:
        """Get all pending detection events from the queue."""
        events = []
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                events.append(event)
            except queue.Empty:
                break
        return events
    
    def _listen_continuously(self) -> None:
        """Continuously listen for audio and process speech."""
        if not SPEECH_RECOGNITION_AVAILABLE or not self.microphone:
            return
        
        try:
            with self.microphone as source:
                # Adjust for ambient noise initially
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
        except Exception as e:
            self.logger.error(f"Failed to initialize microphone: {e}")
            return
        
        while self.is_running:
            try:
                with self.microphone as source:
                    # Listen for audio with timeout
                    audio = self.recognizer.listen(
                        source,
                        timeout=self.listen_timeout,
                        phrase_time_limit=self.phrase_timeout
                    )
                
                # Process audio in background to avoid blocking
                threading.Thread(
                    target=self._process_audio,
                    args=(audio,),
                    daemon=True
                ).start()
                
            except sr.WaitTimeoutError:
                # Normal timeout, continue listening
                continue
            except Exception as e:
                if self.is_running:
                    self.logger.error(f"Error in speech listening: {e}")
                    time.sleep(1)  # Brief pause before retrying
    
    def _process_audio(self, audio) -> None:
        """Process captured audio for speech recognition and analysis."""
        if not SPEECH_RECOGNITION_AVAILABLE:
            return
        
        try:
            # Recognize speech using Google Speech Recognition
            text = self.recognizer.recognize_google(audio)
            self.total_phrases_detected += 1
            
            # Analyze for suspicious content
            detection_event = self._analyze_speech_content(text)
            
            if detection_event:
                self.suspicious_phrases_detected += 1
                self.event_queue.put(detection_event)
            
        except sr.UnknownValueError:
            # Speech was unintelligible
            pass
        except sr.RequestError as e:
            # Speech recognition service error
            self.logger.error(f"Speech recognition service error: {e}")
        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")
    
    def _analyze_speech_content(self, text: str) -> Optional[DetectionEvent]:
        """Analyze recognized speech text for suspicious content."""
        if not text:
            return None
        
        text_lower = text.lower()
        
        # Check for exact keyword matches
        exact_matches = []
        for keyword in self.cheating_keywords:
            if keyword.lower() in text_lower:
                exact_matches.append(keyword)
        
        if not exact_matches:
            return None
        
        # Calculate confidence based on number and specificity of matches
        match_count = len(exact_matches)
        text_length = len(text.split())
        
        # Higher confidence for more matches and shorter phrases
        base_confidence = min(match_count * 0.3, 1.0)
        length_factor = max(0.5, 1.0 - (text_length - match_count) * 0.1)
        
        confidence = normalize_confidence(base_confidence * length_factor)
        
        # Only trigger if confidence meets threshold
        if confidence < self.confidence_threshold:
            return None
        
        metadata = {
            'recognized_text': text,
            'matched_keywords': exact_matches,
            'match_count': match_count,
            'text_length': text_length,
            'base_confidence': float(base_confidence),
            'length_factor': float(length_factor),
            'all_keywords': list(self.cheating_keywords)
        }
        
        return DetectionEvent(
            event_type=DetectionType.SUSPICIOUS_SPEECH,
            timestamp=datetime.now(),
            confidence=confidence,
            source=self.get_source_name(),
            metadata=metadata
        )
    
    def add_keywords(self, keywords: List[str]) -> None:
        """Add new keywords to the detection list."""
        self.cheating_keywords.update(keyword.lower() for keyword in keywords)
    
    def remove_keywords(self, keywords: List[str]) -> None:
        """Remove keywords from the detection list."""
        for keyword in keywords:
            self.cheating_keywords.discard(keyword.lower())
    
    def get_keywords(self) -> Set[str]:
        """Get current set of detection keywords."""
        return self.cheating_keywords.copy()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update analyzer configuration."""
        self.config.update(new_config)
        
        # Update parameters
        if 'cheating_keywords' in new_config:
            self.cheating_keywords = set(new_config['cheating_keywords'])
        
        self.confidence_threshold = self.config.get('confidence_threshold', self.confidence_threshold)
        self.listen_timeout = self.config.get('listen_timeout', self.listen_timeout)
        self.phrase_timeout = self.config.get('phrase_timeout', self.phrase_timeout)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics."""
        total = self.total_phrases_detected
        suspicious = self.suspicious_phrases_detected
        
        return {
            'total_phrases_detected': total,
            'suspicious_phrases_detected': suspicious,
            'detection_rate': suspicious / total if total > 0 else 0.0,
            'keywords_count': len(self.cheating_keywords),
            'pending_events': self.event_queue.qsize()
        }
    
    def reset_statistics(self) -> None:
        """Reset detection statistics."""
        self.total_phrases_detected = 0
        self.suspicious_phrases_detected = 0
"""
Fixed Detector Manager - Comprehensive implementation with proper error handling.

This module manages all detection components with robust error handling
and simplified logging.
"""

import threading
import time
import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import queue

# Import fixed detector implementations
from .fixed_gaze_detector import FixedGazeDetector
from .fixed_mobile_detector import FixedMobileDetector

# Try to import other detectors with fallbacks
try:
    from .lip_movement_detector import LipMovementDetector
except ImportError:
    LipMovementDetector = None

try:
    from .speech_analyzer import SpeechAnalyzer
except ImportError:
    SpeechAnalyzer = None

try:
    from .person_detector import PersonDetector
except ImportError:
    PersonDetector = None

try:
    from .face_spoof_detector import FaceSpoofDetector
except ImportError:
    FaceSpoofDetector = None

try:
    from context_engine.interfaces import DetectionSource
    from context_engine.models import DetectionEvent, SystemConfiguration
    from context_engine.analyzer import ContextCueAnalyzer
except ImportError:
    # Create minimal fallbacks
    class DetectionSource:
        pass
    
    class DetectionEvent:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class SystemConfiguration:
        def __init__(self):
            pass
    
    class ContextCueAnalyzer:
        def __init__(self, config):
            pass


class FixedDetectorManager:
    """
    Fixed detector manager with robust error handling and simplified logging.
    """
    
    def __init__(self, config=None, analyzer=None):
        """Initialize the fixed detector manager."""
        self.config = config
        self.analyzer = analyzer
        
        # Initialize detectors
        self.detectors: Dict[str, Any] = {}
        self._initialize_detectors()
        
        # Processing state
        self.is_running = False
        self.video_thread: Optional[threading.Thread] = None
        self.audio_thread: Optional[threading.Thread] = None
        
        # Video capture
        self.video_capture: Optional[cv2.VideoCapture] = None
        self.current_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        
        # Event callbacks
        self.event_callbacks: List[Callable] = []
        
        # Performance monitoring
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        
        # Health monitoring
        self.detector_health: Dict[str, Dict[str, Any]] = {}
        self.last_health_check = datetime.now()
        
        print(f"FixedDetectorManager initialized with {len(self.detectors)} detectors")
    
    def _initialize_detectors(self) -> None:
        """Initialize all detector components with error handling."""
        detector_configs = self._get_detector_configs()
        
        # Initialize fixed detectors (guaranteed to work)
        try:
            self.detectors['gaze'] = FixedGazeDetector(detector_configs.get('gaze', {}))
            print("✓ Fixed gaze detector initialized")
        except Exception as e:
            print(f"✗ Failed to initialize gaze detector: {e}")
        
        try:
            self.detectors['mobile'] = FixedMobileDetector(detector_configs.get('mobile', {}))
            print("✓ Fixed mobile detector initialized")
        except Exception as e:
            print(f"✗ Failed to initialize mobile detector: {e}")
        
        # Try to initialize other detectors with fallbacks
        if LipMovementDetector:
            try:
                self.detectors['lip_movement'] = LipMovementDetector(detector_configs.get('lip_movement', {}))
                print("✓ Lip movement detector initialized")
            except Exception as e:
                print(f"✗ Failed to initialize lip movement detector: {e}")
        
        if SpeechAnalyzer:
            try:
                self.detectors['speech'] = SpeechAnalyzer(detector_configs.get('speech', {}))
                print("✓ Speech analyzer initialized")
            except Exception as e:
                print(f"✗ Failed to initialize speech analyzer: {e}")
        
        if PersonDetector:
            try:
                self.detectors['person'] = PersonDetector(detector_configs.get('person', {}))
                print("✓ Person detector initialized")
            except Exception as e:
                print(f"✗ Failed to initialize person detector: {e}")
        
        if FaceSpoofDetector:
            try:
                self.detectors['face_spoof'] = FaceSpoofDetector(detector_configs.get('face_spoof', {}))
                print("✓ Face spoof detector initialized")
            except Exception as e:
                print(f"✗ Failed to initialize face spoof detector: {e}")
        
        print(f"Initialized {len(self.detectors)} detectors successfully")
    
    def _get_detector_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get detector-specific configurations."""
        return {
            'gaze': {
                'yaw_threshold': 20,
                'pitch_min': -10,
                'pitch_max': 15,
                'roll_min': 80,
                'roll_max': 110
            },
            'lip_movement': {
                'movement_threshold': 0.003,
                'smoothing_window': 3
            },
            'speech': {
                'cheating_keywords': [
                    "answer", "solution", "help", "cheat", "google", "search",
                    "tell me", "what is", "how do", "give me", "show me"
                ],
                'confidence_threshold': 0.6,
                'listen_timeout': 5
            },
            'mobile': {
                'confidence_threshold': 0.6,
                'model_path': 'yolov8s.pt',
                'mobile_class_ids': [67]
            },
            'person': {
                'confidence_threshold': 0.5,
                'max_expected_people': 1,
                'model_path': 'yolov5s.pt'
            },
            'face_spoof': {
                'confidence_threshold': 0.5,
                'spoof_confidence_threshold': 0.7,
                'real_confidence_threshold': 0.85,
                'model_path': 'yolov8n.pt'
            }
        }
    
    def start_detection(self, use_camera: bool = False) -> None:
        """Start all detectors and begin processing."""
        if self.is_running:
            print("FixedDetectorManager already running")
            return
        
        try:
            # Only initialize video capture if explicitly requested
            if use_camera:
                try:
                    self.video_capture = cv2.VideoCapture(0)
                    if not self.video_capture.isOpened():
                        print("Failed to open camera - will use web streaming mode")
                        self.video_capture = None
                except Exception as e:
                    print(f"Camera initialization failed: {e}")
                    self.video_capture = None
            else:
                print("Starting in web streaming mode - camera will be accessed via browser")
                self.video_capture = None
            
            # Start all available detectors
            failed_detectors = []
            for name, detector in self.detectors.items():
                try:
                    if hasattr(detector, 'is_available') and detector.is_available():
                        if hasattr(detector, 'start_detection'):
                            detector.start_detection()
                        print(f"✓ Started {name} detector")
                    else:
                        failed_detectors.append(name)
                        print(f"✗ {name} detector not available")
                except Exception as e:
                    failed_detectors.append(name)
                    print(f"✗ Failed to start {name} detector: {e}")
            
            if failed_detectors:
                print(f"Failed to start detectors: {failed_detectors}")
            
            # Start processing threads
            self.is_running = True
            
            # Only start video thread if we have direct camera access
            if self.video_capture:
                self.video_thread = threading.Thread(target=self._video_processing_loop, daemon=True)
                self.video_thread.start()
                print("Started video processing thread with direct camera access")
            else:
                print("Video processing will be handled via web streaming")
            
            # Always start audio thread for speech analysis if available
            if 'speech' in self.detectors:
                self.audio_thread = threading.Thread(target=self._audio_processing_loop, daemon=True)
                self.audio_thread.start()
                print("Started audio processing thread")
            
            print("FixedDetectorManager started successfully")
            
        except Exception as e:
            print(f"Failed to start FixedDetectorManager: {e}")
            self.stop_detection()
            raise
    
    def stop_detection(self) -> None:
        """Stop all detectors and cleanup resources."""
        try:
            self.is_running = False
            
            # Wait for threads to finish
            if self.video_thread and self.video_thread.is_alive():
                self.video_thread.join(timeout=2)
            if self.audio_thread and self.audio_thread.is_alive():
                self.audio_thread.join(timeout=2)
            
            # Stop all detectors
            for name, detector in self.detectors.items():
                try:
                    if hasattr(detector, 'stop_detection'):
                        detector.stop_detection()
                    print(f"✓ Stopped {name} detector")
                except Exception as e:
                    print(f"✗ Error stopping {name} detector: {e}")
            
            # Release video capture
            if self.video_capture:
                self.video_capture.release()
                self.video_capture = None
            
            # Clear current frame
            with self.frame_lock:
                self.current_frame = None
            
            print("FixedDetectorManager stopped")
            
        except Exception as e:
            print(f"Error stopping FixedDetectorManager: {e}")
    
    def _video_processing_loop(self) -> None:
        """Main video processing loop."""
        print("Starting video processing loop")
        
        while self.is_running and self.video_capture:
            try:
                ret, frame = self.video_capture.read()
                if not ret:
                    print("Failed to read frame from camera")
                    time.sleep(0.1)
                    continue
                
                # Update current frame
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                # Update FPS counter
                self._update_fps_counter()
                
                # Process frame with video-based detectors
                self._process_video_frame(frame)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                if self.is_running:
                    print(f"Error in video processing loop: {e}")
                time.sleep(0.1)
        
        print("Video processing loop stopped")
    
    def _audio_processing_loop(self) -> None:
        """Main audio processing loop."""
        print("Starting audio processing loop")
        
        speech_analyzer = self.detectors.get('speech')
        if not speech_analyzer:
            print("Speech analyzer not available for audio processing")
            return
        
        while self.is_running:
            try:
                # Process pending speech events
                if hasattr(speech_analyzer, 'get_pending_events'):
                    events = speech_analyzer.get_pending_events()
                    for event in events:
                        self._process_detection_event(event)
                
                time.sleep(0.1)
                
            except Exception as e:
                if self.is_running:
                    print(f"Error in audio processing loop: {e}")
                time.sleep(0.5)
        
        print("Audio processing loop stopped")
    
    def _process_video_frame(self, frame: np.ndarray) -> None:
        """Process a video frame with all video-based detectors."""
        # Process with each video-based detector
        video_detectors = ['gaze', 'lip_movement', 'mobile', 'person', 'face_spoof']
        
        for detector_name in video_detectors:
            detector = self.detectors.get(detector_name)
            if not detector or not hasattr(detector, 'is_running'):
                continue
            
            # Check if detector is running
            if hasattr(detector, 'is_running') and not detector.is_running:
                continue
            
            try:
                # Call appropriate detection method based on detector type
                event = None
                
                if detector_name == 'gaze' and hasattr(detector, 'detect_gaze_direction'):
                    event = detector.detect_gaze_direction(frame)
                elif detector_name == 'lip_movement' and hasattr(detector, 'detect_lip_movement'):
                    event = detector.detect_lip_movement(frame)
                elif detector_name == 'mobile' and hasattr(detector, 'detect_mobile_devices'):
                    event = detector.detect_mobile_devices(frame)
                elif detector_name == 'person' and hasattr(detector, 'detect_people'):
                    event = detector.detect_people(frame)
                elif detector_name == 'face_spoof' and hasattr(detector, 'detect_face_spoof'):
                    event = detector.detect_face_spoof(frame)
                
                if event:
                    self._process_detection_event(event)
                    
            except Exception as e:
                print(f"Error processing frame with {detector_name} detector: {e}")
    
    def _process_detection_event(self, event) -> None:
        """Process a detection event through the analyzer and callbacks."""
        try:
            # Process event through context analyzer if available
            if self.analyzer and hasattr(self.analyzer, 'process_event'):
                try:
                    analysis_result = self.analyzer.process_event(event)
                    
                    if analysis_result:
                        # Trigger callbacks for significant events
                        if hasattr(analysis_result, 'recommendation'):
                            if analysis_result.recommendation.value != 'ignore':
                                for callback in self.event_callbacks:
                                    try:
                                        callback(event)
                                    except Exception as e:
                                        print(f"Error in event callback: {e}")
                except Exception as e:
                    print(f"Error in analyzer: {e}")
            else:
                # If no analyzer, trigger callbacks directly
                for callback in self.event_callbacks:
                    try:
                        callback(event)
                    except Exception as e:
                        print(f"Error in event callback: {e}")
            
        except Exception as e:
            print(f"Error processing detection event: {e}")
    
    def _update_fps_counter(self) -> None:
        """Update FPS counter for performance monitoring."""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:  # Update every second
            self.current_fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def add_event_callback(self, callback: Callable) -> None:
        """Add a callback function to be called when significant events are detected."""
        self.event_callbacks.append(callback)
        print(f"Added event callback: {callback.__name__ if hasattr(callback, '__name__') else 'callback'}")
    
    def remove_event_callback(self, callback: Callable) -> None:
        """Remove an event callback."""
        if callback in self.event_callbacks:
            self.event_callbacks.remove(callback)
            print(f"Removed event callback: {callback.__name__ if hasattr(callback, '__name__') else 'callback'}")
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current video frame."""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def process_web_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a frame received from the web frontend."""
        if not self.is_running:
            return {'error': 'FixedDetectorManager not running'}
        
        try:
            # Update current frame
            with self.frame_lock:
                self.current_frame = frame.copy()
            
            # Update FPS counter
            self._update_fps_counter()
            
            # Process frame with video-based detectors
            self._process_video_frame(frame)
            
            return {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'frame_processed': True,
                'fps': self.current_fps
            }
            
        except Exception as e:
            print(f"Error processing web frame: {e}")
            return {'error': str(e)}
    
    def get_detector_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all detectors."""
        health_status = {}
        
        for name, detector in self.detectors.items():
            try:
                if hasattr(detector, 'get_health_status'):
                    health_status[name] = detector.get_health_status()
                else:
                    health_status[name] = {
                        'is_running': hasattr(detector, 'is_running') and detector.is_running,
                        'is_available': hasattr(detector, 'is_available') and detector.is_available()
                    }
            except Exception as e:
                health_status[name] = {
                    'error': str(e),
                    'is_running': False,
                    'is_available': False
                }
        
        return health_status
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the detector manager."""
        active_detectors = 0
        for detector in self.detectors.values():
            if hasattr(detector, 'is_running') and detector.is_running:
                active_detectors += 1
        
        return {
            'current_fps': self.current_fps,
            'is_running': self.is_running,
            'active_detectors': active_detectors,
            'total_detectors': len(self.detectors),
            'video_capture_active': self.video_capture is not None,
            'analyzer_available': self.analyzer is not None
        }
    
    def update_detector_config(self, detector_name: str, config: Dict[str, Any]) -> None:
        """Update configuration for a specific detector."""
        if detector_name not in self.detectors:
            raise ValueError(f"Unknown detector: {detector_name}")
        
        detector = self.detectors[detector_name]
        if hasattr(detector, 'update_config'):
            detector.update_config(config)
            print(f"Updated configuration for {detector_name} detector")
        else:
            print(f"Detector {detector_name} does not support configuration updates")
    
    def restart_detector(self, detector_name: str) -> None:
        """Restart a specific detector."""
        if detector_name not in self.detectors:
            raise ValueError(f"Unknown detector: {detector_name}")
        
        detector = self.detectors[detector_name]
        
        try:
            if hasattr(detector, 'stop_detection'):
                detector.stop_detection()
            time.sleep(0.5)  # Brief pause
            
            if hasattr(detector, 'is_available') and detector.is_available():
                if hasattr(detector, 'start_detection'):
                    detector.start_detection()
                print(f"Restarted {detector_name} detector")
            else:
                print(f"Cannot restart {detector_name} detector - not available")
                
        except Exception as e:
            print(f"Error restarting {detector_name} detector: {e}")
    
    def get_detector_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics from all detectors that support it."""
        statistics = {}
        
        for name, detector in self.detectors.items():
            try:
                if hasattr(detector, 'get_detection_statistics'):
                    statistics[name] = detector.get_detection_statistics()
                elif hasattr(detector, 'get_statistics'):
                    statistics[name] = detector.get_statistics()
                elif hasattr(detector, 'get_movement_statistics'):
                    statistics[name] = detector.get_movement_statistics()
                else:
                    statistics[name] = {'status': 'no_statistics_available'}
            except Exception as e:
                statistics[name] = {'error': str(e)}
        
        return statistics
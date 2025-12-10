"""
System Audio Detector - Direct system audio capture and analysis.
Captures audio directly from system microphone and analyzes for speech, noise, and suspicious activity.
"""

import time
import threading
import queue
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import os
import wave
import json

# Audio processing imports with fallbacks
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("PyAudio not available - System audio detection disabled")

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("SpeechRecognition not available - Speech detection limited")

# Librosa is optional for advanced audio analysis
LIBROSA_AVAILABLE = False
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    pass  # Librosa not required for basic audio detection

try:
    from context_engine.interfaces import DetectionSource
    from context_engine.models import DetectionEvent, DetectionType
    from shared_utils.detection_utils import normalize_confidence
except ImportError:
    # Fallback classes for standalone operation
    class DetectionSource:
        def get_source_name(self) -> str:
            return "system_audio_detector"
    
    class DetectionEvent:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class DetectionType:
        SPEECH_DETECTED = "SPEECH_DETECTED"
        NOISE_DETECTED = "NOISE_DETECTED"
        MULTIPLE_VOICES = "MULTIPLE_VOICES"
        SUSPICIOUS_AUDIO = "SUSPICIOUS_AUDIO"
        AUDIO_ANOMALY = "AUDIO_ANOMALY"
    
    def normalize_confidence(conf):
        return max(0.0, min(1.0, float(conf)))


class SystemAudioDetector(DetectionSource):
    """
    System-level audio detector that captures and analyzes audio directly from microphone.
    Features:
    - Real-time audio capture from system microphone
    - Speech detection and transcription
    - Noise level analysis
    - Multiple voice detection
    - Suspicious audio pattern detection
    - Continuous audio recording for evidence
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize system audio detector."""
        self.config = config or {}
        
        # Audio configuration
        self.CHUNK_SIZE = self.config.get('chunk_size', 1024)
        self.SAMPLE_RATE = self.config.get('sample_rate', 44100)
        self.CHANNELS = self.config.get('channels', 1)
        self.FORMAT = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None
        
        # Detection thresholds
        self.SPEECH_THRESHOLD = self.config.get('speech_threshold', 0.3)
        self.NOISE_THRESHOLD = self.config.get('noise_threshold', 0.1)
        self.SILENCE_DURATION = self.config.get('silence_duration', 2.0)  # seconds
        
        # Audio processing
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.audio_thread = None
        self.analysis_thread = None
        
        # PyAudio setup
        self.pyaudio_instance = None
        self.audio_stream = None
        
        # Speech recognition
        self.recognizer = sr.Recognizer() if SPEECH_RECOGNITION_AVAILABLE else None
        
        # State tracking
        self.is_running = False
        self.total_detections = 0
        self.detection_counts = {
            'speech_detected': 0,
            'noise_detected': 0,
            'multiple_voices': 0,
            'suspicious_audio': 0,
            'audio_anomaly': 0
        }
        
        # Audio buffer for analysis
        self.audio_buffer = []
        self.buffer_duration = 5.0  # seconds
        self.max_buffer_size = int(self.SAMPLE_RATE * self.buffer_duration)
        
        # Recording for evidence
        self.recording_enabled = self.config.get('record_evidence', True)
        self.recording_buffer = []
        self.last_detection_time = None
        
        print("SystemAudioDetector initialized for direct system audio capture")
    
    def get_source_name(self) -> str:
        """Return the unique name of this detection source."""
        return "system_audio_detector"
    
    def is_available(self) -> bool:
        """Check if audio capture is available."""
        return PYAUDIO_AVAILABLE and self._test_audio_access()
    
    def _test_audio_access(self) -> bool:
        """Test if we can access the microphone."""
        if not PYAUDIO_AVAILABLE:
            return False
        
        try:
            p = pyaudio.PyAudio()
            # Try to get default input device
            device_info = p.get_default_input_device_info()
            p.terminate()
            return device_info is not None
        except Exception as e:
            print(f"Audio access test failed: {e}")
            return False
    
    def start_detection(self) -> None:
        """Start system audio detection."""
        try:
            if not self.is_available():
                raise Exception("Audio capture not available")
            
            self.is_running = True
            self.is_recording = True
            
            # Initialize PyAudio
            self.pyaudio_instance = pyaudio.PyAudio()
            
            # Get audio device info
            device_info = self.pyaudio_instance.get_default_input_device_info()
            print(f"Using audio device: {device_info['name']}")
            
            # Open audio stream
            self.audio_stream = self.pyaudio_instance.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE,
                stream_callback=self._audio_callback
            )
            
            # Start audio capture thread
            self.audio_thread = threading.Thread(target=self._audio_capture_loop, daemon=True)
            self.audio_thread.start()
            
            # Start analysis thread
            self.analysis_thread = threading.Thread(target=self._audio_analysis_loop, daemon=True)
            self.analysis_thread.start()
            
            # Start the stream
            self.audio_stream.start_stream()
            
            print("SystemAudioDetector started successfully")
            
        except Exception as e:
            print(f"Failed to start SystemAudioDetector: {e}")
            self.is_running = False
            raise
    
    def stop_detection(self) -> None:
        """Stop audio detection and cleanup resources."""
        try:
            self.is_running = False
            self.is_recording = False
            
            # Stop audio stream
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None
            
            # Terminate PyAudio
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
                self.pyaudio_instance = None
            
            # Wait for threads to finish
            if self.audio_thread and self.audio_thread.is_alive():
                self.audio_thread.join(timeout=2.0)
            
            if self.analysis_thread and self.analysis_thread.is_alive():
                self.analysis_thread.join(timeout=2.0)
            
            print("SystemAudioDetector stopped")
            
        except Exception as e:
            print(f"Error stopping SystemAudioDetector: {e}")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for audio data."""
        if self.is_recording:
            self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    def _audio_capture_loop(self):
        """Main audio capture loop."""
        while self.is_running:
            try:
                # Get audio data from queue
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get(timeout=0.1)
                    
                    # Convert to numpy array
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    
                    # Add to buffer
                    self.audio_buffer.extend(audio_array)
                    
                    # Maintain buffer size
                    if len(self.audio_buffer) > self.max_buffer_size:
                        self.audio_buffer = self.audio_buffer[-self.max_buffer_size:]
                    
                    # Add to recording buffer if recording enabled
                    if self.recording_enabled:
                        self.recording_buffer.extend(audio_array)
                        
                        # Limit recording buffer to prevent memory issues
                        max_recording_size = self.SAMPLE_RATE * 300  # 5 minutes
                        if len(self.recording_buffer) > max_recording_size:
                            self.recording_buffer = self.recording_buffer[-max_recording_size:]
                
                else:
                    time.sleep(0.01)  # Small delay if no data
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio capture loop: {e}")
                break
    
    def _audio_analysis_loop(self):
        """Main audio analysis loop."""
        last_analysis_time = time.time()
        analysis_interval = 1.0  # Analyze every second
        
        while self.is_running:
            try:
                current_time = time.time()
                
                if current_time - last_analysis_time >= analysis_interval:
                    if len(self.audio_buffer) > 0:
                        # Analyze current audio buffer
                        events = self._analyze_audio_buffer()
                        
                        # Process any detected events
                        for event in events:
                            self._handle_detection_event(event)
                    
                    last_analysis_time = current_time
                
                time.sleep(0.1)  # Small delay
                
            except Exception as e:
                print(f"Error in audio analysis loop: {e}")
                break
    
    def _analyze_audio_buffer(self) -> List[DetectionEvent]:
        """Analyze the current audio buffer for detections."""
        events = []
        
        if len(self.audio_buffer) == 0:
            return events
        
        # Convert to numpy array for analysis
        audio_data = np.array(self.audio_buffer, dtype=np.float32)
        
        # Normalize audio data
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # 1. Volume/Noise Level Analysis
        rms_level = np.sqrt(np.mean(audio_data ** 2))
        
        if rms_level > self.NOISE_THRESHOLD:
            self.detection_counts['noise_detected'] += 1
            
            # Determine if it's speech or just noise
            if rms_level > self.SPEECH_THRESHOLD:
                # Likely speech
                confidence = min(rms_level / self.SPEECH_THRESHOLD, 1.0)
                
                events.append(DetectionEvent(
                    event_type=DetectionType.SPEECH_DETECTED,
                    timestamp=datetime.now(),
                    confidence=normalize_confidence(confidence),
                    source=self.get_source_name(),
                    metadata={
                        'rms_level': float(rms_level),
                        'speech_threshold': self.SPEECH_THRESHOLD,
                        'detection_method': 'rms_analysis',
                        'description': f'Speech detected with RMS level: {rms_level:.3f}'
                    },
                    frame_data=None
                ))
                
                self.detection_counts['speech_detected'] += 1
            else:
                # Just noise
                events.append(DetectionEvent(
                    event_type=DetectionType.NOISE_DETECTED,
                    timestamp=datetime.now(),
                    confidence=normalize_confidence(rms_level / self.NOISE_THRESHOLD),
                    source=self.get_source_name(),
                    metadata={
                        'rms_level': float(rms_level),
                        'noise_threshold': self.NOISE_THRESHOLD,
                        'detection_method': 'rms_analysis',
                        'description': f'Noise detected with RMS level: {rms_level:.3f}'
                    },
                    frame_data=None
                ))
        
        # 2. Frequency Analysis (if librosa is available)
        if LIBROSA_AVAILABLE and len(audio_data) > 1024:
            try:
                # Spectral analysis
                spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.SAMPLE_RATE)[0]
                spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.SAMPLE_RATE)[0]
                
                # Check for suspicious audio patterns
                if np.mean(spectral_centroids) > 3000:  # High frequency content
                    self.detection_counts['suspicious_audio'] += 1
                    
                    events.append(DetectionEvent(
                        event_type=DetectionType.SUSPICIOUS_AUDIO,
                        timestamp=datetime.now(),
                        confidence=0.7,
                        source=self.get_source_name(),
                        metadata={
                            'spectral_centroid': float(np.mean(spectral_centroids)),
                            'spectral_rolloff': float(np.mean(spectral_rolloff)),
                            'detection_method': 'spectral_analysis',
                            'description': 'Suspicious high-frequency audio detected'
                        },
                        frame_data=None
                    ))
                
            except Exception as e:
                print(f"Spectral analysis error: {e}")
        
        # 3. Speech Recognition (if available and speech detected)
        if SPEECH_RECOGNITION_AVAILABLE and self.recognizer and rms_level > self.SPEECH_THRESHOLD:
            try:
                # Convert audio for speech recognition
                audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
                
                # Create AudioData object
                audio_sr = sr.AudioData(audio_bytes, self.SAMPLE_RATE, 2)
                
                # Try to recognize speech
                try:
                    text = self.recognizer.recognize_google(audio_sr, language='en-US')
                    if text and len(text.strip()) > 0:
                        # Speech successfully recognized
                        events.append(DetectionEvent(
                            event_type=DetectionType.SPEECH_DETECTED,
                            timestamp=datetime.now(),
                            confidence=0.9,
                            source=self.get_source_name(),
                            metadata={
                                'transcribed_text': text,
                                'text_length': len(text),
                                'detection_method': 'speech_recognition',
                                'description': f'Speech transcribed: "{text[:50]}..."'
                            },
                            frame_data=None
                        ))
                        
                        print(f"🎤 Speech detected: {text}")
                        
                except sr.UnknownValueError:
                    # Speech detected but not recognizable
                    pass
                except sr.RequestError as e:
                    print(f"Speech recognition error: {e}")
                    
            except Exception as e:
                print(f"Speech recognition processing error: {e}")
        
        return events
    
    def _handle_detection_event(self, event: DetectionEvent):
        """Handle a detection event."""
        self.total_detections += 1
        self.last_detection_time = datetime.now()
        
        # Save audio evidence if significant detection
        if event.confidence > 0.6 and self.recording_enabled:
            self._save_audio_evidence(event)
        
        print(f"🔊 Audio detection: {event.event_type} (confidence: {event.confidence:.2f})")
    
    def _save_audio_evidence(self, event: DetectionEvent):
        """Save audio evidence for a detection event."""
        try:
            if len(self.recording_buffer) == 0:
                return
            
            # Create evidence directory
            evidence_dir = "sessions/audio_evidence"
            os.makedirs(evidence_dir, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            event_type = str(event.event_type).replace('DetectionType.', '').lower()
            filename = f"audio_{event_type}_{timestamp}.wav"
            filepath = os.path.join(evidence_dir, filename)
            
            # Get recent audio (last 5 seconds)
            recent_samples = min(len(self.recording_buffer), self.SAMPLE_RATE * 5)
            audio_data = np.array(self.recording_buffer[-recent_samples:], dtype=np.int16)
            
            # Save as WAV file
            with wave.open(filepath, 'wb') as wav_file:
                wav_file.setnchannels(self.CHANNELS)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.SAMPLE_RATE)
                wav_file.writeframes(audio_data.tobytes())
            
            print(f"💾 Saved audio evidence: {filename} ({len(audio_data)} samples)")
            
        except Exception as e:
            print(f"Error saving audio evidence: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Return health status information."""
        return {
            'source_name': self.get_source_name(),
            'is_running': self.is_running,
            'is_available': self.is_available(),
            'pyaudio_available': PYAUDIO_AVAILABLE,
            'speech_recognition_available': SPEECH_RECOGNITION_AVAILABLE,
            'librosa_available': LIBROSA_AVAILABLE,
            'audio_stream_active': self.audio_stream is not None and self.audio_stream.is_active() if self.audio_stream else False,
            'buffer_size': len(self.audio_buffer),
            'recording_buffer_size': len(self.recording_buffer),
            'detection_counts': self.detection_counts.copy(),
            'total_detections': self.total_detections,
            'config': {
                'sample_rate': self.SAMPLE_RATE,
                'chunk_size': self.CHUNK_SIZE,
                'channels': self.CHANNELS,
                'speech_threshold': self.SPEECH_THRESHOLD,
                'noise_threshold': self.NOISE_THRESHOLD
            }
        }
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive detection statistics."""
        total_violations = sum(self.detection_counts.values())
        
        stats = {
            'total_detections': self.total_detections,
            'total_violations': total_violations,
            'violation_rate': total_violations / max(self.total_detections, 1),
            'detection_counts': self.detection_counts.copy(),
            'is_running': self.is_running,
            'last_detection': self.last_detection_time.isoformat() if self.last_detection_time else None
        }
        
        # Add percentage breakdown
        if total_violations > 0:
            stats['violation_percentages'] = {
                k: (v / total_violations) * 100 
                for k, v in self.detection_counts.items()
            }
        else:
            stats['violation_percentages'] = {k: 0.0 for k in self.detection_counts.keys()}
        
        return stats
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update detector configuration."""
        self.config.update(new_config)
        
        # Update thresholds
        self.SPEECH_THRESHOLD = self.config.get('speech_threshold', self.SPEECH_THRESHOLD)
        self.NOISE_THRESHOLD = self.config.get('noise_threshold', self.NOISE_THRESHOLD)
        self.SILENCE_DURATION = self.config.get('silence_duration', self.SILENCE_DURATION)
        
        print(f"SystemAudioDetector configuration updated: {new_config}")
    
    def reset_statistics(self) -> None:
        """Reset all detection statistics."""
        self.total_detections = 0
        self.detection_counts = {k: 0 for k in self.detection_counts.keys()}
        self.last_detection_time = None
        print("SystemAudioDetector statistics reset")


# Standalone execution for testing
if __name__ == "__main__":
    """
    Standalone test mode - run system audio detector directly.
    """
    import signal
    import sys
    
    def signal_handler(sig, frame):
        print("\nStopping audio detector...")
        detector.stop_detection()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    detector = SystemAudioDetector()
    
    if not detector.is_available():
        print("❌ Audio capture not available. Check microphone permissions.")
        sys.exit(1)
    
    print("🎤 Starting System Audio Detector - Speak to test detection")
    print("Press Ctrl+C to stop")
    
    try:
        detector.start_detection()
        
        # Keep running and show status
        while True:
            time.sleep(5)
            stats = detector.get_detection_statistics()
            print(f"📊 Detections: {stats['total_detections']} | "
                  f"Speech: {stats['detection_counts']['speech_detected']} | "
                  f"Noise: {stats['detection_counts']['noise_detected']}")
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        detector.stop_detection()
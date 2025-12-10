"""
Screen Capture Detector - Direct system screen capture for proctoring.
Captures screenshots of the entire screen or specific windows for cheating detection.
"""

import time
import threading
import queue
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import os
import io
from PIL import Image, ImageGrab
import base64

# Screen capture imports with fallbacks
try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
    # Disable pyautogui failsafe
    pyautogui.FAILSAFE = False
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    print("PyAutoGUI not available - Screen capture limited")

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    print("MSS not available - Using fallback screen capture")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available - Screen capture analysis limited")

try:
    from context_engine.interfaces import DetectionSource
    from context_engine.models import DetectionEvent, DetectionType
    from shared_utils.detection_utils import normalize_confidence
except ImportError:
    # Fallback classes for standalone operation
    class DetectionSource:
        def get_source_name(self) -> str:
            return "screen_capture_detector"
    
    class DetectionEvent:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class DetectionType:
        SCREEN_SHARING = "SCREEN_SHARING"
        MULTIPLE_MONITORS = "MULTIPLE_MONITORS"
        SUSPICIOUS_WINDOW = "SUSPICIOUS_WINDOW"
        BROWSER_TAB_SWITCH = "BROWSER_TAB_SWITCH"
        EXTERNAL_APPLICATION = "EXTERNAL_APPLICATION"
    
    def normalize_confidence(conf):
        return max(0.0, min(1.0, float(conf)))


class ScreenCaptureDetector(DetectionSource):
    """
    Screen capture detector that monitors the entire desktop for suspicious activity.
    Features:
    - Full screen capture at regular intervals
    - Window detection and analysis
    - Browser tab switching detection
    - Multiple monitor detection
    - Suspicious application detection
    - Screen recording for evidence
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize screen capture detector."""
        self.config = config or {}
        
        # Capture configuration
        self.CAPTURE_INTERVAL = self.config.get('capture_interval', 2.0)  # seconds
        self.SAVE_SCREENSHOTS = self.config.get('save_screenshots', True)
        self.ANALYZE_WINDOWS = self.config.get('analyze_windows', True)
        
        # Detection thresholds
        self.WINDOW_CHANGE_THRESHOLD = self.config.get('window_change_threshold', 0.3)
        self.SUSPICIOUS_APP_KEYWORDS = self.config.get('suspicious_apps', [
            'chrome', 'firefox', 'edge', 'safari', 'browser',
            'whatsapp', 'telegram', 'discord', 'slack', 'teams',
            'notepad', 'calculator', 'cmd', 'powershell', 'terminal'
        ])
        
        # Screen capture
        self.capture_queue = queue.Queue()
        self.is_capturing = False
        self.capture_thread = None
        self.analysis_thread = None
        
        # MSS setup for fast screen capture
        self.sct = None
        if MSS_AVAILABLE:
            try:
                self.sct = mss.mss()
                self.monitors = self.sct.monitors
                print(f"Detected {len(self.monitors)-1} monitors")
            except Exception as e:
                print(f"MSS initialization failed: {e}")
                self.sct = None
        
        # State tracking
        self.is_running = False
        self.total_captures = 0
        self.detection_counts = {
            'screen_sharing': 0,
            'multiple_monitors': 0,
            'suspicious_window': 0,
            'browser_tab_switch': 0,
            'external_application': 0
        }
        
        # Previous state for comparison
        self.previous_screenshot = None
        self.previous_windows = []
        self.last_active_window = None
        
        # Evidence storage
        self.evidence_dir = "sessions/screen_evidence"
        os.makedirs(self.evidence_dir, exist_ok=True)
        
        print("ScreenCaptureDetector initialized for system screen monitoring")
    
    def get_source_name(self) -> str:
        """Return the unique name of this detection source."""
        return "screen_capture_detector"
    
    def is_available(self) -> bool:
        """Check if screen capture is available."""
        return PYAUTOGUI_AVAILABLE or MSS_AVAILABLE or hasattr(ImageGrab, 'grab')
    
    def start_detection(self) -> None:
        """Start screen capture detection."""
        try:
            if not self.is_available():
                raise Exception("Screen capture not available")
            
            self.is_running = True
            self.is_capturing = True
            
            # Start capture thread
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            # Start analysis thread
            self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
            self.analysis_thread.start()
            
            print("ScreenCaptureDetector started successfully")
            
        except Exception as e:
            print(f"Failed to start ScreenCaptureDetector: {e}")
            self.is_running = False
            raise
    
    def stop_detection(self) -> None:
        """Stop screen capture and cleanup resources."""
        try:
            self.is_running = False
            self.is_capturing = False
            
            # Close MSS
            if self.sct:
                self.sct.close()
                self.sct = None
            
            # Wait for threads to finish
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2.0)
            
            if self.analysis_thread and self.analysis_thread.is_alive():
                self.analysis_thread.join(timeout=2.0)
            
            print("ScreenCaptureDetector stopped")
            
        except Exception as e:
            print(f"Error stopping ScreenCaptureDetector: {e}")
    
    def _capture_loop(self):
        """Main screen capture loop."""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Capture screenshot
                screenshot = self._capture_screenshot()
                
                if screenshot:
                    self.total_captures += 1
                    
                    # Add to queue for analysis
                    capture_data = {
                        'screenshot': screenshot,
                        'timestamp': datetime.now(),
                        'windows': self._get_window_list() if self.ANALYZE_WINDOWS else []
                    }
                    
                    self.capture_queue.put(capture_data)
                
                # Wait for next capture
                elapsed = time.time() - start_time
                sleep_time = max(0, self.CAPTURE_INTERVAL - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                print(f"Error in capture loop: {e}")
                time.sleep(1.0)
    
    def _analysis_loop(self):
        """Main analysis loop."""
        while self.is_running:
            try:
                # Get capture data from queue
                if not self.capture_queue.empty():
                    capture_data = self.capture_queue.get(timeout=0.1)
                    
                    # Analyze capture
                    events = self._analyze_capture(capture_data)
                    
                    # Process any detected events
                    for event in events:
                        self._handle_detection_event(event, capture_data)
                
                else:
                    time.sleep(0.1)  # Small delay if no data
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in analysis loop: {e}")
                break
    
    def _capture_screenshot(self) -> Optional[Image.Image]:
        """Capture a screenshot using the best available method."""
        try:
            # Method 1: MSS (fastest)
            if self.sct and MSS_AVAILABLE:
                # Capture primary monitor
                monitor = self.sct.monitors[1]  # monitors[0] is all monitors combined
                screenshot_data = self.sct.grab(monitor)
                
                # Convert to PIL Image
                screenshot = Image.frombytes('RGB', screenshot_data.size, screenshot_data.bgra, 'raw', 'BGRX')
                return screenshot
            
            # Method 2: PyAutoGUI
            elif PYAUTOGUI_AVAILABLE:
                screenshot = pyautogui.screenshot()
                return screenshot
            
            # Method 3: PIL ImageGrab (fallback)
            else:
                screenshot = ImageGrab.grab()
                return screenshot
                
        except Exception as e:
            print(f"Screenshot capture failed: {e}")
            return None
    
    def _get_window_list(self) -> List[Dict[str, Any]]:
        """Get list of currently open windows."""
        windows = []
        
        try:
            if PYAUTOGUI_AVAILABLE:
                # Get all windows (platform-specific implementation needed)
                # This is a simplified version
                import psutil
                
                for proc in psutil.process_iter(['pid', 'name', 'create_time']):
                    try:
                        proc_info = proc.info
                        if proc_info['name']:
                            windows.append({
                                'name': proc_info['name'],
                                'pid': proc_info['pid'],
                                'create_time': proc_info['create_time']
                            })
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                        
        except Exception as e:
            print(f"Window list error: {e}")
        
        return windows
    
    def _analyze_capture(self, capture_data: Dict[str, Any]) -> List[DetectionEvent]:
        """Analyze captured screen data for suspicious activity."""
        events = []
        screenshot = capture_data['screenshot']
        timestamp = capture_data['timestamp']
        windows = capture_data['windows']
        
        # 1. Multiple Monitor Detection
        if self.sct and len(self.monitors) > 2:  # More than 1 actual monitor
            self.detection_counts['multiple_monitors'] += 1
            
            events.append(DetectionEvent(
                event_type=DetectionType.MULTIPLE_MONITORS,
                timestamp=timestamp,
                confidence=0.9,
                source=self.get_source_name(),
                metadata={
                    'monitor_count': len(self.monitors) - 1,
                    'detection_method': 'mss_monitor_detection',
                    'description': f'{len(self.monitors)-1} monitors detected'
                },
                frame_data=None
            ))
        
        # 2. Window Analysis
        if windows and self.ANALYZE_WINDOWS:
            suspicious_windows = []
            
            for window in windows:
                window_name = window.get('name', '').lower()
                
                # Check for suspicious applications
                for keyword in self.SUSPICIOUS_APP_KEYWORDS:
                    if keyword in window_name:
                        suspicious_windows.append({
                            'window': window,
                            'keyword': keyword,
                            'suspicion_level': self._calculate_suspicion_level(keyword)
                        })
                        break
            
            if suspicious_windows:
                self.detection_counts['suspicious_window'] += 1
                
                # Get the most suspicious window
                most_suspicious = max(suspicious_windows, key=lambda x: x['suspicion_level'])
                
                events.append(DetectionEvent(
                    event_type=DetectionType.SUSPICIOUS_WINDOW,
                    timestamp=timestamp,
                    confidence=normalize_confidence(most_suspicious['suspicion_level']),
                    source=self.get_source_name(),
                    metadata={
                        'suspicious_windows': len(suspicious_windows),
                        'most_suspicious': most_suspicious['window']['name'],
                        'keyword_matched': most_suspicious['keyword'],
                        'detection_method': 'window_name_analysis',
                        'description': f'Suspicious application detected: {most_suspicious["window"]["name"]}'
                    },
                    frame_data=None
                ))
        
        # 3. Screen Change Detection
        if self.previous_screenshot and CV2_AVAILABLE:
            try:
                # Convert images to numpy arrays for comparison
                current_array = np.array(screenshot)
                previous_array = np.array(self.previous_screenshot)
                
                # Calculate difference
                if current_array.shape == previous_array.shape:
                    diff = np.mean(np.abs(current_array.astype(float) - previous_array.astype(float)))
                    normalized_diff = diff / 255.0
                    
                    # If significant change detected
                    if normalized_diff > self.WINDOW_CHANGE_THRESHOLD:
                        self.detection_counts['browser_tab_switch'] += 1
                        
                        events.append(DetectionEvent(
                            event_type=DetectionType.BROWSER_TAB_SWITCH,
                            timestamp=timestamp,
                            confidence=normalize_confidence(min(normalized_diff * 2, 1.0)),
                            source=self.get_source_name(),
                            metadata={
                                'screen_change_percentage': float(normalized_diff * 100),
                                'change_threshold': self.WINDOW_CHANGE_THRESHOLD,
                                'detection_method': 'screen_diff_analysis',
                                'description': f'Significant screen change detected: {normalized_diff*100:.1f}%'
                            },
                            frame_data=None
                        ))
                        
            except Exception as e:
                print(f"Screen comparison error: {e}")
        
        # Update previous state
        self.previous_screenshot = screenshot
        self.previous_windows = windows
        
        return events
    
    def _calculate_suspicion_level(self, keyword: str) -> float:
        """Calculate suspicion level for a keyword."""
        high_risk = ['chrome', 'firefox', 'whatsapp', 'telegram', 'discord']
        medium_risk = ['notepad', 'calculator', 'teams', 'slack']
        
        if keyword in high_risk:
            return 0.9
        elif keyword in medium_risk:
            return 0.6
        else:
            return 0.4
    
    def _handle_detection_event(self, event: DetectionEvent, capture_data: Dict[str, Any]):
        """Handle a detection event."""
        # Save screenshot evidence if significant detection
        if event.confidence > 0.6 and self.SAVE_SCREENSHOTS:
            self._save_screenshot_evidence(event, capture_data['screenshot'])
        
        print(f"🖥️ Screen detection: {event.event_type} (confidence: {event.confidence:.2f})")
    
    def _save_screenshot_evidence(self, event: DetectionEvent, screenshot: Image.Image):
        """Save screenshot evidence for a detection event."""
        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            event_type = str(event.event_type).replace('DetectionType.', '').lower()
            filename = f"screen_{event_type}_{timestamp}.jpg"
            filepath = os.path.join(self.evidence_dir, filename)
            
            # Save screenshot
            screenshot.save(filepath, 'JPEG', quality=85)
            
            print(f"💾 Saved screen evidence: {filename}")
            
        except Exception as e:
            print(f"Error saving screen evidence: {e}")
    
    def get_screenshot_base64(self) -> Optional[str]:
        """Get current screenshot as base64 string."""
        try:
            screenshot = self._capture_screenshot()
            if screenshot:
                # Convert to base64
                buffer = io.BytesIO()
                screenshot.save(buffer, format='JPEG', quality=85)
                img_str = base64.b64encode(buffer.getvalue()).decode()
                return f"data:image/jpeg;base64,{img_str}"
            
        except Exception as e:
            print(f"Error getting screenshot base64: {e}")
        
        return None
    
    def get_health_status(self) -> Dict[str, Any]:
        """Return health status information."""
        return {
            'source_name': self.get_source_name(),
            'is_running': self.is_running,
            'is_available': self.is_available(),
            'pyautogui_available': PYAUTOGUI_AVAILABLE,
            'mss_available': MSS_AVAILABLE,
            'cv2_available': CV2_AVAILABLE,
            'monitor_count': len(self.monitors) - 1 if self.sct else 1,
            'total_captures': self.total_captures,
            'detection_counts': self.detection_counts.copy(),
            'config': {
                'capture_interval': self.CAPTURE_INTERVAL,
                'save_screenshots': self.SAVE_SCREENSHOTS,
                'analyze_windows': self.ANALYZE_WINDOWS,
                'window_change_threshold': self.WINDOW_CHANGE_THRESHOLD
            }
        }
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive detection statistics."""
        total_violations = sum(self.detection_counts.values())
        
        stats = {
            'total_captures': self.total_captures,
            'total_violations': total_violations,
            'violation_rate': total_violations / max(self.total_captures, 1),
            'detection_counts': self.detection_counts.copy(),
            'is_running': self.is_running
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
        
        # Update settings
        self.CAPTURE_INTERVAL = self.config.get('capture_interval', self.CAPTURE_INTERVAL)
        self.SAVE_SCREENSHOTS = self.config.get('save_screenshots', self.SAVE_SCREENSHOTS)
        self.ANALYZE_WINDOWS = self.config.get('analyze_windows', self.ANALYZE_WINDOWS)
        self.WINDOW_CHANGE_THRESHOLD = self.config.get('window_change_threshold', self.WINDOW_CHANGE_THRESHOLD)
        
        print(f"ScreenCaptureDetector configuration updated: {new_config}")
    
    def reset_statistics(self) -> None:
        """Reset all detection statistics."""
        self.total_captures = 0
        self.detection_counts = {k: 0 for k in self.detection_counts.keys()}
        print("ScreenCaptureDetector statistics reset")


# Standalone execution for testing
if __name__ == "__main__":
    """
    Standalone test mode - run screen capture detector directly.
    """
    import signal
    import sys
    
    def signal_handler(sig, frame):
        print("\nStopping screen capture detector...")
        detector.stop_detection()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    detector = ScreenCaptureDetector()
    
    if not detector.is_available():
        print("❌ Screen capture not available.")
        sys.exit(1)
    
    print("🖥️ Starting Screen Capture Detector")
    print("Press Ctrl+C to stop")
    
    try:
        detector.start_detection()
        
        # Keep running and show status
        while True:
            time.sleep(5)
            stats = detector.get_detection_statistics()
            print(f"📊 Captures: {stats['total_captures']} | "
                  f"Violations: {stats['total_violations']} | "
                  f"Suspicious Windows: {stats['detection_counts']['suspicious_window']}")
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        detector.stop_detection()
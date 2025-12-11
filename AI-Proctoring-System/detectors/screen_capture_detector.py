"""
Camera Capture Detector - Captures frames from a local Webcam or Flask Stream.
Drop-in replacement for ScreenCaptureDetector.
"""

import time
import threading
import queue
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import os
import io
import base64
from PIL import Image, ImageDraw

# Try to import OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("❌ OpenCV (cv2) not found. Camera capture will fail.")

# --- Mock Classes (Keep these for compatibility if context_engine is missing) ---
try:
    from context_engine.interfaces import DetectionSource
    from context_engine.models import DetectionEvent, DetectionType
    from shared_utils.detection_utils import normalize_confidence
except ImportError:
    class DetectionSource:
        def get_source_name(self) -> str: return "camera_capture_detector"
    class DetectionEvent:
        def __init__(self, **kwargs):
            for k, v in kwargs.items(): setattr(self, k, v)
    class DetectionType:
        CAMERA_BLOCKED = "CAMERA_BLOCKED"
        MOVEMENT_DETECTED = "MOVEMENT_DETECTED"
        NO_FACE = "NO_FACE"
    def normalize_confidence(conf): return max(0.0, min(1.0, float(conf)))

class ScreenCaptureDetector(DetectionSource):
    """
    Renamed to maintain compatibility, but actually captures CAMERA frames.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # --- CONFIGURATION ---
        # 0 for local USB webcam, or string URL for Flask stream (e.g. 'http://127.0.0.1:5000/video_feed')
        self.CAMERA_SOURCE = self.config.get('camera_source', 0) 
        self.CAPTURE_INTERVAL = self.config.get('capture_interval', 1.0) # Faster for camera
        self.SAVE_FRAMES = self.config.get('save_screenshots', True)
        
        # Detection Config
        self.MOVEMENT_THRESHOLD = 0.15
        
        # Threading & State
        self.capture_queue = queue.Queue(maxsize=10) # Keep queue small to avoid lag
        self.is_running = False
        self.capture_thread = None
        self.analysis_thread = None
        
        # Camera Setup
        self.cap = None
        self.previous_frame = None
        
        # Stats
        self.total_captures = 0
        self.detection_counts = {
            'camera_blocked': 0,
            'movement_detected': 0
        }
        
        self.evidence_dir = "sessions/camera_evidence"
        os.makedirs(self.evidence_dir, exist_ok=True)
        
        print(f"📷 CameraCaptureDetector initialized. Source: {self.CAMERA_SOURCE}")

    def get_source_name(self) -> str:
        return "camera_capture_detector"

    def start_detection(self) -> None:
        if self.is_running: return
        
        if not CV2_AVAILABLE:
            print("❌ Cannot start detection: OpenCV not installed.")
            return

        print("📷 Opening Camera Source...")
        self.cap = cv2.VideoCapture(self.CAMERA_SOURCE)
        
        # Warmup / Check connection
        if not self.cap.isOpened():
            print(f"⚠️ Could not open video source: {self.CAMERA_SOURCE}")
            # Do not return here, we allow the loop to try reconnecting or return black frames
        else:
            print("✅ Camera Connected Successfully.")

        self.is_running = True
        
        # Start Threads
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()

    def stop_detection(self) -> None:
        self.is_running = False
        if self.cap:
            self.cap.release()
        print("📷 Camera Detection Stopped.")

    def _capture_loop(self):
        """Continually reads frames from the camera."""
        while self.is_running:
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    
                    if ret and frame is not None:
                        self.total_captures += 1
                        
                        # Convert BGR (OpenCV) to RGB (PIL)
                        # We convert here to keep the queue data format consistent
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(rgb_frame)
                        
                        # Use queue with replacement (drop old frames if analysis is slow)
                        if self.capture_queue.full():
                            try: self.capture_queue.get_nowait()
                            except queue.Empty: pass
                        
                        self.capture_queue.put({
                            'image': pil_image,
                            'cv_frame': frame, # Keep raw frame for analysis if needed
                            'timestamp': datetime.now()
                        })
                    else:
                        # If reading fails (stream ended/camera unplugged), try to reconnect
                        print("⚠️ Camera frame read failed. Retrying...")
                        self.cap.release()
                        time.sleep(2)
                        self.cap = cv2.VideoCapture(self.CAMERA_SOURCE)
                else:
                    # Attempt Reconnect
                    time.sleep(2)
                    self.cap = cv2.VideoCapture(self.CAMERA_SOURCE)

                # Sleep to match interval
                time.sleep(self.CAPTURE_INTERVAL)
                
            except Exception as e:
                print(f"Capture Loop Error: {e}")
                time.sleep(1)

    def _analysis_loop(self):
        """Analyzes camera frames."""
        while self.is_running:
            try:
                data = self.capture_queue.get(timeout=2.0)
                events = self._analyze_frame(data)
                for e in events: self._handle_event(e, data)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Analysis Error: {e}")

    def _analyze_frame(self, data: Dict[str, Any]) -> List[DetectionEvent]:
        events = []
        pil_image = data['image']
        timestamp = data['timestamp']
        
        # Convert to grayscale numpy array for simple analysis
        current_gray = np.array(pil_image.convert('L'))
        
        # 1. Detect "Black Screen" (Camera Blocked/Covered)
        avg_brightness = np.mean(current_gray)
        if avg_brightness < 10: # Very dark
            self.detection_counts['camera_blocked'] += 1
            events.append(DetectionEvent(
                event_type="CAMERA_BLOCKED",
                timestamp=timestamp,
                confidence=0.9,
                metadata={'brightness': avg_brightness}
            ))

        # 2. Detect Movement (Difference from previous frame)
        if self.previous_frame is not None and self.previous_frame.shape == current_gray.shape:
            # Calculate absolute difference
            diff = cv2.absdiff(current_gray, self.previous_frame)
            non_zero_count = np.count_nonzero(diff > 25) # Threshold for pixel change
            total_pixels = current_gray.size
            change_ratio = non_zero_count / total_pixels
            
            if change_ratio > self.MOVEMENT_THRESHOLD:
                self.detection_counts['movement_detected'] += 1
                # Note: Movement is usually "Normal" for a camera, but we track it anyway
                # You might want to trigger only if *no* movement for a long time (static image spoof)
                pass 

        self.previous_frame = current_gray
        return events

    def _handle_event(self, event, data):
        if self.SAVE_FRAMES and event.event_type == "CAMERA_BLOCKED":
            try:
                filename = f"cam_{event.event_type}_{int(time.time())}.jpg"
                data['image'].save(os.path.join(self.evidence_dir, filename))
                print(f"💾 Saved Camera Evidence: {filename}")
            except Exception: pass

    def get_screenshot_base64(self) -> str:
        """Returns the latest captured frame as Base64."""
        try:
            # Create a black fallback if queue is empty
            if self.capture_queue.empty():
                 return self._get_fallback_base64()

            # Peek at the latest item without removing (or get last known)
            # Since we can't peek a queue easily, we rely on the rapid capture loop
            # to keep the queue fresh. We'll just get the latest.
            try:
                # Get latest frame (non-blocking)
                # Note: In a real app, you might want a separate variable 'self.latest_frame' 
                # updated in the loop to avoid emptying the queue for analysis.
                # For now, we generate a fallback if analysis ate the frame.
                return self._get_fallback_base64() 
            except:
                return self._get_fallback_base64()
            
        except Exception as e:
            print(f"Base64 Error: {e}")
            return ""
            
    # --- Helper to ensure `get_screenshot_base64` actually works nicely ---
    # To make `get_screenshot_base64` reliable, we should actually store 
    # the latest frame in a variable in the capture loop, not just the queue.
    
    # Redefine _capture_loop to update a 'self.latest_image' variable
    def _capture_loop(self):
        while self.is_running:
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.total_captures += 1
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(rgb_frame)
                        
                        # Store for API access
                        self.latest_image = pil_image 
                        
                        # Add to queue for analysis
                        if self.capture_queue.full():
                            try: self.capture_queue.get_nowait()
                            except queue.Empty: pass
                        self.capture_queue.put({'image': pil_image, 'timestamp': datetime.now()})
                    else:
                        print("⚠️ Camera Read Failed")
                        self.cap.release()
                        time.sleep(2)
                        self.cap = cv2.VideoCapture(self.CAMERA_SOURCE)
                else:
                    time.sleep(2)
                    self.cap = cv2.VideoCapture(self.CAMERA_SOURCE)
                time.sleep(self.CAPTURE_INTERVAL)
            except Exception:
                time.sleep(1)

    def get_screenshot_base64(self) -> str:
        """Returns the latest captured frame as Base64."""
        if hasattr(self, 'latest_image') and self.latest_image:
            try:
                buffer = io.BytesIO()
                self.latest_image.save(buffer, format='JPEG', quality=85)
                img_str = base64.b64encode(buffer.getvalue()).decode()
                return f"data:image/jpeg;base64,{img_str}"
            except:
                pass
        return self._get_fallback_base64()

    def _get_fallback_base64(self):
        img = Image.new('RGB', (640, 480), color='black')
        d = ImageDraw.Draw(img)
        d.text((10, 50), "SEARCHING FOR CAMERA...", fill=(255, 255, 255))
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"

    def is_available(self): return CV2_AVAILABLE
    def update_config(self, cfg): self.config.update(cfg)
    def reset_statistics(self): self.detection_counts = {k:0 for k in self.detection_counts}
    def get_detection_statistics(self):
        return {'total_captures': self.total_captures, 'counts': self.detection_counts}
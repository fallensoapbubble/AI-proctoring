"""
FaceGuard Detector - Advanced face analysis with multiple detection capabilities.
Integrates comprehensive face monitoring including gaze, spoof detection, and lip movement.
"""

import time
from typing import Dict, Any, Optional, List
from datetime import datetime

# Optional imports with fallbacks
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available - FaceGuard will use fallback mode")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available - FaceGuard will use fallback mode")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("NumPy not available - FaceGuard will use fallback mode")

try:
    from context_engine.interfaces import DetectionSource
    from context_engine.models import DetectionEvent, DetectionType
    from shared_utils.detection_utils import normalize_confidence
except ImportError:
    # Fallback classes for standalone operation
    class DetectionSource:
        def get_source_name(self) -> str:
            return "face_guard_detector"
    
    class DetectionEvent:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class DetectionType:
        GAZE_AWAY = "GAZE_AWAY"
        SPOOF_DETECTED = "SPOOF_DETECTED"
        MULTIPLE_FACES = "MULTIPLE_FACES"
        SPEECH_DETECTED = "SPEECH_DETECTED"
        NO_FACE = "NO_FACE"
    
    def normalize_confidence(conf):
        return max(0.0, min(1.0, float(conf)))


class FaceGuardDetector(DetectionSource):
    """
    Advanced face analysis detector combining multiple detection methods:
    - Gaze direction tracking
    - Spoof detection (anti-spoofing)
    - Multiple face detection
    - Lip movement analysis
    - Face presence monitoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize FaceGuard detector with MediaPipe Face Mesh."""
        self.config = config or {}
        
        # Initialize MediaPipe Face Mesh if available
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                max_num_faces=5  # Allow tracking multiple to detect violations
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        else:
            self.mp_face_mesh = None
            self.face_mesh = None
            self.mp_drawing = None
            self.drawing_spec = None
        
        # Detection thresholds
        self.SPOOF_THRESH = self.config.get('spoof_threshold', 100)  # Lower variance = likely spoof
        self.LIP_THRESH = self.config.get('lip_threshold', 8)        # Distance between lips
        self.LOOK_AWAY_X_THRESH = self.config.get('yaw_threshold', 10)   # Yaw angle threshold
        self.LOOK_AWAY_Y_THRESH = self.config.get('pitch_threshold', 10) # Pitch angle threshold
        
        # State tracking
        self.is_running = False
        self.total_detections = 0
        self.detection_counts = {
            'gaze_away': 0,
            'spoof_detected': 0,
            'multiple_faces': 0,
            'speech_detected': 0,
            'no_face': 0
        }
        
        print("FaceGuardDetector initialized with comprehensive face analysis")
    
    def get_source_name(self) -> str:
        """Return the unique name of this detection source."""
        return "face_guard_detector"
    
    def is_available(self) -> bool:
        """Check if MediaPipe and required dependencies are available."""
        return MEDIAPIPE_AVAILABLE and CV2_AVAILABLE and NUMPY_AVAILABLE
    
    def start_detection(self) -> None:
        """Start the FaceGuard detection system."""
        try:
            self.is_running = True
            self.total_detections = 0
            self.detection_counts = {k: 0 for k in self.detection_counts.keys()}
            print("FaceGuardDetector started successfully")
        except Exception as e:
            print(f"Failed to start FaceGuardDetector: {e}")
            self.is_running = False
            raise
    
    def stop_detection(self) -> None:
        """Stop detection and cleanup resources."""
        try:
            if self.face_mesh:
                self.face_mesh.close()
            self.is_running = False
            print("FaceGuardDetector stopped")
        except Exception as e:
            print(f"Error stopping FaceGuardDetector: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Return health status information."""
        return {
            'source_name': self.get_source_name(),
            'is_running': self.is_running,
            'is_available': self.is_available(),
            'face_mesh_initialized': self.face_mesh is not None,
            'thresholds': {
                'spoof_threshold': self.SPOOF_THRESH,
                'lip_threshold': self.LIP_THRESH,
                'yaw_threshold': self.LOOK_AWAY_X_THRESH,
                'pitch_threshold': self.LOOK_AWAY_Y_THRESH
            },
            'detection_counts': self.detection_counts.copy(),
            'total_detections': self.total_detections
        }
    
    def process_frame(self, frame) -> List[DetectionEvent]:
        """
        Process a frame and return all detected events.
        Returns a list of DetectionEvent objects for different types of violations.
        """
        if not self.is_running:
            return []
        
        self.total_detections += 1
        events = []
        
        # Check if dependencies are available
        if not self.is_available():
            # Return a fallback detection event
            events.append(DetectionEvent(
                event_type=DetectionType.NO_FACE,
                timestamp=datetime.now(),
                confidence=0.5,
                source=self.get_source_name(),
                metadata={
                    'detection_method': 'fallback',
                    'reason': 'dependencies_not_available',
                    'description': 'OpenCV/MediaPipe not available'
                },
                frame_data=None
            ))
            return events
        
        try:
            img_h, img_w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_count = len(results.multi_face_landmarks)
                
                # 1. Multiple People Detection
                if face_count > 1:
                    self.detection_counts['multiple_faces'] += 1
                    events.append(DetectionEvent(
                        event_type=DetectionType.MULTIPLE_FACES,
                        timestamp=datetime.now(),
                        confidence=normalize_confidence(min(face_count / 2.0, 1.0)),
                        source=self.get_source_name(),
                        metadata={
                            'face_count': face_count,
                            'detection_method': 'mediapipe_face_mesh',
                            'description': f'{face_count} faces detected'
                        },
                        frame_data=frame.copy()
                    ))
                
                # Process the primary face (first detected)
                primary_face = results.multi_face_landmarks[0]
                
                # Get Bounding Box for Spoof Check
                x_vals = [lm.x for lm in primary_face.landmark]
                y_vals = [lm.y for lm in primary_face.landmark]
                bbox = (
                    int(min(x_vals) * img_w), 
                    int(min(y_vals) * img_h),
                    int((max(x_vals) - min(x_vals)) * img_w), 
                    int((max(y_vals) - min(y_vals)) * img_h)
                )
                
                # 2. Spoof Detection
                spoof_event = self.check_spoof(frame, bbox)
                if spoof_event:
                    events.append(spoof_event)
                
                # 3. Head Pose (Looking Away)
                gaze_event = self.check_gaze_direction(frame, primary_face)
                if gaze_event:
                    events.append(gaze_event)
                
                # 4. Lip Movement (Speech Detection)
                speech_event = self.check_lip_movement(primary_face, img_h, img_w)
                if speech_event:
                    events.append(speech_event)
                
            else:
                # No face detected
                self.detection_counts['no_face'] += 1
                events.append(DetectionEvent(
                    event_type=DetectionType.NO_FACE,
                    timestamp=datetime.now(),
                    confidence=0.9,
                    source=self.get_source_name(),
                    metadata={
                        'detection_method': 'mediapipe_face_mesh',
                        'description': 'No face detected in frame'
                    },
                    frame_data=frame.copy()
                ))
            
            return events
            
        except Exception as e:
            print(f"Error in FaceGuard frame processing: {e}")
            return []
    
    def get_head_pose(self, image, landmarks) -> tuple:
        """Estimate head pose (Yaw, Pitch, Roll) using SolvePnP."""
        img_h, img_w, _ = image.shape
        face_3d = []
        face_2d = []
        
        # Key landmarks for pose estimation
        key_points = [1, 199, 33, 263, 61, 291]  # Nose, Chin, Eyes, Mouth corners
        
        for idx, lm in enumerate(landmarks.landmark):
            if idx in key_points:
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])
        
        if not NUMPY_AVAILABLE:
            return False, "NumPy not available"
            
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)
        
        # Camera matrix approximation
        focal_length = 1 * img_w
        cam_matrix = np.array([
            [focal_length, 0, img_h / 2],
            [0, focal_length, img_w / 2],
            [0, 0, 1]
        ])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        
        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        
        if not success:
            return False, "Pose estimation failed"
        
        rmat, jac = cv2.Rodrigues(rot_vec)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        
        # Angles in degrees
        x = angles[0] * 360
        y = angles[1] * 360
        
        is_looking_away = False
        if abs(y) > self.LOOK_AWAY_X_THRESH or abs(x) > self.LOOK_AWAY_Y_THRESH:
            is_looking_away = True
        
        return is_looking_away, f"X: {int(x)}, Y: {int(y)}"
    
    def check_spoof(self, image, face_bbox: tuple) -> Optional[DetectionEvent]:
        """
        Basic Spoof Detection using Laplacian Variance (Blur Check).
        Real faces usually have high texture/sharpness. Screens/photos often blur.
        """
        x, y, w, h = face_bbox
        
        # Ensure bbox is within bounds
        x, y = max(0, x), max(0, y)
        x2, y2 = min(image.shape[1], x + w), min(image.shape[0], y + h)
        
        if x >= x2 or y >= y2:
            return None
        
        face_roi = image[y:y2, x:x2]
        if face_roi.size == 0:
            return None
        
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        
        # If variance is too low, it might be a flat screen or photo
        is_spoof = laplacian_var < self.SPOOF_THRESH
        
        if is_spoof:
            self.detection_counts['spoof_detected'] += 1
            
            # Calculate confidence based on how far below threshold
            confidence_factor = max(0.0, (self.SPOOF_THRESH - laplacian_var) / self.SPOOF_THRESH)
            confidence = normalize_confidence(0.5 + confidence_factor * 0.5)
            
            return DetectionEvent(
                event_type=DetectionType.SPOOF_DETECTED,
                timestamp=datetime.now(),
                confidence=confidence,
                source=self.get_source_name(),
                metadata={
                    'laplacian_variance': float(laplacian_var),
                    'spoof_threshold': self.SPOOF_THRESH,
                    'face_bbox': [x, y, w, h],
                    'detection_method': 'laplacian_variance',
                    'description': f'Low texture variance detected: {laplacian_var:.2f}'
                },
                frame_data=image.copy()
            )
        
        return None
    
    def check_gaze_direction(self, image, landmarks) -> Optional[DetectionEvent]:
        """Check if person is looking away from the screen."""
        is_looking_away, pose_text = self.get_head_pose(image, landmarks)
        
        if is_looking_away:
            self.detection_counts['gaze_away'] += 1
            
            # Extract angles from pose_text for confidence calculation
            try:
                x_str = pose_text.split("X: ")[1].split(",")[0]
                y_str = pose_text.split("Y: ")[1]
                x_angle = abs(int(x_str))
                y_angle = abs(int(y_str))
                
                # Calculate confidence based on angle deviation
                x_factor = min(x_angle / self.LOOK_AWAY_Y_THRESH, 1.0)
                y_factor = min(y_angle / self.LOOK_AWAY_X_THRESH, 1.0)
                confidence = normalize_confidence(max(x_factor, y_factor))
                
            except (ValueError, IndexError):
                confidence = 0.7  # Default confidence
            
            return DetectionEvent(
                event_type=DetectionType.GAZE_AWAY,
                timestamp=datetime.now(),
                confidence=confidence,
                source=self.get_source_name(),
                metadata={
                    'head_pose': pose_text,
                    'yaw_threshold': self.LOOK_AWAY_X_THRESH,
                    'pitch_threshold': self.LOOK_AWAY_Y_THRESH,
                    'detection_method': 'solvepnp_head_pose',
                    'description': f'Head pose indicates looking away: {pose_text}'
                },
                frame_data=image.copy()
            )
        
        return None
    
    def get_lip_movement(self, landmarks, img_h: int, img_w: int) -> tuple:
        """Check vertical distance between upper (13) and lower (14) lip."""
        upper_lip = landmarks.landmark[13]
        lower_lip = landmarks.landmark[14]
        
        # Calculate vertical distance in pixels
        distance = abs(upper_lip.y - lower_lip.y) * img_h
        is_speaking = distance > self.LIP_THRESH
        
        return is_speaking, distance
    
    def check_lip_movement(self, landmarks, img_h: int, img_w: int) -> Optional[DetectionEvent]:
        """Detect speech through lip movement analysis."""
        is_speaking, lip_distance = self.get_lip_movement(landmarks, img_h, img_w)
        
        if is_speaking:
            self.detection_counts['speech_detected'] += 1
            
            # Calculate confidence based on lip distance
            distance_factor = min((lip_distance - self.LIP_THRESH) / self.LIP_THRESH, 1.0)
            confidence = normalize_confidence(0.6 + distance_factor * 0.4)
            
            return DetectionEvent(
                event_type=DetectionType.SPEECH_DETECTED,
                timestamp=datetime.now(),
                confidence=confidence,
                source=self.get_source_name(),
                metadata={
                    'lip_distance': float(lip_distance),
                    'lip_threshold': self.LIP_THRESH,
                    'detection_method': 'lip_landmark_distance',
                    'description': f'Lip movement detected: {lip_distance:.2f}px'
                },
                frame_data=None  # Don't store frame for speech events to save space
            )
        
        return None
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update detector configuration."""
        self.config.update(new_config)
        
        # Update thresholds
        self.SPOOF_THRESH = self.config.get('spoof_threshold', self.SPOOF_THRESH)
        self.LIP_THRESH = self.config.get('lip_threshold', self.LIP_THRESH)
        self.LOOK_AWAY_X_THRESH = self.config.get('yaw_threshold', self.LOOK_AWAY_X_THRESH)
        self.LOOK_AWAY_Y_THRESH = self.config.get('pitch_threshold', self.LOOK_AWAY_Y_THRESH)
        
        print(f"FaceGuardDetector configuration updated: {new_config}")
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive detection statistics."""
        total_violations = sum(self.detection_counts.values())
        
        stats = {
            'total_detections': self.total_detections,
            'total_violations': total_violations,
            'violation_rate': total_violations / max(self.total_detections, 1),
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
    
    def reset_statistics(self) -> None:
        """Reset all detection statistics."""
        self.total_detections = 0
        self.detection_counts = {k: 0 for k in self.detection_counts.keys()}
        print("FaceGuardDetector statistics reset")


# Standalone execution for testing
if __name__ == "__main__":
    """
    Standalone test mode - run FaceGuard detector directly with camera.
    """
    cap = cv2.VideoCapture(0)
    guard = FaceGuardDetector()
    guard.start_detection()
    
    print("FaceGuard System Running - Press 'q' to quit")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame and get all events
        events = guard.process_frame(frame)
        
        # Display frame with annotations
        display_frame = frame.copy()
        
        # Draw detection results
        if events:
            y_offset = 30
            for event in events:
                event_type = str(event.event_type)
                confidence = event.confidence
                color = (0, 0, 255) if confidence > 0.7 else (0, 165, 255)
                
                text = f"{event_type}: {confidence:.2f}"
                cv2.putText(display_frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y_offset += 30
        else:
            cv2.putText(display_frame, "All Clear", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show statistics
        stats = guard.get_detection_statistics()
        stats_text = f"Detections: {stats['total_detections']} | Violations: {stats['total_violations']}"
        cv2.putText(display_frame, stats_text, (10, display_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('FaceGuard System', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    guard.stop_detection()
    cap.release()
    cv2.destroyAllWindows()
# AI Proctoring System - Detector Components

This directory contains all the detection components for the AI proctoring system. Each detector is designed as a modular, testable component that implements the `DetectionSource` interface.

## Overview

The detector system consists of:

1. **Individual Detectors** - Specialized detection components
2. **Detector Manager** - Coordinates all detectors and manages their

**Technology**: MediaPipe Face Mesh + Head Pose Estimation

**Detection Method**:
- Calculates yaw, pitch, and roll angles from facial landmarks
- Triggers alert when angles exceed thresholds
- Uses PnP (Perspective-n-Point) algorithm for 3D pose estimation

**Configuration**:
```python
{
    'yaw_threshold': 25,      # Degrees left/right
    'pitch_threshold': 20     # Degrees up/down
}
```

**Returns**: `DetectionType.GAZE_AWAY`

---

### 2. LipMovementDetector (`lip_movement_detector.py`)
**Purpose**: Detects lip movement indicating speech

**Technology**: MediaPipe Face Mesh

**Detection Method**:
- Tracks distance between upper and lower lip landmarks
- Calculates movement as change over time
- Uses smoothing window for stability

**Configuration**:
```python
{
    'movement_threshold': 0.003,  # Minimum movement to detect
    'smoothing_window': 3         # Frames for smoothing
}
```

**Returns**: `DetectionType.LIP_MOVEMENT`

---

### 3. MobileDetector (`mobile_detector.py`)
**Purpose**: Detects mobile phones and electronic devices

**Technology**: YOLO (YOLOv8)

**Detection Method**:
- Uses YOLO object detection for cell phones (COCO class 67)
- Maintains detection history for stability
- Filters by confidence threshold

**Configuration**:
```python
{
    'confidence_threshold': 0.6,
    'model_path': 'yolov8s.pt',
    'mobile_class_ids': [67],     # COCO class IDs
    'history_size': 5
}
```

**Returns**: `DetectionType.MOBILE_DETECTED`

---

### 4. PersonDetector (`person_detector.py`)
**Purpose**: Detects multiple people in frame

**Technology**: YOLO (YOLOv8)

**Detection Method**:
- Detects persons using YOLO (COCO class 0)
- Compares count against expected number
- Uses detection history for stability

**Configuration**:
```python
{
    'confidence_threshold': 0.5,
    'max_expected_people': 1,
    'model_path': 'yolov8n.pt',
    'history_size': 5
}
```

**Returns**: `DetectionType.MULTIPLE_PEOPLE`

---

### 5. FaceSpoofDetector (`face_spoof_detector.py`)
**Purpose**: Detects face spoofing attempts (photos, screens)

**Technology**: OpenCV + Computer Vision Techniques

**Detection Method**:
- Texture analysis using Local Binary Patterns (LBP)
- Motion analysis between frames
- Brightness variance analysis
- Edge density analysis
- Requires 2+ indicators to trigger

**Configuration**:
```python
{
    'texture_threshold': 0.3,
    'motion_threshold': 0.1,
    'brightness_variance_threshold': 500
}
```

**Returns**: `DetectionType.FACE_SPOOF`

---

### 6. SpeechAnalyzer (`speech_analyzer.py`)
**Purpose**: Analyzes speech for suspicious keywords and patterns

**Technology**: Audio Analysis + Keyword Matching

**Detection Method**:
- Calculates RMS volume level
- Tracks speech duration
- Matches suspicious keywords in transcribed text
- Maintains audio buffer for analysis

**Configuration**:
```python
{
    'volume_threshold': 0.1,
    'speech_duration_threshold': 2.0,  # seconds
    'suspicious_keywords': [
        'answer', 'solution', 'help', 'cheat',
        'google', 'search', 'tell me', 'what is'
    ]
}
```

**Returns**: `DetectionType.SUSPICIOUS_SPEECH`

---

## DetectorManager (`detector_manager.py`)

Coordinates all detectors and integrates with the Context Engine.

**Features**:
- Manages detector lifecycle (start/stop)
- Processes video frames through all detectors
- Handles audio processing
- Forwards events to ContextCueAnalyzer
- Provides health monitoring
- Supports event callbacks

**Usage**:
```python
from detectors.detector_manager import DetectorManager
from context_engine.analyzer import ContextCueAnalyzer
from context_engine.models import SystemConfiguration

config = SystemConfiguration()
analyzer = ContextCueAnalyzer(config)
manager = DetectorManager(config, analyzer)

# Start detection (web streaming mode)
manager.start_detection(use_camera=False)

# Process frame from web
result = manager.process_web_frame(frame)

# Stop detection
manager.stop_detection()
```

---

## Common Interface

All detectors implement the `DetectionSource` interface:

```python
class DetectionSource(ABC):
    @abstractmethod
    def get_source_name(self) -> str:
        """Return unique detector name"""
        
    @abstractmethod
    def is_available(self) -> bool:
        """Check if detector can run"""
        
    @abstractmethod
    def start_detection(self) -> None:
        """Initialize and start detection"""
        
    @abstractmethod
    def stop_detection(self) -> None:
        """Stop detection and cleanup"""
        
    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Return health status"""
```

---

## Detection Events

All detectors return `DetectionEvent` objects:

```python
@dataclass
class DetectionEvent:
    event_type: DetectionType
    timestamp: datetime
    confidence: float
    source: str
    metadata: Dict[str, Any]
    frame_data: Optional[np.ndarray]
```

---

## Session Data Structure

Each session creates a UUID folder:

```
sessions/{session_id}/
├── session_data.json
├── cheating_YYYYMMDD_HHMMSS_mmm.jpg
├── audio_YYYYMMDD_HHMMSS.wav
└── [additional evidence]
```

**session_data.json**:
```json
{
  "session_id": "uuid",
  "start_time": "ISO timestamp",
  "answers": [
    {
      "timestamp": "ISO timestamp",
      "question_id": "1",
      "answer": "user answer"
    }
  ],
  "cheating_detections": [
    {
      "timestamp": "ISO timestamp",
      "detection_type": "multiple_faces",
      "confidence": 0.95,
      "description": "Multiple people detected",
      "screenshot": "filename.jpg"
    }
  ]
}
```

---

## Testing

Run the test script to verify all detectors:

```bash
python test_detectors.py
```

This will:
1. Test imports
2. Initialize all detectors
3. Check availability
4. Test start/stop
5. Test detection with dummy data
6. Verify session folder structure

---

## Performance

- **Frame Rate**: ~30 FPS target
- **Detection Latency**: <50ms per detector
- **Memory Usage**: Bounded buffers (5-10 frames)
- **CPU Usage**: Optimized for real-time processing

---

## Error Handling

All detectors:
- Return `None` on errors (don't crash)
- Print error messages for debugging
- Continue operation when possible
- Implement graceful degradation

---

## Dependencies

- **OpenCV**: Image processing and face detection
- **MediaPipe**: Face mesh and landmarks
- **Ultralytics YOLO**: Object detection
- **NumPy**: Numerical operations
- **PyAudio** (optional): Audio recording

---

## Configuration Best Practices

1. **Adjust thresholds** based on environment
2. **Use history buffers** to reduce false positives
3. **Balance sensitivity** vs false alarm rate
4. **Test with real scenarios** before deployment
5. **Monitor performance metrics** in production

---

## Troubleshooting

### Detector Not Available
- Check dependencies are installed
- Verify model files exist
- Check system permissions

### Low Detection Rate
- Adjust confidence thresholds
- Increase history buffer size
- Check lighting conditions

### High False Positive Rate
- Increase confidence thresholds
- Increase stability requirements
- Add more context to analysis

### Performance Issues
- Reduce frame rate
- Use smaller YOLO models
- Disable unused detectors

---

## Future Enhancements

1. **Deep learning speech recognition** for better keyword detection
2. **Advanced face spoof detection** using neural networks
3. **Eye tracking** for more precise gaze detection
4. **Gesture recognition** for suspicious movements
5. **Multi-camera support** for comprehensive monitoring

---

## License

Part of the AI Proctoring System project.

---

## Support

For issues or questions, refer to the main project documentation.

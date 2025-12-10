# AI Proctoring System - Enhanced Edition

A comprehensive AI-powered proctoring system with screen capture, audio recording, and advanced cheating detection.

## 🚀 Features

### Core Proctoring Capabilities
- **Face Detection & Analysis**: Real-time face detection with anti-spoofing
- **Gaze Tracking**: Detects when students look away from the screen
- **Mobile Device Detection**: Identifies mobile phones in the camera view
- **Speech Detection**: Monitors for talking during the exam
- **Multiple Person Detection**: Alerts when multiple people are detected

### Enhanced Monitoring
- **Screen Capture**: Automatic screenshots of the desktop for evidence
- **System Audio Recording**: Direct microphone access for audio evidence
- **Browser Monitoring**: Detects tab switching and window changes
- **Keyboard/Mouse Monitoring**: Prevents cheating shortcuts

### Evidence Collection
- **Automatic Screenshots**: Saved when violations are detected
- **Audio Recordings**: Continuous audio monitoring with evidence files
- **Session Logs**: Detailed logs of all detection events
- **Violation Reports**: Comprehensive reports for review

## 📋 Requirements

### System Requirements
- Python 3.8+
- Windows 10/11, macOS 10.14+, or Linux
- Webcam (required)
- Microphone (recommended)
- Modern web browser with WebRTC support

### Python Dependencies
```bash
pip install -r requirements.txt
```

Key dependencies:
- Flask (web framework)
- OpenCV (computer vision)
- MediaPipe (face detection)
- PyAudio (audio recording)
- PyAutoGUI (screen capture)
- MSS (fast screen capture)
- Ultralytics (object detection)

## 🛠️ Installation

### Quick Start
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python app.py`
4. Open browser to `http://localhost:5000`

### Docker Deployment
```bash
# Standard deployment
docker-compose up -d

# Windows deployment 
docker-compose -f docker-compose.yml -f docker-compose.windows.yml up -d
```

## 🎯 Usage Flow

### 1. Student Information Entry
- Students enter their name, ID, and email on the main page
- Information is validated and passed to the permissions page

### 2. Permissions & Setup
- Camera and microphone permissions are requested
- System diagnostics verify browser compatibility
- Screen capture and audio recording are initialized

### 3. Exam Taking
- Real-time monitoring begins automatically
- Multiple detection systems run simultaneously:
  - Video analysis (face, gaze, mobile detection)
  - Audio monitoring (speech detection)
  - Screen capture (desktop monitoring)
  - Browser behavior tracking

### 4. Results & Evidence
- Exam results with violation summary
- Evidence files (screenshots, audio recordings)
- Detailed detection logs for review

## 🔧 Configuration

### Detection Thresholds
Edit detector configurations in `detectors/fixed_detector_manager.py`:

```python
detector_configs = {
    'gaze': {
        'yaw_threshold': 20,      # Head rotation threshold
        'pitch_threshold': 15     # Head tilt threshold
    },
    'mobile': {
        'confidence_threshold': 0.6  # Mobile detection confidence
    },
    'face_guard': {
        'spoof_threshold': 100,   # Anti-spoofing sensitivity
        'lip_threshold': 8        # Speech detection sensitivity
    },
    'system_audio': {
        'speech_threshold': 0.3,  # Audio speech threshold
        'noise_threshold': 0.1    # Background noise threshold
    },
    'screen_capture': {
        'capture_interval': 2.0,  # Screenshot interval (seconds)
        'window_change_threshold': 0.3  # Screen change sensitivity
    }
}
```

### Environment Variables
```bash
FLASK_ENV=development          # Development/production mode
FLASK_SECRET_KEY=your-key     # Session encryption key
PORT=5000                     # Server port
```

## 📁 File Structure

```
AI-Proctoring-System/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── cleanup.py               # Cleanup utility
├── detectors/               # Detection modules
│   ├── face_guard_detector.py      # Face analysis
│   ├── fixed_gaze_detector.py      # Gaze tracking
│   ├── fixed_mobile_detector.py    # Mobile detection
│   ├── system_audio_detector.py    # Audio recording
│   ├── screen_capture_detector.py  # Screen capture
│   └── fixed_detector_manager.py   # Detector coordination
├── templates/               # HTML templates
│   ├── index.html          # Student info entry
│   ├── permissions.html    # Permissions setup
│   ├── exam.html          # Exam interface
│   └── results.html       # Results display
├── static/                 # Static assets
│   ├── css/style.css      # Styling
│   └── js/exam.js         # Frontend JavaScript
└── sessions/              # Evidence storage
    └── [session-id]/
        ├── screenshots/   # Violation screenshots
        ├── audio/        # Audio recordings
        └── answers/      # Exam answers
```

## 🔍 Detection Methods

### Video-Based Detection
- **Face Guard**: MediaPipe-based face analysis with anti-spoofing
- **Gaze Tracking**: Head pose estimation using SolvePnP
- **Mobile Detection**: YOLOv8-based object detection
- **Multiple Person Detection**: Face count monitoring

### Audio-Based Detection
- **Speech Recognition**: Google Speech API integration
- **Volume Analysis**: RMS level monitoring
- **Spectral Analysis**: Frequency domain analysis (with librosa)
- **Continuous Recording**: Evidence collection

### System-Level Detection
- **Screen Capture**: Desktop monitoring with MSS/PyAutoGUI
- **Window Analysis**: Process monitoring with psutil
- **Browser Events**: Tab switching and focus detection
- **Keyboard Monitoring**: Shortcut prevention

## 🛡️ Security Features

### Anti-Cheating Measures
- Right-click context menu disabled
- Developer tools shortcuts blocked
- Tab switching detection and warnings
- Full-screen enforcement
- Copy/paste prevention

### Privacy & Data Protection
- Local evidence storage
- Session-based data isolation
- Automatic cleanup utilities
- Configurable retention policies

## 🚨 Troubleshooting

### Common Issues

**Camera Access Denied**
- Check browser permissions
- Close other applications using camera
- Restart browser

**Audio Recording Failed**
- Verify microphone permissions
- Check system audio settings
- Install PyAudio dependencies

**Screen Capture Not Working**
- Install MSS: `pip install mss`
- Check system permissions
- Verify PyAutoGUI installation

**Detection Not Working**
- Check model files in `models/` directory
- Verify OpenCV installation
- Check MediaPipe compatibility

### Debug Mode
Enable debug logging:
```bash
FLASK_ENV=development python app.py
```

### Cleanup
Remove temporary files:
```bash
python cleanup.py
```

## 📊 Performance Optimization

### Resource Usage
- CPU: 2-4 cores recommended
- RAM: 2GB minimum, 4GB recommended
- Storage: 1GB for models + evidence storage
- Network: Minimal (local processing)

### Optimization Tips
- Use Docker for consistent performance
- Configure detection thresholds appropriately
- Monitor evidence storage usage
- Regular cleanup of old sessions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the debug console output
3. Check browser compatibility
4. Verify system requirements

---

**Note**: This system is designed for educational and testing purposes. Ensure compliance with local privacy laws and institutional policies when deploying in production environments.
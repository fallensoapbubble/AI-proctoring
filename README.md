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


### See Errors
```bash
docker logs -f --tail 100 -t ai-proctoring-system-proctoring-app-1
```

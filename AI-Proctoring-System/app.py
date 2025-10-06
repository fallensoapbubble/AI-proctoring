#!/usr/bin/env python3
"""
AI Proctoring System - Complete Flask Web Application
Connects all templates and detector functions properly.
"""

import os
import sys
import uuid
import json
import base64
import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our components
try:
    from simple_proctoring_system import UUIDSessionManager
    print("✅ Successfully imported UUIDSessionManager")
except ImportError as e:
    print(f"❌ Failed to import UUIDSessionManager: {e}")
    # Create a fallback session manager
    class UUIDSessionManager:
        def __init__(self):
            self.base_dir = "sessions"
            import os
            os.makedirs(self.base_dir, exist_ok=True)
        
        def create_session(self):
            import uuid
            return str(uuid.uuid4())
        
        def save_screenshot(self, session_id, image_data, detection_type):
            import os
            from datetime import datetime
            session_dir = os.path.join(self.base_dir, session_id)
            screenshots_dir = os.path.join(session_dir, "screenshots")
            os.makedirs(screenshots_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"screenshot_{detection_type}_{timestamp}.jpg"
            filepath = os.path.join(screenshots_dir, filename)
            
            with open(filepath, 'wb') as f:
                f.write(image_data)
            
            print(f"Saved screenshot: {filename}")
            return filepath
        
        def save_answer(self, session_id, question_id, answer, question_text):
            import os, json
            from datetime import datetime
            session_dir = os.path.join(self.base_dir, session_id)
            answers_dir = os.path.join(session_dir, "answers")
            os.makedirs(answers_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"answer_{question_id}_{timestamp}.json"
            filepath = os.path.join(answers_dir, filename)
            
            answer_data = {
                "question_id": question_id,
                "answer": answer,
                "question_text": question_text,
                "timestamp": timestamp
            }
            
            with open(filepath, 'w') as f:
                json.dump(answer_data, f, indent=2)
            
            return filepath
        
        def finalize_session(self, session_id):
            import os
            session_dir = os.path.join(self.base_dir, session_id)
            file_counts = {"screenshots": 0, "audio": 0, "answers": 0}
            
            for subdir in file_counts.keys():
                subdir_path = os.path.join(session_dir, subdir)
                if os.path.exists(subdir_path):
                    file_counts[subdir] = len(os.listdir(subdir_path))
            
            return file_counts
from detectors.fixed_detector_manager import FixedDetectorManager
from detectors.fixed_gaze_detector import FixedGazeDetector
from detectors.fixed_mobile_detector import FixedMobileDetector

# Import context engine components
try:
    from context_engine.simple_analyzer import SimpleAnalyzer
    from context_engine.simple_config import SimpleConfig
except ImportError:
    SimpleAnalyzer = None
    SimpleConfig = None

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')

# Global components
session_manager = UUIDSessionManager()
detector_manager = None
current_sessions = {}

# Sample exam questions
EXAM_QUESTIONS = [
    {
        "id": 1,
        "question": "What is the capital of France?",
        "options": ["London", "Berlin", "Paris", "Madrid"],
        "correct": 2
    },
    {
        "id": 2,
        "question": "Which programming language is known for its use in data science?",
        "options": ["JavaScript", "Python", "C++", "Java"],
        "correct": 1
    },
    {
        "id": 3,
        "question": "What does AI stand for?",
        "options": ["Automated Intelligence", "Artificial Intelligence", "Advanced Integration", "Algorithmic Interface"],
        "correct": 1
    },
    {
        "id": 4,
        "question": "Which of the following is a machine learning framework?",
        "options": ["React", "TensorFlow", "Bootstrap", "jQuery"],
        "correct": 1
    },
    {
        "id": 5,
        "question": "What is the primary purpose of computer vision?",
        "options": ["Process text", "Analyze images", "Store data", "Network communication"],
        "correct": 1
    }
]

def initialize_detector_manager():
    """Initialize the detector manager with proper configuration."""
    global detector_manager
    
    try:
        # Create simple configuration
        config = SimpleConfig() if SimpleConfig else None
        analyzer = SimpleAnalyzer(config) if SimpleAnalyzer else None
        
        # Initialize detector manager
        detector_manager = FixedDetectorManager(config=config, analyzer=analyzer)
        
        # Add event callback for handling detections
        detector_manager.add_event_callback(handle_detection_event)
        
        # Start detection in web mode (no direct camera access)
        detector_manager.start_detection(use_camera=False)
        
        print("✅ Detector manager initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize detector manager: {e}")
        return False

def handle_detection_event(event):
    """Handle detection events from the detector manager."""
    try:
        # Store detection event in session data
        session_id = session.get('session_id')
        if session_id and session_id in current_sessions:
            if 'detections' not in current_sessions[session_id]:
                current_sessions[session_id]['detections'] = []
            
            detection_data = {
                'timestamp': event.timestamp.isoformat() if hasattr(event, 'timestamp') else datetime.now().isoformat(),
                'type': str(event.event_type) if hasattr(event, 'event_type') else 'unknown',
                'confidence': float(event.confidence) if hasattr(event, 'confidence') else 0.0,
                'source': event.source if hasattr(event, 'source') else 'unknown',
                'metadata': event.metadata if hasattr(event, 'metadata') else {}
            }
            
            current_sessions[session_id]['detections'].append(detection_data)
            
            # Save detection to persistent storage (only for real frame data)
            if hasattr(event, 'frame_data') and event.frame_data is not None:
                # Check if frame_data looks like real JPEG data (starts with JPEG magic bytes)
                if isinstance(event.frame_data, bytes) and len(event.frame_data) > 4:
                    # Check for JPEG magic bytes (FF D8 FF)
                    if event.frame_data[:3] == b'\xff\xd8\xff':
                        # Clean up detection type - remove enum prefix and make it simple
                        raw_type = str(event.event_type)
                        print(f"🔍 Event detection type: {raw_type}")
                        
                        # Clean the type name properly
                        if 'DetectionType.' in raw_type:
                            clean_type = raw_type.replace('DetectionType.', '').lower()
                        else:
                            clean_type = raw_type.lower()
                        
                        # Remove any existing timestamps from the type name
                        import re
                        clean_type = re.sub(r'_\d{8}_\d{6}.*', '', clean_type)
                        
                        # Map to simple names
                        type_mapping = {
                            'gaze_away': 'gaze',
                            'mobile_detected': 'mobile',
                            'multiple_people': 'people',
                            'face_not_visible': 'face',
                            'lip_movement': 'lips'
                        }
                        
                        detection_type = type_mapping.get(clean_type, clean_type)
                        print(f"🔍 Cleaned event detection type: {detection_type}")
                        
                        session_manager.save_screenshot(
                            session_id, 
                            event.frame_data, 
                            detection_type
                        )
                        print(f"📸 Saved real screenshot for {detection_type} detection")
                    else:
                        print(f"⚠️ Skipping screenshot save - frame_data doesn't contain valid JPEG data")
                else:
                    print(f"⚠️ Skipping screenshot save - invalid frame_data format")
            
            print(f"🚨 Detection event handled: {detection_data['type']} (confidence: {detection_data['confidence']:.2f})")
            
    except Exception as e:
        print(f"Error handling detection event: {e}")

@app.route('/')
def index():
    """Main landing page."""
    return render_template('index.html')

@app.route('/permissions')
def permissions():
    """Permissions page for camera and microphone access."""
    return render_template('permissions.html')

@app.route('/start_exam', methods=['POST'])
def start_exam():
    """Start a new exam session."""
    try:
        # Get student information
        data = request.get_json() or {}
        full_name = data.get('full_name', 'Anonymous Student')
        student_id = data.get('student_id', 'UNKNOWN')
        email = data.get('email', 'unknown@example.com')
        
        # Create new session
        session_id = session_manager.create_session()
        
        # Store session data
        session['session_id'] = session_id
        session['student_name'] = full_name
        session['student_id'] = student_id
        session['email'] = email
        session['start_time'] = datetime.now().isoformat()
        
        # Initialize session tracking
        current_sessions[session_id] = {
            'student_name': full_name,
            'student_id': student_id,
            'email': email,
            'start_time': datetime.now(),
            'answers': {},
            'detections': [],
            'status': 'active'
        }
        
        print(f"✅ Started exam session {session_id} for {full_name}")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'redirect': url_for('exam')
        })
        
    except Exception as e:
        print(f"❌ Error starting exam: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/exam')
def exam():
    """Main exam page with proctoring."""
    session_id = session.get('session_id')
    if not session_id:
        flash('Please start the exam from the beginning.', 'error')
        return redirect(url_for('index'))
    
    return render_template('exam.html', 
                         questions=EXAM_QUESTIONS,
                         session_id=session_id,
                         student_name=session.get('student_name', 'Student'))

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    """Process video frame from the frontend."""
    try:
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({'error': 'No frame data provided'}), 400
        
        session_id = data.get('session_id')
        if not session_id or session_id not in current_sessions:
            return jsonify({'error': 'Invalid session'}), 400
        
        # Decode base64 frame
        frame_data = data['frame']
        if frame_data.startswith('data:image'):
            frame_data = frame_data.split(',')[1]
        
        # Convert to OpenCV format
        frame_bytes = base64.b64decode(frame_data)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Failed to decode frame'}), 400
        
        # Initialize result
        result = {
            'status': 'processed', 
            'timestamp': datetime.now().isoformat(),
            'frame_processed': True
        }
        
        # Process frame with detector manager
        if detector_manager and detector_manager.is_running:
            try:
                detection_result = detector_manager.process_web_frame(frame)
                result.update(detection_result)
                
                # Run individual detectors manually to ensure they work
                detections_found = []
                
                # Test gaze detection
                gaze_detector = detector_manager.detectors.get('gaze')
                if gaze_detector and hasattr(gaze_detector, 'detect_gaze_direction'):
                    try:
                        gaze_event = gaze_detector.detect_gaze_direction(frame)
                        if gaze_event:
                            detections_found.append({
                                'type': 'GAZE_AWAY',
                                'confidence': float(gaze_event.confidence) if hasattr(gaze_event, 'confidence') else 0.8,
                                'timestamp': datetime.now().isoformat(),
                                'source': 'gaze_detector'
                            })
                            print(f"👁️ Gaze detection triggered: confidence {gaze_event.confidence}")
                    except Exception as e:
                        print(f"Gaze detection error: {e}")
                
                # Test mobile detection
                mobile_detector = detector_manager.detectors.get('mobile')
                if mobile_detector and hasattr(mobile_detector, 'detect_mobile_devices'):
                    try:
                        mobile_event = mobile_detector.detect_mobile_devices(frame)
                        if mobile_event:
                            detections_found.append({
                                'type': 'MOBILE_DETECTED',
                                'confidence': float(mobile_event.confidence) if hasattr(mobile_event, 'confidence') else 0.8,
                                'timestamp': datetime.now().isoformat(),
                                'source': 'mobile_detector'
                            })
                            print(f"📱 Mobile detection triggered: confidence {mobile_event.confidence}")
                    except Exception as e:
                        print(f"Mobile detection error: {e}")
                
                # Test person detection (multiple people)
                person_detector = detector_manager.detectors.get('person')
                if person_detector and hasattr(person_detector, 'detect_people'):
                    try:
                        person_event = person_detector.detect_people(frame)
                        if person_event:
                            detections_found.append({
                                'type': 'MULTIPLE_PEOPLE',
                                'confidence': float(person_event.confidence) if hasattr(person_event, 'confidence') else 0.8,
                                'timestamp': datetime.now().isoformat(),
                                'source': 'person_detector'
                            })
                            print(f"👥 Person detection triggered: confidence {person_event.confidence}")
                    except Exception as e:
                        print(f"Person detection error: {e}")
                
                # Add any new detections to session
                if detections_found:
                    if 'detections' not in current_sessions[session_id]:
                        current_sessions[session_id]['detections'] = []
                    
                    current_sessions[session_id]['detections'].extend(detections_found)
                    
                    # Save screenshot for significant detections
                    for detection in detections_found:
                        if detection['confidence'] > 0.6:
                            try:
                                # Encode frame as JPEG
                                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                                
                                # Clean up detection type - remove enum prefix and make it simple
                                raw_type = str(detection['type'])
                                print(f"🔍 Raw detection type: {raw_type}")
                                
                                # Clean the type name properly
                                if 'DetectionType.' in raw_type:
                                    clean_type = raw_type.replace('DetectionType.', '').lower()
                                else:
                                    clean_type = raw_type.lower()
                                
                                # Map to simple names
                                type_mapping = {
                                    'gaze_away': 'gaze',
                                    'mobile_detected': 'mobile',
                                    'multiple_people': 'people',
                                    'face_not_visible': 'face',
                                    'lip_movement': 'lips'
                                }
                                
                                detection_type = type_mapping.get(clean_type, clean_type)
                                print(f"🔍 Final detection type: {detection_type}")
                                
                                # Save using session manager (it creates its own filename)
                                saved_path = session_manager.save_screenshot(
                                    session_id, 
                                    buffer.tobytes(), 
                                    detection_type
                                )
                                print(f"📸 Saved screenshot: {saved_path} ({len(buffer.tobytes())} bytes)")
                                
                            except Exception as e:
                                print(f"❌ Error saving screenshot: {e}")
                    
                    # ALWAYS return alert for ANY detection (immediate response)
                    highest_detection = max(detections_found, key=lambda x: x['confidence'])
                    result['alert'] = {
                        'type': highest_detection['type'],
                        'message': get_alert_message(highest_detection['type']),
                        'confidence': highest_detection['confidence'],
                        'timestamp': highest_detection['timestamp']
                    }
                    
                    print(f"🚨 IMMEDIATE ALERT: {highest_detection['type']} (confidence: {highest_detection['confidence']:.2f})")
                
                result['detections_count'] = len(detections_found)
                result['has_detections'] = len(detections_found) > 0
                
            except Exception as e:
                print(f"Error in detector processing: {e}")
                result['detector_error'] = str(e)
        else:
            result['detector_status'] = 'not_running'
            print("⚠️ Detector manager not running")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/save_answer', methods=['POST'])
def save_answer():
    """Save student answer."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        question_id = data.get('question_id')
        answer = data.get('answer')
        answer_type = data.get('type', 'mcq')
        
        if not session_id or session_id not in current_sessions:
            return jsonify({'error': 'Invalid session'}), 400
        
        # Save answer to session tracking
        current_sessions[session_id]['answers'][question_id] = {
            'answer': answer,
            'type': answer_type,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to persistent storage
        question_text = next((q['question'] for q in EXAM_QUESTIONS if q['id'] == int(question_id)), 'Unknown Question')
        session_manager.save_answer(session_id, question_id, answer, question_text)
        
        print(f"💾 Saved answer for session {session_id}, question {question_id}: {answer}")
        
        return jsonify({'status': 'saved', 'timestamp': datetime.now().isoformat()})
        
    except Exception as e:
        print(f"❌ Error saving answer: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/submit_exam', methods=['POST'])
def submit_exam():
    """Submit the exam and finalize session."""
    try:
        session_id = session.get('session_id')
        if not session_id or session_id not in current_sessions:
            return jsonify({'error': 'Invalid session'}), 400
        
        # Mark session as completed
        current_sessions[session_id]['status'] = 'completed'
        current_sessions[session_id]['end_time'] = datetime.now()
        
        # Calculate score
        answers = current_sessions[session_id].get('answers', {})
        score = calculate_exam_score(answers)
        current_sessions[session_id]['score'] = score
        
        # Save final session data to persistent storage
        try:
            # Save all answers to session manager
            for q_id, answer_data in answers.items():
                question_text = next((q['question'] for q in EXAM_QUESTIONS if q['id'] == int(q_id)), 'Unknown Question')
                session_manager.save_answer(session_id, q_id, answer_data['answer'], question_text)
            
            # Save detection summary
            detections = current_sessions[session_id].get('detections', [])
            if detections:
                detection_summary = {
                    'total_detections': len(detections),
                    'detection_types': {},
                    'high_confidence_detections': 0,
                    'detections': detections
                }
                
                for detection in detections:
                    det_type = detection.get('type', 'unknown')
                    detection_summary['detection_types'][det_type] = detection_summary['detection_types'].get(det_type, 0) + 1
                    if detection.get('confidence', 0) > 0.7:
                        detection_summary['high_confidence_detections'] += 1
                
                # Save detection summary as JSON file
                session_manager.save_detection_summary(session_id, detection_summary)
            
            # Finalize session in persistent storage
            file_counts = session_manager.finalize_session(session_id)
            
        except Exception as e:
            print(f"⚠️ Error saving session data: {e}")
            file_counts = {'screenshots': 0, 'audio': 0, 'answers': len(answers)}
        
        # Store final results in Flask session
        session['exam_completed'] = True
        session['final_score'] = score
        session['file_counts'] = file_counts
        session['session_data'] = {
            'session_id': session_id,
            'student_name': current_sessions[session_id].get('student_name', 'Unknown'),
            'student_id': current_sessions[session_id].get('student_id', 'Unknown'),
            'email': current_sessions[session_id].get('email', 'Unknown'),
            'start_time': current_sessions[session_id].get('start_time', datetime.now()).isoformat() if isinstance(current_sessions[session_id].get('start_time'), datetime) else current_sessions[session_id].get('start_time'),
            'end_time': current_sessions[session_id]['end_time'].isoformat(),
            'status': 'completed',
            'detections': current_sessions[session_id].get('detections', []),
            'answers_count': len(answers)
        }
        
        print(f"✅ Exam submitted for session {session_id}, score: {score['percentage']:.1f}%")
        print(f"📊 Session summary: {len(answers)} answers, {len(current_sessions[session_id].get('detections', []))} detections")
        
        return jsonify({
            'success': True,
            'score': score,
            'redirect': url_for('results')
        })
        
    except Exception as e:
        print(f"❌ Error submitting exam: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/results')
def results():
    """Show exam results."""
    if not session.get('exam_completed'):
        flash('Please complete the exam first.', 'error')
        return redirect(url_for('index'))
    
    # Get data from Flask session (more reliable than current_sessions)
    final_score = session.get('final_score', {})
    session_data = session.get('session_data', {})
    file_counts = session.get('file_counts', {})
    
    # If session_data is empty, try to get from current_sessions as fallback
    if not session_data:
        session_id = session.get('session_id')
        if session_id and session_id in current_sessions:
            session_data = current_sessions[session_id]
            session_data['session_id'] = session_id
    
    print(f"📊 Displaying results: Score={final_score}, Detections={len(session_data.get('detections', []))}, Files={file_counts}")
    
    return render_template('results.html',
                         score=final_score.get('correct', 0),
                         total=final_score.get('total', len(EXAM_QUESTIONS)),
                         percentage=final_score.get('percentage', 0),
                         grade=final_score.get('grade', 'F'),
                         session_data=session_data,
                         file_counts=file_counts)

@app.route('/completion')
def completion():
    """Exam completion page."""
    return render_template('completion.html')

@app.route('/api/session_files/<session_id>')
def get_session_files(session_id):
    """Get list of files for a session."""
    try:
        import os
        
        # Use absolute path
        base_path = os.path.abspath('sessions')
        session_path = os.path.join(base_path, session_id)
        
        print(f"🔍 Looking for session: {session_path}")
        print(f"📁 Session exists: {os.path.exists(session_path)}")
        
        if not os.path.exists(session_path):
            # List available sessions for debugging
            if os.path.exists(base_path):
                available_sessions = os.listdir(base_path)
                print(f"📋 Available sessions: {available_sessions}")
            return jsonify({'error': f'Session not found: {session_id}'}), 404
        
        files = {
            'screenshots': [],
            'audio': [],
            'answers': [],
            'metadata': []
        }
        
        # List screenshots
        screenshots_path = os.path.join(session_path, 'screenshots')
        print(f"📸 Screenshots path: {screenshots_path}")
        print(f"📸 Screenshots path exists: {os.path.exists(screenshots_path)}")
        
        if os.path.exists(screenshots_path):
            screenshot_files = os.listdir(screenshots_path)
            print(f"📸 Found screenshot files: {screenshot_files}")
            
            for file in screenshot_files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(screenshots_path, file)
                    # URL encode the filename to handle special characters
                    from urllib.parse import quote
                    encoded_filename = quote(file, safe='')
                    files['screenshots'].append({
                        'filename': file,
                        'url': f'/api/session_file/{session_id}/screenshots/{encoded_filename}',
                        'size': os.path.getsize(file_path)
                    })
        
        # List audio files
        audio_path = os.path.join(session_path, 'audio')
        print(f"🎵 Audio path: {audio_path}")
        print(f"🎵 Audio path exists: {os.path.exists(audio_path)}")
        
        if os.path.exists(audio_path):
            audio_files = os.listdir(audio_path)
            print(f"🎵 Found audio files: {audio_files}")
            
            for file in audio_files:
                if file.endswith(('.wav', '.mp3', '.ogg')):
                    file_path = os.path.join(audio_path, file)
                    files['audio'].append({
                        'filename': file,
                        'url': f'/api/session_file/{session_id}/audio/{file}',
                        'size': os.path.getsize(file_path)
                    })
        
        # List answer files
        answers_path = os.path.join(session_path, 'answers')
        if os.path.exists(answers_path):
            for file in os.listdir(answers_path):
                if file.endswith('.json'):
                    file_path = os.path.join(answers_path, file)
                    files['answers'].append({
                        'filename': file,
                        'url': f'/api/session_file/{session_id}/answers/{file}',
                        'size': os.path.getsize(file_path)
                    })
        
        print(f"📊 Found files: {len(files['screenshots'])} screenshots, {len(files['audio'])} audio, {len(files['answers'])} answers")
        
        return jsonify(files)
        
    except Exception as e:
        print(f"❌ Error getting session files: {e}")
        return jsonify({'error': str(e)}), 500



@app.route('/api/debug_session/<session_id>')
def debug_session(session_id):
    """Debug endpoint to show session file structure."""
    try:
        import os
        
        base_path = os.path.abspath('sessions')
        session_path = os.path.join(base_path, session_id)
        
        debug_info = {
            'session_id': session_id,
            'session_path': session_path,
            'session_exists': os.path.exists(session_path),
            'files': {}
        }
        
        if os.path.exists(session_path):
            for subdir in ['screenshots', 'audio', 'answers']:
                subdir_path = os.path.join(session_path, subdir)
                if os.path.exists(subdir_path):
                    files = os.listdir(subdir_path)
                    debug_info['files'][subdir] = []
                    for file in files:
                        file_path = os.path.join(subdir_path, file)
                        debug_info['files'][subdir].append({
                            'filename': file,
                            'size': os.path.getsize(file_path),
                            'exists': os.path.exists(file_path),
                            'url': f'/api/session_file/{session_id}/{subdir}/{file}'
                        })
                else:
                    debug_info['files'][subdir] = 'Directory does not exist'
        
        return jsonify(debug_info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cleanup_session_files/<session_id>')
def cleanup_session_files(session_id):
    """Clean up session files by renaming problematic filenames."""
    try:
        import os
        import re
        
        base_path = os.path.abspath('sessions')
        session_path = os.path.join(base_path, session_id)
        
        if not os.path.exists(session_path):
            return jsonify({'error': 'Session not found'}), 404
        
        screenshots_path = os.path.join(session_path, 'screenshots')
        if not os.path.exists(screenshots_path):
            return jsonify({'error': 'Screenshots directory not found'}), 404
        
        renamed_files = []
        errors = []
        
        for filename in os.listdir(screenshots_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                old_path = os.path.join(screenshots_path, filename)
                
                # Check if filename needs cleaning
                if 'DetectionType.' in filename or len(filename) > 80:
                    try:
                        # Clean the filename using the same logic as session manager
                        clean_name = session_manager._sanitize_filename(filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', ''))
                        
                        # Add timestamp to ensure uniqueness
                        from datetime import datetime
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                        
                        # Get file extension
                        ext = os.path.splitext(filename)[1]
                        new_filename = f"screenshot_{clean_name}_{timestamp}{ext}"
                        new_path = os.path.join(screenshots_path, new_filename)
                        
                        # Rename the file
                        os.rename(old_path, new_path)
                        
                        renamed_files.append({
                            'old_name': filename,
                            'new_name': new_filename,
                            'old_length': len(filename),
                            'new_length': len(new_filename)
                        })
                        
                        print(f"📝 Renamed: {filename} -> {new_filename}")
                        
                    except Exception as e:
                        errors.append({
                            'filename': filename,
                            'error': str(e)
                        })
                        print(f"❌ Error renaming {filename}: {e}")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'renamed_files': renamed_files,
            'errors': errors,
            'total_renamed': len(renamed_files),
            'total_errors': len(errors)
        })
        
    except Exception as e:
        print(f"❌ Error cleaning up session files: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/session_file/<session_id>/<file_type>/<filename>')
def serve_session_file(session_id, file_type, filename):
    """Serve a specific session file."""
    try:
        import os
        from flask import send_file, send_from_directory
        
        # Validate file type
        if file_type not in ['screenshots', 'audio', 'answers', 'metadata']:
            return jsonify({'error': 'Invalid file type'}), 400
        
        # URL decode the filename
        from urllib.parse import unquote
        decoded_filename = unquote(filename)
        
        # Build file path
        base_path = os.path.abspath('sessions')
        session_path = os.path.join(base_path, session_id)
        file_path = os.path.join(session_path, file_type, decoded_filename)
        
        print(f"🔍 Serving file: {file_path} (decoded from: {filename})")
        print(f"📁 File exists: {os.path.exists(file_path)}")
        
        # Security check - ensure file is within session directory
        if not file_path.startswith(session_path):
            return jsonify({'error': 'Access denied'}), 403
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            # List files in directory for debugging
            dir_path = os.path.join(session_path, file_type)
            if os.path.exists(dir_path):
                available_files = os.listdir(dir_path)
                print(f"📋 Available files in {file_type}: {available_files}")
            return jsonify({'error': 'File not found'}), 404
        
        # Determine MIME type
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg', 
            '.png': 'image/png',
            '.wav': 'audio/wav',
            '.mp3': 'audio/mpeg',
            '.ogg': 'audio/ogg',
            '.json': 'application/json'
        }
        
        file_ext = os.path.splitext(decoded_filename)[1].lower()
        mime_type = mime_types.get(file_ext, 'application/octet-stream')
        
        print(f"📤 Serving {decoded_filename} as {mime_type}")
        
        return send_file(file_path, mimetype=mime_type)
        
    except Exception as e:
        print(f"❌ Error serving session file: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint."""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'flask': True,
                'session_manager': session_manager is not None,
                'detector_manager': detector_manager is not None and detector_manager.is_running if detector_manager else False,
                'cv2': True  # OpenCV is required and should be available
            }
        }
        
        # Add detector health if available
        if detector_manager:
            detector_health = detector_manager.get_detector_health()
            health_status['detectors'] = detector_health
        
        return jsonify(health_status)
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/session_status')
def session_status():
    """Get current session status."""
    session_id = session.get('session_id')
    if not session_id or session_id not in current_sessions:
        return jsonify({'error': 'No active session'}), 400
    
    session_data = current_sessions[session_id]
    
    return jsonify({
        'session_id': session_id,
        'status': session_data.get('status', 'unknown'),
        'answers_count': len(session_data.get('answers', {})),
        'detections_count': len(session_data.get('detections', [])),
        'start_time': session_data.get('start_time', datetime.now()).isoformat() if isinstance(session_data.get('start_time'), datetime) else session_data.get('start_time')
    })

def calculate_exam_score(answers: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate exam score based on answers."""
    correct_answers = 0
    total_questions = len(EXAM_QUESTIONS)
    
    for question in EXAM_QUESTIONS:
        question_id = str(question['id'])
        if question_id in answers:
            student_answer = answers[question_id]['answer']
            if isinstance(student_answer, int) and student_answer == question['correct']:
                correct_answers += 1
    
    percentage = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
    
    return {
        'correct': correct_answers,
        'total': total_questions,
        'percentage': percentage,
        'grade': get_letter_grade(percentage)
    }

def get_letter_grade(percentage: float) -> str:
    """Convert percentage to letter grade."""
    if percentage >= 90:
        return 'A'
    elif percentage >= 80:
        return 'B'
    elif percentage >= 70:
        return 'C'
    elif percentage >= 60:
        return 'D'
    else:
        return 'F'

def get_alert_message(detection_type: str) -> str:
    """Get user-friendly alert message for detection type."""
    messages = {
        'GAZE_AWAY': 'Looking away from the screen detected',
        'MOBILE_DETECTED': 'Mobile device detected in frame',
        'MULTIPLE_PEOPLE': 'Multiple people detected',
        'FACE_NOT_VISIBLE': 'Face not clearly visible',
        'SUSPICIOUS_AUDIO': 'Suspicious audio activity detected',
        'LIP_MOVEMENT': 'Unusual lip movement detected'
    }
    return messages.get(detection_type, 'Suspicious activity detected')

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('error.html', 
                         error_code=404,
                         error_message="Page not found"), 404

@app.route('/test_results/<session_id>')
def test_results(session_id):
    """Test results page with a specific session."""
    # Create fake session data for testing
    fake_session_data = {
        'session_id': session_id,
        'student_name': 'Test Student',
        'student_id': 'TEST123',
        'email': 'test@example.com',
        'start_time': datetime.now().isoformat(),
        'end_time': datetime.now().isoformat(),
        'status': 'completed',
        'detections': [
            {'type': 'GAZE_AWAY', 'confidence': 0.8, 'timestamp': datetime.now().isoformat(), 'source': 'test'},
            {'type': 'MOBILE_DETECTED', 'confidence': 0.9, 'timestamp': datetime.now().isoformat(), 'source': 'test'}
        ],
        'answers_count': 5
    }
    
    fake_score = {
        'correct': 4,
        'total': 5,
        'percentage': 80.0,
        'grade': 'B'
    }
    
    fake_file_counts = {
        'screenshots': 7,
        'audio': 1,
        'answers': 2
    }
    
    return render_template('results.html',
                         score=fake_score['correct'],
                         total=fake_score['total'],
                         percentage=fake_score['percentage'],
                         grade=fake_score['grade'],
                         session_data=fake_session_data,
                         file_counts=fake_file_counts)

@app.route('/api/test_screenshot/<session_id>')
def test_screenshot(session_id):
    """Create a test screenshot for debugging."""
    try:
        import os
        import numpy as np
        
        # Create a test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[:] = (100, 150, 200)  # Fill with a color
        
        # Add some text
        cv2.putText(test_image, 'Test Screenshot', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.putText(test_image, f'Session: {session_id[:8]}', (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(test_image, f'Time: {datetime.now().strftime("%H:%M:%S")}', (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', test_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # Save directly to file (bypass session manager for testing)
        sessions_dir = os.path.abspath('sessions')
        session_dir = os.path.join(sessions_dir, session_id)
        screenshots_dir = os.path.join(session_dir, 'screenshots')
        
        os.makedirs(screenshots_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"test_screenshot_{timestamp}.jpg"
        filepath = os.path.join(screenshots_dir, filename)
        
        with open(filepath, 'wb') as f:
            f.write(buffer.tobytes())
        
        # Verify file was saved
        file_exists = os.path.exists(filepath)
        
        print(f"📸 Created test screenshot: {filepath} (exists: {file_exists})")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'path': filepath,
            'file_exists': file_exists,
            'size': len(buffer.tobytes()),
            'url': f'/api/session_file/{session_id}/screenshots/{filename}'
        })
        
    except Exception as e:
        print(f"❌ Error creating test screenshot: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('error.html', 
                         error_code=404,
                         error_message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return render_template('error.html',
                         error_code=500,
                         error_message="Internal server error"), 500

if __name__ == '__main__':
    print("🚀 Starting AI Proctoring System Web Application...")
    
    # Initialize detector manager
    detector_initialized = initialize_detector_manager()
    if not detector_initialized:
        print("⚠️  Detector manager initialization failed - continuing with limited functionality")
    
    # Run the application
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    port = int(os.environ.get('PORT', 5000))
    
    print(f"🌐 Starting Flask server on port {port}")
    print(f"🔧 Debug mode: {debug_mode}")
    print(f"📁 Templates folder: {app.template_folder}")
    print(f"📁 Static folder: {app.static_folder}")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
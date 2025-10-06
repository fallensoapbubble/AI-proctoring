#!/usr/bin/env python3
"""
Simple Proctoring System - UUID Session Management

Core session management system with UUID-based folder structure.
"""

import os
import json
import uuid
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


class UUIDSessionManager:
    """Manages proctoring sessions with UUID-based folder structure."""
    
    def __init__(self, base_dir: str = "sessions"):
        """Initialize the session manager."""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        print(f"UUIDSessionManager initialized with base directory: {self.base_dir}")
    
    def _sanitize_filename(self, text: str) -> str:
        """Sanitize text for use in filenames."""
        import re
        
        # Remove or replace problematic characters
        # First, handle enum-style names
        if 'DetectionType.' in text:
            text = text.replace('DetectionType.', '')
        
        # Remove extra timestamps and duplicated parts
        # Pattern like "LIP_MOVEMENT_20251006_173156_20251006_173156_712"
        text = re.sub(r'_\d{8}_\d{6}_\d{8}_\d{6}_\d+', '', text)
        text = re.sub(r'_\d{8}_\d{6}_\d+', '', text)
        
        # Keep only alphanumeric, underscore, and hyphen
        text = re.sub(r'[^a-zA-Z0-9_-]', '_', text)
        
        # Remove multiple consecutive underscores
        text = re.sub(r'_+', '_', text)
        
        # Remove leading/trailing underscores
        text = text.strip('_')
        
        # Limit length to prevent filesystem issues
        if len(text) > 50:
            text = text[:50]
        
        # Ensure we have something
        if not text:
            text = 'unknown'
        
        return text.lower()
    
    def create_session(self) -> str:
        """Create a new session with UUID."""
        session_id = str(uuid.uuid4())
        session_dir = self.base_dir / session_id
        
        # Create session directory structure
        session_dir.mkdir(exist_ok=True)
        (session_dir / "screenshots").mkdir(exist_ok=True)
        (session_dir / "audio").mkdir(exist_ok=True)
        (session_dir / "answers").mkdir(exist_ok=True)
        
        # Create session metadata
        metadata = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "files_created": 0,
            "detections": 0,
            "answers": 0,
            "screenshots": 0,
            "audio_files": 0
        }
        
        metadata_file = session_dir / "session_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Created session: {session_id}")
        return session_id
    
    def save_screenshot(self, session_id: str, image_data: bytes, detection_type: str) -> str:
        """Save a screenshot with timestamp."""
        session_dir = self.base_dir / session_id
        if not session_dir.exists():
            raise ValueError(f"Session {session_id} not found")
        
        # Clean and sanitize detection type for filename
        clean_detection_type = self._sanitize_filename(detection_type)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"screenshot_{clean_detection_type}_{timestamp}.jpg"
        filepath = session_dir / "screenshots" / filename
        
        # Save image data (or create dummy file if data is fake)
        with open(filepath, 'wb') as f:
            if isinstance(image_data, str):
                f.write(image_data.encode())
            else:
                f.write(image_data)
        
        self._update_metadata(session_id, "screenshots", 1)
        print(f"Saved screenshot: {filename}")
        return str(filepath)
    
    def save_audio(self, session_id: str, audio_data: bytes, audio_type: str) -> str:
        """Save audio recording with timestamp."""
        session_dir = self.base_dir / session_id
        if not session_dir.exists():
            raise ValueError(f"Session {session_id} not found")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"audio_{audio_type}_{timestamp}.wav"
        filepath = session_dir / "audio" / filename
        
        # Save audio data (or create dummy file if data is fake)
        with open(filepath, 'wb') as f:
            if isinstance(audio_data, str):
                f.write(audio_data.encode())
            else:
                f.write(audio_data)
        
        self._update_metadata(session_id, "audio_files", 1)
        print(f"Saved audio: {filename}")
        return str(filepath)
    
    def save_answer(self, session_id: str, question_id: int, answer: str, question_text: str = "") -> str:
        """Save an exam answer."""
        session_dir = self.base_dir / session_id
        if not session_dir.exists():
            raise ValueError(f"Session {session_id} not found")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"answer_{question_id}_{timestamp}.json"
        filepath = session_dir / "answers" / filename
        
        answer_data = {
            "question_id": question_id,
            "question_text": question_text,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id
        }
        
        with open(filepath, 'w') as f:
            json.dump(answer_data, f, indent=2)
        
        self._update_metadata(session_id, "answers", 1)
        print(f"Saved answer: {filename}")
        return str(filepath)
    
    def save_detection(self, session_id: str, detection_data: Dict[str, Any]) -> str:
        """Save a detection event."""
        session_dir = self.base_dir / session_id
        if not session_dir.exists():
            raise ValueError(f"Session {session_id} not found")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"detection_{timestamp}.json"
        filepath = session_dir / filename
        
        detection_record = {
            **detection_data,
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id
        }
        
        with open(filepath, 'w') as f:
            json.dump(detection_record, f, indent=2)
        
        self._update_metadata(session_id, "detections", 1)
        print(f"Saved detection: {filename}")
        return str(filepath)
    
    def _update_metadata(self, session_id: str, field: str, increment: int):
        """Update session metadata."""
        session_dir = self.base_dir / session_id
        metadata_file = session_dir / "session_metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        metadata[field] = metadata.get(field, 0) + increment
        metadata["files_created"] = metadata.get("files_created", 0) + increment
        metadata["last_updated"] = datetime.now().isoformat()
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def finalize_session(self, session_id: str) -> Dict[str, int]:
        """Finalize a session and return file counts."""
        session_dir = self.base_dir / session_id
        if not session_dir.exists():
            raise ValueError(f"Session {session_id} not found")
        
        # Count files in each directory
        file_counts = {
            "screenshots": len(list((session_dir / "screenshots").glob("*"))),
            "audio": len(list((session_dir / "audio").glob("*"))),
            "answers": len(list((session_dir / "answers").glob("*"))),
            "detections": len(list(session_dir.glob("detection_*.json"))),
            "total": len(list(session_dir.rglob("*"))) - len(list(session_dir.glob("*/")))
        }
        
        # Update metadata
        metadata_file = session_dir / "session_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        metadata.update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "file_counts": file_counts
        })
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create consolidated answers file
        self._create_consolidated_answers(session_id)
        
        print(f"Finalized session {session_id}: {file_counts}")
        return file_counts
    
    def _create_consolidated_answers(self, session_id: str):
        """Create a consolidated answers file."""
        session_dir = self.base_dir / session_id
        answers_dir = session_dir / "answers"
        
        all_answers = []
        for answer_file in answers_dir.glob("*.json"):
            with open(answer_file, 'r') as f:
                answer_data = json.load(f)
                all_answers.append(answer_data)
        
        # Sort by question_id
        all_answers.sort(key=lambda x: x.get("question_id", 0))
        
        consolidated_file = session_dir / "session_answers.json"
        with open(consolidated_file, 'w') as f:
            json.dump({
                "session_id": session_id,
                "total_answers": len(all_answers),
                "answers": all_answers,
                "created_at": datetime.now().isoformat()
            }, f, indent=2)
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get session summary."""
        session_dir = self.base_dir / session_id
        if not session_dir.exists():
            raise ValueError(f"Session {session_id} not found")
        
        metadata_file = session_dir / "session_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        
        return {"error": "No metadata found"}
    
    def list_sessions(self) -> List[str]:
        """List all session IDs."""
        sessions = []
        for session_dir in self.base_dir.iterdir():
            if session_dir.is_dir():
                sessions.append(session_dir.name)
        return sorted(sessions)


class SimpleDetectionSimulator:
    """Simulates detection events for testing."""
    
    def __init__(self):
        """Initialize the detection simulator."""
        self.detection_types = [
            "face_not_visible",
            "multiple_faces",
            "looking_away",
            "mobile_device",
            "audio_anomaly",
            "suspicious_movement",
            "unauthorized_object"
        ]
        
        self.descriptions = {
            "face_not_visible": "Student's face is not clearly visible",
            "multiple_faces": "Multiple faces detected in frame",
            "looking_away": "Student looking away from screen",
            "mobile_device": "Mobile device detected in frame",
            "audio_anomaly": "Unusual audio pattern detected",
            "suspicious_movement": "Suspicious movement detected",
            "unauthorized_object": "Unauthorized object detected"
        }
        
        print("SimpleDetectionSimulator initialized")
    
    def simulate_detection(self) -> Dict[str, Any]:
        """Simulate a detection event."""
        detection_type = random.choice(self.detection_types)
        confidence = random.uniform(0.3, 0.95)
        
        return {
            "type": detection_type,
            "confidence": confidence,
            "description": self.descriptions.get(detection_type, "Unknown detection"),
            "timestamp": datetime.now().isoformat(),
            "severity": "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
        }


if __name__ == "__main__":
    print("🚀 Simple Proctoring System Test")
    print("=" * 40)
    
    # Test session manager
    manager = UUIDSessionManager()
    
    # Create a test session
    session_id = manager.create_session()
    print(f"Created session: {session_id}")
    
    # Test detection simulator
    simulator = SimpleDetectionSimulator()
    
    # Simulate some activity
    for i in range(3):
        # Save an answer
        manager.save_answer(session_id, i+1, f"Answer {i+1}", f"Question {i+1}?")
        
        # Simulate detection
        detection = simulator.simulate_detection()
        manager.save_detection(session_id, detection)
        
        # Save screenshot if high confidence
        if detection['confidence'] > 0.7:
            fake_image = f"fake_screenshot_data_{i}".encode() * 50
            manager.save_screenshot(session_id, fake_image, detection['type'])
    
    # Save audio
    fake_audio = b"fake_audio_data" * 100
    manager.save_audio(session_id, fake_audio, "exam_recording")
    
    # Finalize session
    file_counts = manager.finalize_session(session_id)
    
    # Get summary
    summary = manager.get_session_summary(session_id)
    
    print(f"\n✅ Test completed!")
    print(f"Files created: {file_counts}")
    print(f"Session status: {summary.get('status', 'unknown')}")
    
    # List all sessions
    all_sessions = manager.list_sessions()
    print(f"Total sessions: {len(all_sessions)}")
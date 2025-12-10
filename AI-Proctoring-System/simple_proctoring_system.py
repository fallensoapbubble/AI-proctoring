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
        (session_dir / "metadata").mkdir(exist_ok=True)
        
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
    
    def save_screenshot(self, session_id: str, image_data: bytes, detection_type: str, timestamp: datetime = None) -> str:
        """Save a screenshot with timestamp."""
        session_dir = self.base_dir / session_id
        if not session_dir.exists():
            raise ValueError(f"Session {session_id} not found")
        
        # Clean and sanitize detection type for filename
        clean_detection_type = self._sanitize_filename(detection_type)
        
        # Use provided timestamp or create new one
        if timestamp is None:
            timestamp = datetime.now()
        
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{clean_detection_type}_{timestamp_str}.jpg"
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
    
    def save_detection_summary(self, session_id: str, summary_data: dict) -> str:
        """Save detection summary as JSON file."""
        session_dir = self.base_dir / session_id
        if not session_dir.exists():
            raise ValueError(f"Session {session_id} not found")
        
        # Create metadata directory if it doesn't exist
        metadata_dir = session_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"detection_summary_{timestamp}.json"
        filepath = metadata_dir / filename
        
        # Save summary data as JSON
        with open(filepath, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"Saved detection summary: {filename}")
        return str(filepath)
    
    def save_audio(self, session_id: str, audio_data: bytes, audio_type: str, timestamp: datetime = None, convert_to_mp3: bool = True) -> str:
        """Save audio recording with timestamp, optionally converting to MP3."""
        session_dir = self.base_dir / session_id
        if not session_dir.exists():
            raise ValueError(f"Session {session_id} not found")
        
        # Use provided timestamp or create new one
        if timestamp is None:
            timestamp = datetime.now()
        
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Save as WAV first
        wav_filename = f"audio_{audio_type}_{timestamp_str}.wav"
        wav_filepath = session_dir / "audio" / wav_filename
        
        # Save audio data
        with open(wav_filepath, 'wb') as f:
            if isinstance(audio_data, str):
                f.write(audio_data.encode())
            else:
                f.write(audio_data)
        
        print(f"Saved audio (WAV): {wav_filename}")
        
        # Convert to MP3 if requested
        if convert_to_mp3:
            try:
                from pydub import AudioSegment
                
                mp3_filename = f"audio_{audio_type}_{timestamp_str}.mp3"
                mp3_filepath = session_dir / "audio" / mp3_filename
                
                # Load WAV and export as MP3
                audio = AudioSegment.from_wav(str(wav_filepath))
                audio.export(str(mp3_filepath), format="mp3", bitrate="128k")
                
                # Remove WAV file to save space
                wav_filepath.unlink()
                
                self._update_metadata(session_id, "audio_files", 1)
                print(f"Converted to MP3: {mp3_filename}")
                return str(mp3_filepath)
                
            except ImportError:
                print("⚠️ pydub not available, keeping WAV format")
                self._update_metadata(session_id, "audio_files", 1)
                return str(wav_filepath)
            except Exception as e:
                print(f"⚠️ MP3 conversion failed: {e}, keeping WAV format")
                self._update_metadata(session_id, "audio_files", 1)
                return str(wav_filepath)
        else:
            self._update_metadata(session_id, "audio_files", 1)
            return str(wav_filepath)
    
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
            "metadata": len(list((session_dir / "metadata").glob("*"))),
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


# Removed SimpleDetectionSimulator - no longer needed for production


if __name__ == "__main__":
    print("🚀 Simple Proctoring System")
    print("=" * 40)
    
    # Test session manager
    manager = UUIDSessionManager()
    
    # Create a test session
    session_id = manager.create_session()
    print(f"Created session: {session_id}")
    
    # Test basic functionality
    manager.save_answer(session_id, 1, "Test Answer", "Test Question?")
    
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
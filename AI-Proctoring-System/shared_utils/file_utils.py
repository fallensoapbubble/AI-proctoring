"""
File utility functions for evidence collection and storage.

This module provides utilities for managing files, directories, and evidence
storage in the AI proctoring system.
"""

import os
import shutil
import uuid
import json
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging


def ensure_directory_exists(directory_path: Union[str, Path]) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
    """
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def get_session_directory(session_id: str) -> str:
    """
    Get the directory path for a specific session.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Path to session directory
    """
    session_dir = f"sessions/{session_id}"
    ensure_directory_exists(session_dir)
    return session_dir


def save_evidence_file(
    data: Union[np.ndarray, bytes, str],
    filename: str,
    session_id: str,
    evidence_type: str = "general"
) -> str:
    """
    Save evidence data to the appropriate session directory.
    
    Args:
        data: Evidence data (image, audio, text, etc.)
        filename: Name for the evidence file
        session_id: Session identifier
        evidence_type: Type of evidence (screenshot, audio, answers, etc.)
        
    Returns:
        Path to saved evidence file
    """
    # Create evidence directory structure
    session_dir = get_session_directory(session_id)
    evidence_dir = os.path.join(session_dir, evidence_type)
    ensure_directory_exists(evidence_dir)
    
    # Generate unique filename if needed
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evidence_{timestamp}_{uuid.uuid4().hex[:8]}"
    
    file_path = os.path.join(evidence_dir, filename)
    
    try:
        if isinstance(data, np.ndarray):
            # Save image data
            cv2.imwrite(file_path, data)
        elif isinstance(data, bytes):
            # Save binary data
            with open(file_path, 'wb') as f:
                f.write(data)
        elif isinstance(data, str):
            # Save text data
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(data)
        elif isinstance(data, (dict, list)):
            # Save JSON data
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        return file_path
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to save evidence file {filename}: {e}")
        raise


def save_screenshot(frame: np.ndarray, session_id: str, event_type: str = "detection") -> str:
    """
    Save a screenshot as evidence.
    
    Args:
        frame: Video frame to save
        session_id: Session identifier
        event_type: Type of event that triggered the screenshot
        
    Returns:
        Path to saved screenshot
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
    filename = f"screenshot_{event_type}_{timestamp}.jpg"
    
    return save_evidence_file(frame, filename, session_id, "screenshots")


def save_audio_evidence(audio_data: bytes, session_id: str, event_type: str = "speech") -> str:
    """
    Save audio data as evidence.
    
    Args:
        audio_data: Audio data in bytes
        session_id: Session identifier
        event_type: Type of audio event
        
    Returns:
        Path to saved audio file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"audio_{event_type}_{timestamp}.wav"
    
    return save_evidence_file(audio_data, filename, session_id, "audio")


def save_answers(answers: Dict[str, Any], session_id: str) -> str:
    """
    Save exam answers as evidence.
    
    Args:
        answers: Dictionary containing exam answers
        session_id: Session identifier
        
    Returns:
        Path to saved answers file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"answers_{timestamp}.json"
    
    return save_evidence_file(answers, filename, session_id, "answers")


def save_detection_log(detections: List[Dict[str, Any]], session_id: str) -> str:
    """
    Save detection events log.
    
    Args:
        detections: List of detection events
        session_id: Session identifier
        
    Returns:
        Path to saved detection log
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detections_{timestamp}.json"
    
    return save_evidence_file(detections, filename, session_id, "logs")


def get_session_evidence(session_id: str) -> Dict[str, List[str]]:
    """
    Get all evidence files for a session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Dictionary mapping evidence types to lists of file paths
    """
    session_dir = get_session_directory(session_id)
    evidence = {}
    
    if not os.path.exists(session_dir):
        return evidence
    
    for item in os.listdir(session_dir):
        item_path = os.path.join(session_dir, item)
        if os.path.isdir(item_path):
            evidence[item] = []
            for file in os.listdir(item_path):
                file_path = os.path.join(item_path, file)
                if os.path.isfile(file_path):
                    evidence[item].append(file_path)
    
    return evidence


def cleanup_old_sessions(retention_days: int = 30) -> int:
    """
    Clean up session directories older than retention period.
    
    Args:
        retention_days: Number of days to retain sessions
        
    Returns:
        Number of sessions cleaned up
    """
    sessions_dir = Path("sessions")
    if not sessions_dir.exists():
        return 0
    
    cutoff_time = datetime.now().timestamp() - (retention_days * 24 * 60 * 60)
    cleaned_count = 0
    
    for session_dir in sessions_dir.iterdir():
        if session_dir.is_dir():
            # Check directory modification time
            if session_dir.stat().st_mtime < cutoff_time:
                try:
                    shutil.rmtree(session_dir)
                    cleaned_count += 1
                except Exception as e:
                    logger = logging.getLogger(__name__)
                    logger.error(f"Failed to clean up session {session_dir.name}: {e}")
    
    return cleaned_count


def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in bytes, or 0 if file doesn't exist
    """
    try:
        return os.path.getsize(file_path)
    except (OSError, FileNotFoundError):
        return 0


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f} {size_names[i]}"


def create_session_summary(session_id: str) -> Dict[str, Any]:
    """
    Create a summary of session evidence and statistics.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Dictionary containing session summary
    """
    evidence = get_session_evidence(session_id)
    session_dir = get_session_directory(session_id)
    
    summary = {
        'session_id': session_id,
        'session_directory': session_dir,
        'evidence_types': list(evidence.keys()),
        'total_files': sum(len(files) for files in evidence.values()),
        'evidence_breakdown': {},
        'total_size_bytes': 0,
        'created_time': None,
        'last_modified': None
    }
    
    # Calculate evidence breakdown and total size
    for evidence_type, files in evidence.items():
        type_size = sum(get_file_size(file) for file in files)
        summary['evidence_breakdown'][evidence_type] = {
            'file_count': len(files),
            'total_size_bytes': type_size,
            'total_size_formatted': format_file_size(type_size)
        }
        summary['total_size_bytes'] += type_size
    
    summary['total_size_formatted'] = format_file_size(summary['total_size_bytes'])
    
    # Get directory timestamps
    try:
        session_path = Path(session_dir)
        if session_path.exists():
            stat = session_path.stat()
            summary['created_time'] = datetime.fromtimestamp(stat.st_ctime).isoformat()
            summary['last_modified'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
    except Exception:
        pass
    
    return summary


def export_session_evidence(session_id: str, export_path: str) -> str:
    """
    Export all session evidence to a zip file.
    
    Args:
        session_id: Session identifier
        export_path: Path where to save the export
        
    Returns:
        Path to the exported zip file
    """
    import zipfile
    
    session_dir = get_session_directory(session_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"session_{session_id}_{timestamp}.zip"
    zip_path = os.path.join(export_path, zip_filename)
    
    ensure_directory_exists(export_path)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        session_path = Path(session_dir)
        for file_path in session_path.rglob('*'):
            if file_path.is_file():
                # Add file to zip with relative path
                arcname = file_path.relative_to(session_path.parent)
                zipf.write(file_path, arcname)
    
    return zip_path
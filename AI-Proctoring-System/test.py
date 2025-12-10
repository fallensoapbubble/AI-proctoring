import os
import cv2
import csv
import uuid
import base64
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify

# Create a Blueprint
test_bp = Blueprint('test_lab', __name__)

# CSV Configuration
CSV_FILE = 'test_dataset.csv'
IMG_DIR = 'static/test_captures'
os.makedirs(IMG_DIR, exist_ok=True)

# Initialize CSV if needed
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'image_path', 'ai_verdict', 'human_label', 'result_type', 'detection_details'])

@test_bp.route('/test_lab')
def test_lab():
    """Renders the Testing Dashboard"""
    stats = {'total': 0, 'accuracy': 0, 'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            if len(df) > 0:
                stats['total'] = len(df)
                
                # Calculate confusion matrix counts
                stats['tp'] = len(df[df['result_type'] == 'TP'])
                stats['tn'] = len(df[df['result_type'] == 'TN'])
                stats['fp'] = len(df[df['result_type'] == 'FP'])
                stats['fn'] = len(df[df['result_type'] == 'FN'])
                
                # Calculate Accuracy
                correct = stats['tp'] + stats['tn']
                stats['accuracy'] = round((correct / len(df)) * 100, 1)
        except:
            pass # Handle empty or corrupt CSV
            
    return render_template('test_lab.html', stats=stats)

@test_bp.route('/api/test/process', methods=['POST'])
def process_test_frame():
    """Uses the MAIN app's FaceGuard to analyze a frame"""
    try:
        # Import the global face_guard instance from your main app
        # We do this inside the function to avoid circular imports
        from app import face_guard
        
        # 1. Decode Image
        data = request.json['image']
        header, encoded = data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # 2. Run YOUR FaceGuard Logic
        # Note: We use the actual FaceGuard instance from app.py
        result = face_guard.process_frame(frame)
        
        # 3. Determine Verdict (Cheating vs Normal)
        # Logic: If any detections exist (other than NO_FACE if you prefer), it's cheating
        detections = result.get('detections', [])
        
        # Filter out 'NO_FACE' if you don't want to count it as cheating for testing
        # detections = [d for d in detections if d['type'] != 'NO_FACE']
        
        is_cheating = len(detections) > 0
        verdict = "CHEATING" if is_cheating else "NORMAL"
        
        # 4. Save Temporary Image for review
        temp_filename = f"temp_review_{uuid.uuid4().hex[:8]}.jpg"
        cv2.imwrite(os.path.join(IMG_DIR, temp_filename), frame)

        return jsonify({
            'status': 'success',
            'verdict': verdict,
            'details': detections,
            'image_url': f"/static/test_captures/{temp_filename}",
            'filename': temp_filename
        })
        
    except Exception as e:
        print(f"Test Lab Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@test_bp.route('/api/test/submit', methods=['POST'])
def submit_label():
    """Saves the result to CSV"""
    data = request.json
    
    ai_verdict = data['ai_verdict']      # "CHEATING" or "NORMAL"
    human_label = data['human_label']    # "CHEATING" or "NORMAL"
    
    # Calculate Result Type for Metrics
    if ai_verdict == "CHEATING" and human_label == "CHEATING": result = "TP"
    elif ai_verdict == "NORMAL" and human_label == "NORMAL": result = "TN"
    elif ai_verdict == "CHEATING" and human_label == "NORMAL": result = "FP"
    elif ai_verdict == "NORMAL" and human_label == "CHEATING": result = "FN"
    
    # Save to CSV
    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data['filename'],
            ai_verdict,
            human_label,
            result,
            str(data['details']) # Save the raw detection reasons
        ])
        
    return jsonify({'status': 'saved'})
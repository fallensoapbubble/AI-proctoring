import os
import cv2
import csv
import uuid
import base64
import numpy as np
import pandas as pd
import io
import json
import threading
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify

# --- Imports for Advanced Analytics ---
import matplotlib
matplotlib.use('Agg') # Crucial: Use 'Agg' backend for non-GUI server environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_recall_fscore_support
# ----------------------------------------

# Create a Blueprint
test_bp = Blueprint('test', __name__) 

# --- Configuration ---
CSV_FILE = 'test_dataset.csv'
IMG_DIR = 'static/test_captures'

# Thread lock for CSV writing to prevent corruption
csv_lock = threading.Lock()

# Ensure directories exist
os.makedirs(IMG_DIR, exist_ok=True)

# Define Detection Labels (Must match FaceGuard output types)
DETECTION_LABELS = [
    'MULTIPLE_FACES', 
    'SPOOF_DETECTED', 
    'SPEECH_DETECTED', 
    'GAZE_AWAY', 
    'NO_FACE',
    'MOBILE_DETECTED'
]

# Initialize CSV with headers if it doesn't exist
if not os.path.exists(CSV_FILE):
    header = ['timestamp', 'image_path', 'ai_verdict', 'human_label', 'result_type', 'detection_details']
    for label in DETECTION_LABELS:
        header.append(f'ai_conf_{label}')    # AI confidence score (float)
        header.append(f'human_bool_{label}') # Human's True/False assessment (1 or 0)
        
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

# --- Utility Functions ---

def get_base64_img(fig):
    """Saves a matplotlib figure to a base64 encoded string for embedding in HTML."""
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    finally:
        plt.close(fig) # Explicitly close the specific figure to free memory

# --- Routes ---

@test_bp.route('/test_lab')
def test_lab():
    """Renders the Testing Dashboard and calculates live summary stats."""
    stats = {'total': 0, 'accuracy': 0, 'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    
    if os.path.exists(CSV_FILE):
        try:
            # Use lock when reading just to be safe vs writes
            with csv_lock:
                try:
                    df = pd.read_csv(CSV_FILE)
                except pd.errors.EmptyDataError:
                    df = pd.DataFrame() # Handle empty CSV case

            # Filter out rows where human didn't label (if any)
            if not df.empty and 'human_label' in df.columns:
                df = df.dropna(subset=['human_label']) 

                if len(df) > 0:
                    stats['total'] = len(df)
                    
                    # Calculate counts based on Overall Verdict
                    tp = len(df[(df['ai_verdict'] == 'CHEATING') & (df['human_label'] == 'CHEATING')])
                    tn = len(df[(df['ai_verdict'] == 'NORMAL') & (df['human_label'] == 'NORMAL')])
                    fp = len(df[(df['ai_verdict'] == 'CHEATING') & (df['human_label'] == 'NORMAL')])
                    fn = len(df[(df['ai_verdict'] == 'NORMAL') & (df['human_label'] == 'CHEATING')])

                    stats['tp'] = tp
                    stats['tn'] = tn
                    stats['fp'] = fp
                    stats['fn'] = fn
                    
                    # Calculate Overall Accuracy
                    correct = tp + tn
                    stats['accuracy'] = round((correct / len(df)) * 100, 1)
        except Exception as e:
            print(f"Error loading stats in test_lab: {e}") 
            # Proceed with zeroed stats on error
            
    return render_template('test_lab.html', stats=stats, detection_labels=DETECTION_LABELS)


@test_bp.route('/api/test/process', methods=['POST'])
def process_test_frame():
    try:
        # Import global objects inside function to avoid circular imports
        from app import face_guard, detector_manager
        
        # --- FIX: Extract data from request ---
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'status': 'error', 'message': 'No image provided'})
            
        encoded = data['image']
        
        # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
        if "," in encoded:
            encoded = encoded.split(",")[1]
        # --------------------------------------

        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'status': 'error', 'message': 'Failed to decode image'})

        # 1. Run FaceGuard (Standard Face checks)
        # Ensure face_guard is available
        if not face_guard:
             return jsonify({'status': 'error', 'message': 'FaceGuard not initialized'})

        result = face_guard.process_frame(frame)
        detections = result.get('detections', [])

        # 2. Run Mobile/Object Detection (via detector_manager)
        if detector_manager and detector_manager.is_running:
            mobile_detector = detector_manager.detectors.get('mobile')
            if mobile_detector:
                try:
                    mobile_event = mobile_detector.detect_mobile_devices(frame)
                    # Handle both single event object or list of events
                    if mobile_event:
                        # If it returns a list, take the highest confidence one or loop
                        if isinstance(mobile_event, list):
                            for event in mobile_event:
                                detections.append({
                                    'type': 'MOBILE_DETECTED',
                                    'confidence': float(event.confidence)
                                })
                        # If it returns a single detection object
                        elif hasattr(mobile_event, 'confidence'):
                             detections.append({
                                'type': 'MOBILE_DETECTED',
                                'confidence': float(mobile_event.confidence)
                            })
                except Exception as e:
                    print(f"Mobile detection error: {e}")

        # 3. Determine Overall Verdict
        # Filter out NO_FACE for the "Cheating" verdict logic
        relevant_detections = [d for d in detections if d['type'] != 'NO_FACE']
        is_cheating = len(relevant_detections) > 0
        verdict = "CHEATING" if is_cheating else "NORMAL"
        
        # 4. Extract confidences
        detection_data = {}
        for label in DETECTION_LABELS:
            # Find detection with highest confidence for this label
            matching_dets = [d for d in detections if d['type'] == label]
            if matching_dets:
                # Get max confidence if multiple of same type
                max_conf = max(float(d['confidence']) for d in matching_dets)
                detection_data[label] = max_conf
            else:
                detection_data[label] = 0.0

        # 5. Save Temporary Image for the Review UI
        temp_filename = f"temp_review_{uuid.uuid4().hex[:8]}.jpg"
        cv2.imwrite(os.path.join(IMG_DIR, temp_filename), frame)

        return jsonify({
            'status': 'success',
            'verdict': verdict,
            'details': detections, 
            'image_url': f"/static/test_captures/{temp_filename}",
            'filename': temp_filename,
            'detection_data': detection_data 
        })
        
    except ImportError:
        return jsonify({'status': 'error', 'message': 'Could not import face_guard. Is app.py running?'})
    except Exception as e:
        print(f"Test Lab Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@test_bp.route('/api/test/submit', methods=['POST'])
def submit_label():
    """Saves the Ground Truth vs AI Result to CSV."""
    try:
        data = request.json
        if not data:
            return jsonify({'status': 'error', 'message': 'No data provided'})
            
        ai_verdict = data.get('ai_verdict', 'UNKNOWN')
        human_label = data.get('human_label', 'UNKNOWN')
        
        # Calculate Logic Result Type
        if ai_verdict == "CHEATING" and human_label == "CHEATING": result_type = "TP"
        elif ai_verdict == "NORMAL" and human_label == "NORMAL": result_type = "TN"
        elif ai_verdict == "CHEATING" and human_label == "NORMAL": result_type = "FP"
        elif ai_verdict == "NORMAL" and human_label == "CHEATING": result_type = "FN"
        else: result_type = "UNKNOWN"

        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data.get('filename', ''),
            ai_verdict,
            human_label,
            result_type,
            str(data.get('details', []))
        ]
        
        # Add individual data points (AI Confidence vs Human Boolean)
        human_label_data = data.get('human_label_data', {})
        detection_data = data.get('detection_data', {})
        
        for label in DETECTION_LABELS:
            # AI Confidence
            ai_conf = detection_data.get(label, 0.0)
            row.append(ai_conf)
            
            # Human Boolean (Robust parsing)
            h_val = human_label_data.get(label)
            # Handle boolean types, integers, and strings
            if isinstance(h_val, bool):
                human_bool = 1 if h_val else 0
            else:
                human_bool = 1 if str(h_val).lower() in ['1', 'true', 'yes'] else 0
            row.append(human_bool)

        # Write to CSV safely
        with csv_lock:
            with open(CSV_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
        # Clean up temporary image
        filename = data.get('filename')
        if filename:
            temp_path = os.path.join(IMG_DIR, filename)
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass # File might be in use, skip
                
        return jsonify({'status': 'saved'})
    except Exception as e:
        print(f"Error saving label: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@test_bp.route('/analytics')
def show_analytics():
    """Generates advanced performance analytics/plots."""
    
    # Validation
    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        return render_template('analytics.html', error="No test data available.")

    try:
        with csv_lock:
            df = pd.read_csv(CSV_FILE)
            
        if 'human_label' in df.columns:
            df = df.dropna(subset=['human_label']) 
            
        if len(df) < 2:
            return render_template('analytics.html', error="Need at least 2 labeled samples to generate graphs.")
    except Exception as e:
        return render_template('analytics.html', error=f"Error reading CSV: {e}")

    # Clean up any open plots from previous requests
    plt.close('all')

    # --- 1. Overall Confusion Matrix (Verdict Level) ---
    try:
        cm_data = pd.crosstab(df['human_label'], df['ai_verdict'], rownames=['True'], colnames=['Predicted'])
        # Ensure all columns/rows exist for correct matrix shape
        for cat in ['NORMAL', 'CHEATING']:
            if cat not in cm_data.columns: cm_data[cat] = 0
            if cat not in cm_data.index: cm_data.loc[cat] = 0
            
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Overall System Verdict')
        overall_cm_plot = get_base64_img(fig)
    except Exception as e:
        print(f"Error generating overall CM: {e}")
        overall_cm_plot = None

    # --- 2. Per-Label Metrics ---
    label_analytics = []
    
    for label in DETECTION_LABELS:
        ai_col = f'ai_conf_{label}'
        human_col = f'human_bool_{label}'
        
        # Skip if columns missing
        if ai_col not in df.columns or human_col not in df.columns:
            continue
            
        y_true = df[human_col].astype(int).tolist()
        y_score = df[ai_col].fillna(0).tolist()
        # Create binary prediction (Threshold 0.5)
        y_pred = [1 if x > 0.5 else 0 for x in y_score]
        
        # Check if we have data
        if len(y_true) == 0: continue

        # --- A. Confusion Matrix for this Label ---
        cm_plot = None
        try:
            # Only generate if we have both 0 and 1 class represented roughly, or handle exception
            label_cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            cm_fig, cm_ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(label_cm, annot=True, fmt='d', cmap='viridis', ax=cm_ax,
                        xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
            cm_ax.set_title(f'{label}')
            cm_plot = get_base64_img(cm_fig)
        except Exception as e:
            pass

        # --- B. ROC Curve ---
        roc_plot = None
        roc_auc_val = 'N/A'
        try:
            # ROC requires at least one positive and one negative sample in y_true
            if len(np.unique(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc_val = auc(fpr, tpr)
                
                roc_fig, roc_ax = plt.subplots(figsize=(4, 3))
                roc_ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC={roc_auc_val:.2f}')
                roc_ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
                roc_ax.legend(loc="lower right")
                roc_ax.set_title(f'ROC: {label}')
                roc_plot = get_base64_img(roc_fig)
        except Exception as e:
            # print(f"ROC Error for {label}: {e}") # Suppress noise
            pass

        # --- C. Scalar Metrics ---
        # zero_division=0 handles cases where model predicts 0 positives
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        acc = accuracy_score(y_true, y_pred)

        label_analytics.append({
            'label': label,
            'cm_plot': cm_plot,
            'roc_plot': roc_plot,
            'auc': f'{roc_auc_val:.2f}' if isinstance(roc_auc_val, float) else roc_auc_val,
            'precision': f'{p:.3f}',
            'recall': f'{r:.3f}',
            'f1': f'{f1:.3f}',
            'accuracy': f'{acc:.3f}',
            'total_samples': len(y_true)
        })

    return render_template('analytics.html', 
        overall_cm_plot=overall_cm_plot,
        label_analytics=label_analytics,
        detection_labels=DETECTION_LABELS
    )
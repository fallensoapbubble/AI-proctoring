/**
 * Exam JavaScript - Enhanced with screen capture and audio recording
 * Handles video streaming, screen capture, audio recording, and proctoring features
 */

class ExamProctoring {
    constructor() {
        this.video = null;
        this.canvas = null;
        this.context = null;
        this.stream = null;
        this.sessionId = null;
        this.isRecording = false;
        this.frameInterval = null;
        this.screenCaptureInterval = null;
        this.audioRecording = false;
        
        // Configuration
        this.FRAME_RATE = 2; // frames per second for video analysis
        this.SCREEN_CAPTURE_RATE = 5; // seconds between screen captures
        
        // Detection tracking
        this.detectionCount = 0;
        this.lastAlert = null;
        
        this.init();
    }
    
    async init() {
        console.log('🚀 Initializing Enhanced Exam Proctoring System');
        
        // Get session ID from page
        this.sessionId = document.getElementById('session-id')?.textContent || 'unknown';
        
        // Initialize video elements
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        
        if (this.canvas) {
            this.context = this.canvas.getContext('2d');
        }
        
        // Initialize camera
        await this.initializeCamera();
        
        // Initialize audio recording
        await this.initializeAudio();
        
        // Initialize screen capture
        this.initializeScreenCapture();
        
        // Start proctoring
        this.startProctoring();
        
        // Setup event handlers
        this.setupEventHandlers();
        
        console.log('✅ Enhanced Exam Proctoring System initialized');
    }
    
    async initializeCamera() {
        try {
            console.log('📹 Requesting camera access...');
            
            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    frameRate: { ideal: 30 }
                },
                audio: false // We'll handle audio separately
            };
            
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            
            if (this.video) {
                this.video.srcObject = this.stream;
                this.video.play();
                
                this.video.addEventListener('loadedmetadata', () => {
                    if (this.canvas) {
                        this.canvas.width = this.video.videoWidth;
                        this.canvas.height = this.video.videoHeight;
                    }
                    console.log('✅ Camera initialized successfully');
                });
            }
            
        } catch (error) {
            console.error('❌ Camera initialization failed:', error);
            this.showError('Camera access denied. Please allow camera access and refresh the page.');
        }
    }
    
    async initializeAudio() {
        try {
            console.log('🎤 Initializing audio recording...');
            
            // Check if audio recording is supported
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                console.warn('⚠️ Audio recording not supported in this browser');
                return;
            }
            
            // Request microphone access
            const audioStream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: 44100
                } 
            });
            
            console.log('✅ Microphone access granted');
            
            // Start server-side audio recording
            const response = await fetch(`/api/start_audio_recording/${this.sessionId}`);
            const result = await response.json();
            
            if (result.success) {
                this.audioRecording = true;
                console.log('✅ Server-side audio recording started');
                this.updateStatus('Audio recording active');
            } else {
                console.warn('⚠️ Server-side audio recording failed:', result.error);
            }
            
            // Stop the test stream (server handles actual recording)
            audioStream.getTracks().forEach(track => track.stop());
            
        } catch (error) {
            console.error('❌ Audio initialization failed:', error);
            console.log('ℹ️ Continuing without audio recording');
        }
    }
    
    initializeScreenCapture() {
        console.log('🖥️ Initializing screen capture monitoring...');
        
        // Start periodic screen capture
        this.screenCaptureInterval = setInterval(() => {
            this.captureScreen();
        }, this.SCREEN_CAPTURE_RATE * 1000);
        
        console.log('✅ Screen capture monitoring started');
    }
    
    async captureScreen() {
        try {
            const response = await fetch('/api/capture_screen');
            const result = await response.json();
            
            if (result.success) {
                console.log('📸 Screen captured successfully');
                
                // Optionally display screen capture status
                this.updateStatus('Screen monitoring active');
                
                // You could display the screenshot in a small preview if needed
                // this.displayScreenPreview(result.screenshot);
                
            } else {
                console.warn('⚠️ Screen capture failed:', result.error);
            }
            
        } catch (error) {
            console.error('❌ Screen capture error:', error);
        }
    }
    
    startProctoring() {
        if (!this.video || !this.canvas) {
            console.error('❌ Video or canvas not available for proctoring');
            return;
        }
        
        console.log('🔍 Starting proctoring surveillance...');
        
        // Start frame processing
        this.frameInterval = setInterval(() => {
            this.processFrame();
        }, 1000 / this.FRAME_RATE);
        
        this.isRecording = true;
        this.updateStatus('Proctoring active - You are being monitored');
    }
    
    stopProctoring() {
        console.log('⏹️ Stopping proctoring...');
        
        this.isRecording = false;
        
        // Clear intervals
        if (this.frameInterval) {
            clearInterval(this.frameInterval);
            this.frameInterval = null;
        }
        
        if (this.screenCaptureInterval) {
            clearInterval(this.screenCaptureInterval);
            this.screenCaptureInterval = null;
        }
        
        // Stop camera stream
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
        }
        
        this.updateStatus('Proctoring stopped');
    }
    
    async processFrame() {
        if (!this.isRecording || !this.video || !this.canvas || !this.context) {
            return;
        }
        
        try {
            // Draw current video frame to canvas
            this.context.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            
            // Convert canvas to base64
            const frameData = this.canvas.toDataURL('image/jpeg', 0.8);
            
            // Send frame to server for analysis
            const response = await fetch('/api/process_frame', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    frame: frameData,
                    session_id: this.sessionId,
                    timestamp: new Date().toISOString()
                })
            });
            
            const result = await response.json();
            
            if (result.error) {
                console.error('❌ Frame processing error:', result.error);
                return;
            }
            
            // Handle detection results
            if (result.alert) {
                this.handleAlert(result.alert);
            }
            
            // Update detection count
            if (result.has_detections) {
                this.detectionCount += result.detections_count || 1;
                this.updateDetectionCounter();
            }
            
            // Update status
            if (result.frame_processed) {
                this.updateStatus('Monitoring active - Stay focused on the exam');
            }
            
        } catch (error) {
            console.error('❌ Frame processing failed:', error);
        }
    }
    
    handleAlert(alert) {
        console.warn('🚨 Detection Alert:', alert);
        
        // Prevent spam alerts
        const now = Date.now();
        if (this.lastAlert && (now - this.lastAlert) < 3000) {
            return;
        }
        this.lastAlert = now;
        
        // Show alert to user
        this.showAlert(alert);
        
        // Update UI
        this.flashWarning();
    }
    
    showAlert(alert) {
        const alertMessages = {
            'GAZE_AWAY': 'Please look at the screen and focus on the exam',
            'MOBILE_DETECTED': 'Mobile device detected - please put away all electronic devices',
            'SPOOF_DETECTED': 'Please ensure you are taking the exam yourself',
            'MULTIPLE_FACES': 'Multiple people detected - only you should be visible',
            'SPEECH_DETECTED': 'Speaking detected - please remain silent during the exam',
            'NO_FACE': 'Please stay visible in the camera view'
        };
        
        const message = alertMessages[alert.type] || 'Suspicious activity detected';
        
        // Create alert element
        const alertDiv = document.createElement('div');
        alertDiv.className = 'proctoring-alert';
        alertDiv.innerHTML = `
            <div class="alert-content">
                <span class="alert-icon">⚠️</span>
                <span class="alert-message">${message}</span>
                <span class="alert-confidence">Confidence: ${(alert.confidence * 100).toFixed(0)}%</span>
            </div>
        `;
        
        // Add to page
        document.body.appendChild(alertDiv);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.parentNode.removeChild(alertDiv);
            }
        }, 5000);
        
        // Play alert sound (if available)
        this.playAlertSound();
    }
    
    flashWarning() {
        const video = document.getElementById('video');
        if (video) {
            video.style.border = '3px solid red';
            setTimeout(() => {
                video.style.border = '1px solid #ddd';
            }, 1000);
        }
    }
    
    playAlertSound() {
        try {
            // Create a simple beep sound
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();
            
            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);
            
            oscillator.frequency.value = 800;
            oscillator.type = 'sine';
            
            gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
            
            oscillator.start(audioContext.currentTime);
            oscillator.stop(audioContext.currentTime + 0.5);
        } catch (error) {
            console.log('Audio alert not available');
        }
    }
    
    updateStatus(message) {
        const statusElement = document.getElementById('proctoring-status');
        if (statusElement) {
            statusElement.textContent = message;
            statusElement.className = 'status-active';
        }
    }
    
    updateDetectionCounter() {
        const counterElement = document.getElementById('detection-counter');
        if (counterElement) {
            counterElement.textContent = `Detections: ${this.detectionCount}`;
        }
    }
    
    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'proctoring-error';
        errorDiv.innerHTML = `
            <div class="error-content">
                <span class="error-icon">❌</span>
                <span class="error-message">${message}</span>
            </div>
        `;
        
        document.body.appendChild(errorDiv);
        
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.parentNode.removeChild(errorDiv);
            }
        }, 10000);
    }
    
    setupEventHandlers() {
        // Handle page visibility changes
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                console.warn('⚠️ Page hidden - potential tab switching');
                this.handleAlert({
                    type: 'BROWSER_TAB_SWITCH',
                    message: 'Tab switching detected',
                    confidence: 0.8
                });
            }
        });
        
        // Handle window focus changes
        window.addEventListener('blur', () => {
            console.warn('⚠️ Window lost focus');
        });
        
        window.addEventListener('focus', () => {
            console.log('✅ Window regained focus');
        });
        
        // Handle exam submission
        const submitButton = document.getElementById('submit-exam');
        if (submitButton) {
            submitButton.addEventListener('click', () => {
                this.stopProctoring();
            });
        }
        
        // Prevent right-click context menu
        document.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            this.handleAlert({
                type: 'SUSPICIOUS_ACTIVITY',
                message: 'Right-click disabled during exam',
                confidence: 0.5
            });
        });
        
        // Prevent certain keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Prevent F12, Ctrl+Shift+I, Ctrl+U, etc.
            if (e.key === 'F12' || 
                (e.ctrlKey && e.shiftKey && e.key === 'I') ||
                (e.ctrlKey && e.key === 'u')) {
                e.preventDefault();
                this.handleAlert({
                    type: 'SUSPICIOUS_ACTIVITY',
                    message: 'Developer tools access attempted',
                    confidence: 0.9
                });
            }
        });
    }
    
    // Public methods for external control
    pauseProctoring() {
        this.isRecording = false;
        this.updateStatus('Proctoring paused');
    }
    
    resumeProctoring() {
        this.isRecording = true;
        this.updateStatus('Proctoring resumed');
    }
    
    getStatus() {
        return {
            isRecording: this.isRecording,
            audioRecording: this.audioRecording,
            detectionCount: this.detectionCount,
            sessionId: this.sessionId
        };
    }
}

// Initialize when page loads
let examProctoring = null;

document.addEventListener('DOMContentLoaded', () => {
    examProctoring = new ExamProctoring();
});

// Export for external access
window.ExamProctoring = ExamProctoring;
window.examProctoring = examProctoring;
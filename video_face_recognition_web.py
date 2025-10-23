"""
Web-based Video Face Recognition with Real-time Streaming
Access the processed video stream via web browser on Azure server
"""
import cv2
import numpy as np
import insightface
import time
import warnings
import os
import sys
from typing import List, Tuple
from flask import Flask, Response, render_template_string
import threading

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.shared.config import FACE_RECOGNITION_CONFIG, INTRUSION_CONFIG
from src.shared.models import FaceDatabase, cosine_similarity
from src.face_recognition.config.alignment_config import AlignmentConfig

# Intrusion detection imports
from src.alerts.alert_manager import AlertManager
from src.logging.intrusion_logger import IntrusionLogger
from src.detection.intrusion_detector import IntrusionDetector

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

class WebVideoFaceRecognizer:
    """Face recognition system with web streaming"""
    
    def __init__(self, database_path: str = 'data/combined_face_database.pkl'):
        print("🚀 Initializing Web Video Face Recognition System...")
        
        # Initialize InsightFace
        try:
            self.app_insight = insightface.app.FaceAnalysis(
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            det_size = FACE_RECOGNITION_CONFIG.get('det_size', (640, 640))
            self.app_insight.prepare(ctx_id=0, det_size=det_size)
            print("✅ InsightFace initialized (GPU-accelerated)")
        except Exception as e:
            print(f"❌ Error initializing InsightFace: {e}")
            raise
        
        # Initialize database
        self.face_db = FaceDatabase(database_path)
        
        # Initialize alignment config
        self.alignment_config = AlignmentConfig()
        
        # Initialize intrusion detection components
        self.alert_manager = AlertManager(
            max_alerts=INTRUSION_CONFIG['max_concurrent_alerts'],
            cooldown_seconds=INTRUSION_CONFIG['cooldown_period_seconds']
        )
        self.intrusion_logger = IntrusionLogger(
            log_directory=INTRUSION_CONFIG['log_directory']
        )
        self.intrusion_detector = IntrusionDetector(
            alert_manager=self.alert_manager,
            logger=self.intrusion_logger,
            config=INTRUSION_CONFIG
        )
        print("✅ Intrusion detection system initialized!")
        
        # Video capture
        self.cap = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'faces_detected': 0,
            'faces_recognized': 0,
            'unknown_faces': 0,
            'processing_time': 0
        }
        
        print("✅ Web Video Face Recognition System ready!")
    
    def start_camera(self, camera_source=0):
        """Start camera capture"""
        try:
            self.cap = cv2.VideoCapture(camera_source)
            if not self.cap.isOpened():
                raise Exception(f"Could not open camera {camera_source}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"📹 Camera started (source: {camera_source})")
            return True
        except Exception as e:
            print(f"❌ Error starting camera: {e}")
            return False
    
    def detect_faces(self, frame):
        """Detect faces in frame"""
        try:
            faces = self.app_insight.get(frame)
            return faces
        except Exception as e:
            print(f"❌ Error detecting faces: {e}")
            return []
    
    def recognize_faces(self, faces):
        """Recognize faces and return results"""
        results = []
        
        if not faces:
            return results
        
        try:
            # Get all known embeddings
            known_embeddings = self.face_db.get_all_embeddings()
            
            for face in faces:
                if face.embedding is None:
                    continue
                
                # Find best match
                best_match = None
                best_similarity = 0
                
                for person_id, name, embeddings in known_embeddings:
                    for embedding in embeddings:
                        similarity = cosine_similarity(face.embedding, embedding)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = (person_id, name)
                
                # Determine if recognized
                threshold = FACE_RECOGNITION_CONFIG['similarity_threshold']
                if best_similarity >= threshold and best_match:
                    person_id, name = best_match
                    results.append((person_id, name, face.bbox, best_similarity))
                else:
                    results.append(("Unknown", "Unknown", face.bbox, best_similarity))
        
        except Exception as e:
            print(f"❌ Error recognizing faces: {e}")
        
        return results
    
    def process_video_stream(self):
        """Process video stream in background thread"""
        while True:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.1)
                continue
            
            ret, frame = self.cap.read()
            if not ret:
                print("⚠️ Failed to read frame")
                time.sleep(0.1)
                continue
            
            start_time = time.time()
            
            # Detect and recognize faces
            faces = self.detect_faces(frame)
            results = self.recognize_faces(faces)
            
            # INTRUSION DETECTION - Check for unknown persons
            new_alerts = self.intrusion_detector.process_results(
                results, frame, time.time()
            )
            
            # Update stats
            self.stats['faces_detected'] += len(faces)
            self.stats['faces_recognized'] += len([r for r in results if r[0] != "Unknown"])
            self.stats['unknown_faces'] += len([r for r in results if r[0] == "Unknown"])
            
            # Draw results on frame
            self.draw_results(frame, results)
            
            # Update current frame
            with self.frame_lock:
                self.current_frame = frame.copy()
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.stats['processing_time'] = processing_time
            
            # Control frame rate
            target_fps = 30
            sleep_time = max(0, (1.0 / target_fps) - processing_time)
            time.sleep(sleep_time)
    
    def draw_results(self, frame, results):
        """Draw recognition results on frame"""
        for person_id, name, bbox, confidence in results:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Choose color based on recognition
            if person_id == "Unknown":
                color = (0, 0, 255)  # Red for unknown
                label = f"Unknown ({confidence:.2f})"
            else:
                color = (0, 255, 0)  # Green for known
                label = f"{name} ({confidence:.2f})"
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def get_frame(self):
        """Get current frame for streaming"""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None

# Global recognizer instance
recognizer = None

def generate_frames():
    """Generate frames for video streaming"""
    while True:
        if recognizer is None:
            time.sleep(0.1)
            continue
        
        frame = recognizer.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(1/30)  # 30 FPS

@app.route('/')
def index():
    """Main page with video stream and intrusion alerts"""
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hostel Intrusion Detection System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .main-container {
            display: flex;
            min-height: 100vh;
            gap: 20px;
            padding: 20px;
        }
        
        .video-section {
            flex: 0.65;
            background: #fff;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            position: relative;
        }
        
        .alert-panel {
            flex: 0.35;
            background: #fff;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            overflow-y: auto;
        }
        
        .status-bar {
            position: absolute;
            top: 20px;
            right: 20px;
            background: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            font-weight: bold;
            z-index: 10;
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        }
        
        .status-bar.alert {
            background: #f44336;
            animation: pulse 1s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .video-container {
            position: relative;
            width: 100%;
            height: 500px;
            border-radius: 10px;
            overflow: hidden;
            background: #000;
        }
        
        .video-container img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        .alert-panel h2 {
            color: #333;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .alert-count {
            background: #4CAF50;
            color: white;
            padding: 10px 15px;
            border-radius: 20px;
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .alert-count.has-alerts {
            background: #f44336;
            animation: pulse 1s infinite;
        }
        
        .alert-list {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .alert-item {
            background: #f8f9fa;
            border-left: 4px solid #f44336;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .alert-item.resolved {
            border-left-color: #4CAF50;
            background: #e8f5e8;
        }
        
        .alert-time {
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }
        
        .alert-confidence {
            font-size: 14px;
            color: #333;
            margin-bottom: 5px;
        }
        
        .alert-severity {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
        }
        
        .severity-high {
            background: #ffebee;
            color: #c62828;
        }
        
        .severity-medium {
            background: #fff3e0;
            color: #ef6c00;
        }
        
        .severity-low {
            background: #e8f5e8;
            color: #2e7d32;
        }
        
        .controls {
            margin-top: 20px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .btn-primary {
            background: #2196F3;
            color: white;
        }
        
        .btn-primary:hover {
            background: #1976D2;
            transform: translateY(-2px);
        }
        
        .btn-danger {
            background: #f44336;
            color: white;
        }
        
        .btn-danger:hover {
            background: #d32f2f;
            transform: translateY(-2px);
        }
        
        .btn-success {
            background: #4CAF50;
            color: white;
        }
        
        .btn-success:hover {
            background: #388E3C;
            transform: translateY(-2px);
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 20px;
        }
        
        .stat-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .stat-number {
            font-size: 24px;
            font-weight: bold;
            color: #2196F3;
        }
        
        .stat-label {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
        
        .no-alerts {
            text-align: center;
            color: #666;
            font-style: italic;
            padding: 40px 20px;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #2196F3;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @media (max-width: 768px) {
            .main-container {
                flex-direction: column;
                padding: 10px;
            }
            
            .video-section, .alert-panel {
                flex: 1;
            }
            
            .video-container {
                height: 300px;
            }
        }
    </style>
</head>
<body>
    <div class="main-container">
        <div class="video-section">
            <div class="status-bar" id="statusBar">SECURE</div>
            <div class="video-container">
                <img src="/video_feed" alt="Video Stream" id="videoStream">
            </div>
        </div>
        
        <div class="alert-panel">
            <h2>🚨 Intrusion Alerts</h2>
            <div class="alert-count" id="alertCount">0 Active Alerts</div>
            
            <div class="alert-list" id="alertList">
                <div class="loading">
                    <div class="spinner"></div>
                    Loading alerts...
                </div>
            </div>
            
            <div class="controls">
                <button class="btn btn-danger" onclick="clearAllAlerts()">Clear All Alerts</button>
                <button class="btn btn-primary" onclick="toggleSound()">🔊 Sound On</button>
                <button class="btn btn-success" onclick="exportLogs()">📊 Export Logs</button>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number" id="totalIntrusions">0</div>
                    <div class="stat-label">Total Intrusions</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="detectionStats">0</div>
                    <div class="stat-label">Active Detections</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let soundEnabled = true;
        let alertSound = null;
        
        // Initialize alert sound
        function initAlertSound() {
            try {
                // Create a simple beep sound using Web Audio API
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                alertSound = () => {
                    const oscillator = audioContext.createOscillator();
                    const gainNode = audioContext.createGain();
                    
                    oscillator.connect(gainNode);
                    gainNode.connect(audioContext.destination);
                    
                    oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
                    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
                    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
                    
                    oscillator.start(audioContext.currentTime);
                    oscillator.stop(audioContext.currentTime + 0.5);
                };
            } catch (e) {
                console.log('Audio not supported');
            }
        }
        
        // Update alerts every 2 seconds
        function updateAlerts() {
            fetch('/api/alerts')
                .then(response => response.json())
                .then(data => {
                    updateUI(data);
                    if (data.count > 0 && soundEnabled) {
                        playAlertSound();
                    }
                })
                .catch(error => {
                    console.error('Error fetching alerts:', error);
                });
        }
        
        // Update stats
        function updateStats() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('totalIntrusions').textContent = data.total_intrusions || 0;
                    document.getElementById('detectionStats').textContent = data.detection_stats?.active_detections || 0;
                })
                .catch(error => {
                    console.error('Error fetching stats:', error);
                });
        }
        
        // Update UI with alert data
        function updateUI(data) {
            const alertCount = document.getElementById('alertCount');
            const alertList = document.getElementById('alertList');
            const statusBar = document.getElementById('statusBar');
            
            // Update alert count
            alertCount.textContent = `${data.count} Active Alert${data.count !== 1 ? 's' : ''}`;
            alertCount.className = data.count > 0 ? 'alert-count has-alerts' : 'alert-count';
            
            // Update status bar
            if (data.count > 0) {
                statusBar.textContent = 'ALERT';
                statusBar.className = 'status-bar alert';
            } else {
                statusBar.textContent = 'SECURE';
                statusBar.className = 'status-bar';
            }
            
            // Update alert list
            if (data.alerts && data.alerts.length > 0) {
                alertList.innerHTML = data.alerts.map(alert => `
                    <div class="alert-item">
                        <div class="alert-time">${new Date(alert.timestamp).toLocaleString()}</div>
                        <div class="alert-confidence">Confidence: ${(alert.confidence * 100).toFixed(1)}%</div>
                        <div class="alert-severity severity-${alert.severity || 'low'}">${alert.severity || 'low'}</div>
                        ${alert.image_path ? `<div style="margin-top: 10px;"><img src="/intrusion_image/${alert.image_path.split('/').pop()}" style="width: 100%; max-width: 200px; border-radius: 5px;"></div>` : ''}
                    </div>
                `).join('');
            } else {
                alertList.innerHTML = '<div class="no-alerts">No active alerts</div>';
            }
        }
        
        // Clear all alerts
        function clearAllAlerts() {
            fetch('/api/clear_alerts', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        updateAlerts();
                        updateStats();
                    }
                })
                .catch(error => {
                    console.error('Error clearing alerts:', error);
                });
        }
        
        // Toggle sound
        function toggleSound() {
            soundEnabled = !soundEnabled;
            const button = event.target;
            button.textContent = soundEnabled ? '🔊 Sound On' : '🔇 Sound Off';
            button.className = soundEnabled ? 'btn btn-primary' : 'btn btn-danger';
        }
        
        // Play alert sound
        function playAlertSound() {
            if (alertSound && soundEnabled) {
                alertSound();
            }
        }
        
        // Export logs
        function exportLogs() {
            window.open('/api/export_logs', '_blank');
        }
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            initAlertSound();
            updateAlerts();
            updateStats();
            
            // Update every 2 seconds
            setInterval(updateAlerts, 2000);
            setInterval(updateStats, 5000);
        });
    </script>
</body>
</html>
    ''')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/alerts')
def get_alerts():
    """Get current active alerts"""
    if recognizer:
        return recognizer.alert_manager.get_active_alerts()
    return {'alerts': [], 'count': 0}

@app.route('/api/clear_alerts', methods=['POST'])
def clear_alerts():
    """Clear all active alerts"""
    if recognizer:
        recognizer.alert_manager.clear_alerts()
        return {'status': 'success'}
    return {'status': 'error'}

@app.route('/intrusion_image/<filename>')
def serve_intrusion_image(filename):
    """Serve saved intrusion images"""
    from flask import send_from_directory
    return send_from_directory(INTRUSION_CONFIG['log_directory'], filename)

@app.route('/api/stats')
def get_stats():
    """Get intrusion statistics"""
    if recognizer:
        return {
            'active_alert_count': len(recognizer.alert_manager.active_alerts),
            'total_intrusions': len(recognizer.alert_manager.alert_history),
            'detection_stats': recognizer.intrusion_detector.get_detection_stats()
        }
    return {'active_alert_count': 0, 'total_intrusions': 0}

if __name__ == '__main__':
    print("🚀 Starting Hostel Intrusion Detection System...")
    
    # Initialize recognizer
    recognizer = WebVideoFaceRecognizer()
    
    # Start camera (try different sources)
    camera_sources = [0, 1, 2]  # Try different camera indices
    camera_started = False
    
    for source in camera_sources:
        if recognizer.start_camera(source):
            camera_started = True
            break
    
    if not camera_started:
        print("⚠️ No camera found, using video file fallback")
        # You can add video file fallback here
    
    # Start video processing thread
    video_thread = threading.Thread(target=recognizer.process_video_stream, daemon=True)
    video_thread.start()
    
    print("🌐 Starting web server...")
    print("📱 Access the system at: http://localhost:5000")
    print("🔒 Or from remote: http://your-server-ip:5000")
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
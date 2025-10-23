"""
Intrusion Detector for tracking unknown persons and triggering alerts
"""
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime


class IntrusionDetector:
    """Core intrusion detection logic for tracking unknown persons"""
    
    def __init__(self, alert_manager, logger, config):
        self.unknown_detections: Dict[str, Dict[str, Any]] = {}  # bbox_key -> detection_data
        self.alert_manager = alert_manager
        self.logger = logger
        self.config = config
        
        print(f"🔍 Intrusion Detector initialized (threshold: {config['alert_threshold_seconds']}s)")
    
    def get_bbox_key(self, bbox: np.ndarray) -> str:
        """Generate unique key for face location"""
        x1, y1, x2, y2 = map(int, bbox)
        # Round to nearest 10 pixels to group nearby detections
        x1_rounded = (x1 // 10) * 10
        y1_rounded = (y1 // 10) * 10
        return f"{x1_rounded}_{y1_rounded}"
    
    def process_results(self, results: List[Tuple[str, str, np.ndarray, float]], 
                       frame: np.ndarray, current_time: float) -> List[Dict[str, Any]]:
        """Main processing - returns list of new alerts"""
        new_alerts = []
        
        try:
            # Track unknown persons
            for person_id, name, bbox, confidence in results:
                if person_id == "Unknown" and confidence >= self.config['min_confidence_for_alert']:
                    bbox_key = self.get_bbox_key(bbox)
                    self.track_unknown_person(bbox, confidence, current_time, bbox_key)
            
            # Check for alerts
            for bbox_key, detection_data in list(self.unknown_detections.items()):
                alert = self.check_for_alert(bbox_key, detection_data, current_time, frame)
                if alert:
                    new_alerts.append(alert)
            
            # Cleanup old detections
            self.cleanup_old_detections(current_time)
            
        except Exception as e:
            print(f"❌ Error in intrusion detection: {e}")
        
        return new_alerts
    
    def track_unknown_person(self, bbox: np.ndarray, confidence: float, 
                           current_time: float, bbox_key: str):
        """Track unknown person over time"""
        if bbox_key not in self.unknown_detections:
            # New detection
            self.unknown_detections[bbox_key] = {
                'first_seen': current_time,
                'last_seen': current_time,
                'count': 1,
                'bbox': bbox.copy(),
                'max_confidence': confidence,
                'positions': [bbox.copy()]
            }
        else:
            # Update existing detection
            detection = self.unknown_detections[bbox_key]
            detection['last_seen'] = current_time
            detection['count'] += 1
            detection['max_confidence'] = max(detection['max_confidence'], confidence)
            
            # Track position changes (for movement analysis)
            if len(detection['positions']) < 10:  # Keep last 10 positions
                detection['positions'].append(bbox.copy())
            else:
                detection['positions'].pop(0)
                detection['positions'].append(bbox.copy())
    
    def check_for_alert(self, bbox_key: str, detection_data: Dict[str, Any], 
                       current_time: float, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Check if should trigger alert"""
        duration = current_time - detection_data['first_seen']
        
        # Check if duration exceeds threshold
        if duration < self.config['alert_threshold_seconds']:
            return None
        
        # Check cooldown
        if not self.alert_manager.should_alert(bbox_key, current_time):
            return None
        
        try:
            # Create alert
            alert = self.create_alert(detection_data, current_time, frame)
            
            # Add to alert manager
            if self.alert_manager.add_alert(alert):
                # Update cooldown
                self.alert_manager.update_cooldown(bbox_key, current_time)
                return alert
            
        except Exception as e:
            print(f"❌ Error creating alert: {e}")
        
        return None
    
    def create_alert(self, detection_data: Dict[str, Any], current_time: float, 
                    frame: np.ndarray) -> Dict[str, Any]:
        """Create alert with image capture and logging"""
        duration = current_time - detection_data['first_seen']
        bbox = detection_data['bbox']
        confidence = detection_data['max_confidence']
        timestamp = datetime.now()
        
        # Save face image
        image_path = None
        if self.config['save_intrusion_images']:
            image_path = self.logger.save_face_image(frame, bbox, timestamp)
        
        # Log intrusion
        alert_data = self.logger.log_intrusion(
            timestamp=timestamp,
            bbox=bbox,
            confidence=confidence,
            duration=duration,
            image_path=image_path
        )
        
        # Add additional metadata
        alert_data.update({
            'bbox_key': self.get_bbox_key(bbox),
            'detection_count': detection_data['count'],
            'positions': detection_data['positions'][-3:] if detection_data['positions'] else [],  # Last 3 positions
            'alert_type': 'intrusion',
            'severity': self.calculate_severity(confidence, duration, detection_data['count'])
        })
        
        return alert_data
    
    def calculate_severity(self, confidence: float, duration: float, count: int) -> str:
        """Calculate alert severity based on confidence, duration, and detection count"""
        if confidence > 0.7 and duration > 10 and count > 20:
            return 'high'
        elif confidence > 0.5 and duration > 5 and count > 10:
            return 'medium'
        else:
            return 'low'
    
    def cleanup_old_detections(self, current_time: float):
        """Remove stale detections"""
        timeout = self.config['detection_cleanup_timeout']
        to_remove = []
        
        for bbox_key, detection_data in self.unknown_detections.items():
            if current_time - detection_data['last_seen'] > timeout:
                to_remove.append(bbox_key)
        
        for bbox_key in to_remove:
            del self.unknown_detections[bbox_key]
        
        if to_remove:
            print(f"🧹 Cleaned up {len(to_remove)} stale detections")
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get current detection statistics"""
        active_detections = len(self.unknown_detections)
        total_duration = 0
        max_duration = 0
        
        for detection in self.unknown_detections.values():
            duration = time.time() - detection['first_seen']
            total_duration += duration
            max_duration = max(max_duration, duration)
        
        return {
            'active_detections': active_detections,
            'avg_duration': total_duration / active_detections if active_detections > 0 else 0,
            'max_duration': max_duration,
            'threshold_seconds': self.config['alert_threshold_seconds']
        }
    
    def reset_detections(self):
        """Reset all detections (for testing or manual reset)"""
        count = len(self.unknown_detections)
        self.unknown_detections = {}
        print(f"🔄 Reset {count} detections")
    
    def get_active_detections(self) -> List[Dict[str, Any]]:
        """Get current active detections for debugging"""
        current_time = time.time()
        active = []
        
        for bbox_key, detection in self.unknown_detections.items():
            duration = current_time - detection['first_seen']
            active.append({
                'bbox_key': bbox_key,
                'duration': duration,
                'count': detection['count'],
                'confidence': detection['max_confidence'],
                'time_remaining': self.config['alert_threshold_seconds'] - duration
            })
        
        return active
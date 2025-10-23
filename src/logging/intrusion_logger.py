"""
Intrusion Logger for capturing images and maintaining CSV logs
"""
import os
import cv2
import csv
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np


class IntrusionLogger:
    """Handles logging of intrusion events with image capture and CSV logging"""
    
    def __init__(self, log_directory: str = 'logs/intrusions'):
        self.log_directory = log_directory
        self.ensure_directories()
        self.current_log_file = self.get_daily_log_file()
        self._init_csv_file()
    
    def ensure_directories(self):
        """Create log directories if they don't exist"""
        os.makedirs(self.log_directory, exist_ok=True)
        print(f"📁 Log directory: {self.log_directory}")
    
    def get_daily_log_file(self) -> str:
        """Get CSV file path for today's log"""
        today = datetime.now().strftime('%Y%m%d')
        return os.path.join(self.log_directory, f"intrusion_log_{today}.csv")
    
    def _init_csv_file(self):
        """Initialize CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.current_log_file):
            with open(self.current_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'image_path', 'confidence', 'duration_seconds', 
                    'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'
                ])
            print(f"📝 Created new log file: {self.current_log_file}")
    
    def save_face_image(self, frame: np.ndarray, bbox: np.ndarray, timestamp: datetime) -> Optional[str]:
        """Extract face from frame and save as image"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Add padding around face
            padding = 20
            h, w = frame.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # Extract face region
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                print("⚠️ Empty face crop, skipping image save")
                return None
            
            # Generate filename
            timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S_%f')[:-3]  # Include milliseconds
            image_filename = f"intrusion_{timestamp_str}.jpg"
            image_path = os.path.join(self.log_directory, image_filename)
            
            # Save image
            success = cv2.imwrite(image_path, face_crop)
            if success:
                print(f"📸 Saved intrusion image: {image_filename}")
                return image_path
            else:
                print(f"❌ Failed to save image: {image_path}")
                return None
                
        except Exception as e:
            print(f"❌ Error saving face image: {e}")
            return None
    
    def log_intrusion(self, timestamp: datetime, bbox: np.ndarray, confidence: float, 
                     duration: float, image_path: Optional[str] = None) -> Dict[str, Any]:
        """Write intrusion event to CSV log"""
        try:
            # Prepare data for CSV
            x1, y1, x2, y2 = map(int, bbox)
            csv_data = [
                timestamp.isoformat(),
                image_path or '',
                f"{confidence:.3f}",
                f"{duration:.2f}",
                str(x1), str(y1), str(x2), str(y2)
            ]
            
            # Write to CSV
            with open(self.current_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(csv_data)
            
            # Create alert data
            alert_data = {
                'timestamp': timestamp.isoformat(),
                'image_path': image_path,
                'confidence': confidence,
                'duration': duration,
                'bbox': bbox.tolist(),
                'status': 'active'
            }
            
            print(f"📝 Logged intrusion: {timestamp.strftime('%H:%M:%S')} (conf: {confidence:.2f})")
            return alert_data
            
        except Exception as e:
            print(f"❌ Error logging intrusion: {e}")
            return {}
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get statistics about logged intrusions"""
        try:
            if not os.path.exists(self.current_log_file):
                return {'total_entries': 0, 'file_size': 0}
            
            # Count entries
            with open(self.current_log_file, 'r') as f:
                reader = csv.reader(f)
                entries = list(reader)
                total_entries = len(entries) - 1  # Subtract header
            
            # Get file size
            file_size = os.path.getsize(self.current_log_file)
            
            return {
                'total_entries': total_entries,
                'file_size': file_size,
                'log_file': self.current_log_file
            }
            
        except Exception as e:
            print(f"❌ Error getting log stats: {e}")
            return {'total_entries': 0, 'file_size': 0}
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up log files older than specified days"""
        try:
            import glob
            from datetime import timedelta
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            pattern = os.path.join(self.log_directory, "intrusion_log_*.csv")
            log_files = glob.glob(pattern)
            
            deleted_count = 0
            for log_file in log_files:
                file_time = datetime.fromtimestamp(os.path.getmtime(log_file))
                if file_time < cutoff_date:
                    os.remove(log_file)
                    deleted_count += 1
                    print(f"🗑️ Deleted old log: {os.path.basename(log_file)}")
            
            if deleted_count > 0:
                print(f"🧹 Cleaned up {deleted_count} old log files")
                
        except Exception as e:
            print(f"❌ Error cleaning up logs: {e}")
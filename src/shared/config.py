"""
Shared configuration for the face recognition system
"""
import os

# Database configuration
DATABASE_FILE = 'data/face_database.pkl'
ASSETS_DIR = 'assets'
FACES_DIR = os.path.join(ASSETS_DIR, 'faces')

# Camera configuration
CAMERA_URLS = [
    "rtsp://admin:cctv@136@172.168.14.250/",
    "rtsp://admin:cctv@136@172.168.14.250:554/"
]

# RTSP optimization settings
RTSP_SETTINGS = {
    'buffer_size': 1,
    'fps': 15,
    'width': 640,
    'height': 480,
    'fourcc': 'MJPG'
}

# Face recognition settings - IMMUTABLE CONFIGURATION
# These values can ONLY be changed by modifying this file directly
class ImmutableConfig:
    """Immutable configuration that prevents runtime modifications"""
    
    def __init__(self, config_dict):
        self._config = config_dict.copy()
    
    def __getitem__(self, key):
        return self._config[key]
    
    def __setitem__(self, key, value):
        raise RuntimeError(f"Configuration is immutable! Cannot modify '{key}'. "
                          f"Modify config.py directly to change settings.")
    
    def get(self, key, default=None):
        return self._config.get(key, default)
    
    def keys(self):
        return self._config.keys()
    
    def values(self):
        return self._config.values()
    
    def items(self):
        return self._config.items()
    
    def copy(self):
        return self._config.copy()
    
    def update(self, other):
        raise RuntimeError("Configuration is immutable! Modify config.py directly to change settings.")

# Create immutable configuration
FACE_RECOGNITION_CONFIG = ImmutableConfig({
    'similarity_threshold': 0.42,  # Lowered from 0.5 to better recognize distant faces
    'min_face_size': 30,  # Further reduced to catch even smaller faces
    'max_pose_angle': 35,
    'detection_scales': [1.0, 0.75, 0.5, 1.25, 0.25],  # Added 0.25 scale for very small faces
    'enable_multi_scale': True,
    'det_size': (960, 960),  # Increased from 640x640 for better quality on distant faces
    'det_size_fast': (640, 640)  # Increased from 320x320 for better real-time quality
})

# Performance settings
PERFORMANCE_CONFIG = {
    'skip_frames': 3,  # Process every 4th frame
    'fps_monitor_interval': 60,  # Show FPS every 60 frames
    'reconnect_delay': 2,  # Seconds to wait before reconnecting
    'use_fast_mode': False  # Set to True for real-time performance, False for accuracy
}

# Create directories if they don't exist
def ensure_directories():
    """Create necessary directories"""
    for directory in [ASSETS_DIR, FACES_DIR]:
        os.makedirs(directory, exist_ok=True)

# Initialize directories
ensure_directories()

def get_current_detection_config():
    """Get current detection configuration"""
    return {
        'det_size': FACE_RECOGNITION_CONFIG['det_size'],
        'min_face_size': FACE_RECOGNITION_CONFIG['min_face_size'],
        'detection_scales': FACE_RECOGNITION_CONFIG['detection_scales'],
        'enable_multi_scale': FACE_RECOGNITION_CONFIG['enable_multi_scale'],
        'similarity_threshold': FACE_RECOGNITION_CONFIG['similarity_threshold']
    }

# Industry-Standard Intrusion Detection Configuration
INTRUSION_CONFIG = {
    # Alert Management
    'alert_threshold_seconds': 3,
    'cooldown_period_seconds': 60,
    'max_concurrent_alerts': 5,
    'save_intrusion_images': True,
    'log_directory': 'logs/intrusions',
    
    # Temporal Fusion Parameters (Industry Standard)
            'temporal_fusion': {
                'min_frames_for_identity': 3,           # Reduced from 5 - faster unknown detection
                'identity_lock_frames': 15,              # Increased from 10 - maintain known identity longer
                'max_occlusion_frames': 20,              # Increased from 15 - handle longer occlusions
                'confidence_decay_rate': 0.05,           # Reduced from 0.1 - slower confidence decay
                'temporal_smoothing_window': 5,         # Window for moving average
            },
    
    # Confidence Thresholds (Industry Standard)
    'confidence_thresholds': {
        'high_confidence': 0.8,                  # Immediate identity confirmation
        'medium_confidence': 0.5,                # Require temporal consistency
        'low_confidence': 0.25,                   # Lowered from 0.3 - better for small faces
        'occlusion_threshold': 0.4,              # Below this = likely occluded
        'min_confidence_for_alert': 0.25,        # Lowered from 0.3 - better for small/distant faces
        'small_face_threshold': 0.2,             # NEW: Special threshold for small faces
        'min_face_size_pixels': 30,               # NEW: Minimum meaningful face size
    },
    
    # Feature Quality Assessment
    'feature_quality': {
        'min_landmark_visibility': 0.6,         # Minimum visible landmarks ratio
        'quality_decay_rate': 0.05,              # Quality decay during occlusion
        'reconstruction_threshold': 0.7,         # Threshold for face reconstruction
    },
    
    # Motion Prediction (Kalman Integration)
    'motion_prediction': {
        'prediction_horizon': 5,                 # Frames to predict ahead
        'motion_uncertainty_threshold': 0.3,     # Motion prediction uncertainty
        'reappearance_search_radius': 50,        # Pixels to search for reappearance
    },
    
    # System Performance
    'detection_cleanup_timeout': 5.0,
    'track_cleanup_interval': 30,                # Frames between track cleanup
    'max_tracks_per_frame': 10,                 # Maximum active tracks
}

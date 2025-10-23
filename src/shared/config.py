"""
Configuration settings for the intrusion detection system
"""

# Camera Configuration
CAMERA_URLS = {
    'default': 0,  # Default webcam
    'rtsp': 'rtsp://username:password@ip:port/stream',
    'video_file': 'path/to/video.mp4'
}

# RTSP Settings
RTSP_SETTINGS = {
    'timeout': 30,
    'retry_attempts': 3,
    'buffer_size': 1
}

# Face Recognition Configuration
FACE_RECOGNITION_CONFIG = {
    'model_name': 'buffalo_l',
    'similarity_threshold': 0.5,
    'min_face_size': 50,
    'max_faces': 10,
    'detection_confidence': 0.5,
    'recognition_confidence': 0.3
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'max_fps': 30,
    'frame_skip': 1,
    'processing_threads': 2,
    'memory_limit_mb': 1024
}

# Intrusion Detection Configuration
INTRUSION_CONFIG = {
    'alert_threshold_seconds': 3,
    'cooldown_period_seconds': 60,
    'min_confidence_for_alert': 0.3,
    'save_intrusion_images': True,
    'log_directory': 'logs/intrusions',
    'max_concurrent_alerts': 5,
    'detection_cleanup_timeout': 5.0
}
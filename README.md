# AIIMS Attendance System

A Flask-based face recognition system for attendance tracking with advanced features including multi-threading, GPU acceleration, and real-time video processing.

## Features

- **Real-time Face Recognition**: Advanced face detection and recognition using InsightFace
- **Multi-threaded Processing**: Optimized performance with configurable worker threads
- **GPU Acceleration**: CUDA support for faster processing
- **Web Interface**: Modern, responsive web dashboard for monitoring
- **Video Processing**: Support for both live camera feeds and video files
- **Track Management**: Advanced tracking system with temporal smoothing
- **Unknown Face Detection**: Automatic snapshot capture for unknown faces
- **Statistics Dashboard**: Real-time performance metrics and analytics

## System Requirements

- Python 3.8+
- CUDA-compatible GPU (optional but recommended)
- OpenCV
- InsightFace
- Flask
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/prashantsingh91/aiims-attendance.git
cd aiims-attendance
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up the face database:
```bash
# Place face images in assets/faces/ directory
# Run the registration script to add known faces
```

## Usage

1. Start the Flask server:
```bash
python face_recognition_flask.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Use the web interface to:
   - Start/stop video processing
   - Monitor real-time statistics
   - View recognized people
   - Access unknown face snapshots

## Configuration

The system uses immutable configuration in `src/shared/config.py`. Key settings include:

- **Similarity Threshold**: 0.42 (adjustable for recognition sensitivity)
- **Minimum Face Size**: 30 pixels
- **Detection Scales**: Multi-scale detection for various face sizes
- **GPU Memory Fraction**: 0.8 (80% of available GPU memory)

## Project Structure

```
aiims-attendance/
├── face_recognition_flask.py    # Main Flask application
├── src/
│   └── shared/
│       ├── config.py            # Configuration settings
│       └── models.py            # Database models
├── assets/
│   └── faces/                   # Face images for registration
├── data/                        # Face database files
├── snapshots/                   # Unknown face snapshots
└── recorded_videos/             # Video files for processing
```

## API Endpoints

- `GET /` - Main dashboard
- `GET /video_feed` - Live video stream
- `GET /api/stats` - System statistics
- `POST /api/start` - Start video processing
- `POST /api/stop` - Stop video processing
- `POST /api/snapshot` - Take manual snapshot
- `GET /api/snapshots/unknown` - List unknown snapshots

## Performance Features

- **Phase 1 Architecture**: Multi-threaded processing with GPU acceleration
- **Frame Skipping**: Configurable frame processing intervals
- **Temporal Smoothing**: Improved recognition accuracy over time
- **Memory Management**: Efficient GPU memory usage
- **Track Management**: Advanced person tracking with identity persistence

## Troubleshooting

1. **Camera Connection Issues**: Check RTSP URL and network connectivity
2. **GPU Memory Errors**: Reduce `gpu_memory_fraction` in configuration
3. **Low Recognition Accuracy**: Adjust `similarity_threshold` in config
4. **Performance Issues**: Enable `use_fast_mode` for real-time processing

## License

This project is developed for AIIMS attendance tracking system.

## Support

For technical support or questions, please contact the development team.

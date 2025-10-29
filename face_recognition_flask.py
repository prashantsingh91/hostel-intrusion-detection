"""
Advanced Flask-based Face Recognition System - Phase 1
Multi-threaded architecture with GPU acceleration and tracking preparation
"""
import cv2
import numpy as np
import insightface
import time
import warnings
import os
import sys
import threading
import queue
import json
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from collections import deque
from flask import Flask, Response, render_template_string, render_template, request, jsonify
try:
    # Optional DeepSORT dependency
    from deep_sort_realtime.deepsort_tracker import DeepSort  # type: ignore
    HAVE_DEEPSORT = True
except Exception:
    HAVE_DEEPSORT = False
import torch
import gc

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.shared.config import FACE_RECOGNITION_CONFIG
from src.shared.models import FaceDatabase, cosine_similarity

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# Phase 1 Configuration
@dataclass
class SystemConfig:
    """System configuration for Phase 1"""
    # Multi-threading settings
    max_threads: int = 4
    frame_queue_size: int = 10
    processing_timeout: float = 0.1
    
    # GPU settings
    use_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    
    # Frame processing
    frame_skip_interval: int = 3  # Process every 3rd frame for recognition
    temporal_smoothing_frames: int = 10  # Vote over last 10 frames
    embedding_update_interval_frames: int = 5  # Update track appearance features cadence
    
    # Track management
    track_timeout_seconds: float = 28.0
    min_hit_count: int = 10
    alarm_trigger_seconds: float = 8.0
    identity_lock_seconds: float = 10.0
    
    # Performance
    target_fps: float = 24.0
    max_faces_per_frame: int = 10
    
    # Quality Gating Thresholds
    min_detection_confidence: float = 0.65  # Minimum detection score to process
    max_yaw_angle: float = 45.0  # Max head rotation left/right (degrees)
    max_pitch_angle: float = 30.0  # Max head rotation up/down (degrees)
    min_face_sharpness: float = 0.3  # Minimum blur score (0-1, higher = sharper)
    max_occlusion_ratio: float = 0.5  # Max face occlusion to process (50%)

@dataclass
class TrackData:
    """Track data structure for future DeepSORT integration"""
    track_id: int
    bbox: np.ndarray
    embedding: np.ndarray
    confidence: float
    last_seen: float
    hit_count: int = 0
    identity: str = "Unknown"
    identity_confidence: float = 0.0
    temporal_buffer: deque = field(default_factory=lambda: deque(maxlen=10))
    last_unknown_snapshot_time: float = 0.0
    last_known_identity: str = ""
    last_identity_update_time: float = 0.0
    identity_confidence_history: deque = field(default_factory=lambda: deque(maxlen=5))
    last_unknown_embedding: np.ndarray = field(default_factory=lambda: np.zeros(512, dtype=np.float32))
    # Quality metrics
    last_quality_score: float = 0.0
    last_pose_angles: tuple = (0.0, 0.0, 0.0)  # (yaw, pitch, roll)
    quality_good_count: int = 0  # Count of consecutive good quality detections
    
    def update(self, bbox: np.ndarray, embedding: np.ndarray, confidence: float):
        """Update track with new detection"""
        self.bbox = bbox
        self.embedding = embedding
        self.confidence = confidence
        self.last_seen = time.time()
        self.hit_count += 1
        
    def is_expired(self, timeout: float) -> bool:
        """Check if track has expired"""
        return time.time() - self.last_seen > timeout

@dataclass
class AlarmData:
    """Alarm data structure"""
    track_id: int
    start_time: float
    duration: float
    status: str = "active"  # active, resolved, suppressed
    resolution_reason: str = ""

class ThreadSafeStats:
    """Thread-safe statistics container"""
    def __init__(self):
        self._lock = threading.Lock()
        self._stats = {
            'frame_count': 0,
            'total_frames': 0,
            'faces_detected': 0,
            'people_recognized': set(),
            'fps': 0.0,
            'progress': 0.0,
            'active_tracks': 0,
            'active_alarms': 0,
            'gpu_memory_used': 0.0,
            'processing_time': 0.0
        }
    
    def get(self, key: str, default=None):
        with self._lock:
            return self._stats.get(key, default)
    
    def set(self, key: str, value):
        with self._lock:
            self._stats[key] = value
    
    def increment(self, key: str, amount: int = 1):
        with self._lock:
            if key in self._stats:
                if isinstance(self._stats[key], (int, float)):
                    self._stats[key] += amount
                elif isinstance(self._stats[key], set):
                    if isinstance(amount, str):
                        self._stats[key].add(amount)
    
    def get_all(self) -> Dict:
        with self._lock:
            # Convert set to list for JSON serialization
            stats_copy = self._stats.copy()
            if 'people_recognized' in stats_copy:
                stats_copy['people_recognized'] = list(stats_copy['people_recognized'])
            return stats_copy

class AdvancedFlaskFaceRecognizer:
    """Advanced face recognition system with multi-threading and GPU acceleration - Phase 1"""
    
    def __init__(self, database_path: str = '/home/psingh/medgemma/aiims-attendance/data/combined_face_database.pkl'):
        print("🚀 Initializing Advanced Flask Face Recognition System - Phase 1...")
        
        # Initialize configuration first
        self.config = SystemConfig()
        print(f"🔧 System config: GPU={self.config.use_gpu}, Threads={self.config.max_threads}")
        
        # Initialize basic data structures before clearing
        self.tracks: Dict[int, TrackData] = {}
        self.alarms: Dict[int, AlarmData] = {}
        self.next_track_id = 1
        self.unknown_embeddings_history: deque = deque(maxlen=50)
        
        # Clear previous run state
        self._clear_previous_state()
        
        # Initialize GPU settings
        self._setup_gpu()
        
        # Initialize InsightFace with enhanced GPU support
        try:
            print("🔧 Initializing InsightFace with GPU acceleration...")
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.config.use_gpu else ['CPUExecutionProvider']
            self.app_insight = insightface.app.FaceAnalysis(providers=providers)
            
            det_size = FACE_RECOGNITION_CONFIG.get('det_size', (640, 640))
            print(f"🔧 Preparing InsightFace with det_size: {det_size}")
            self.app_insight.prepare(ctx_id=0, det_size=det_size)
            print("✅ InsightFace initialized with GPU acceleration")
        except Exception as e:
            print(f"❌ Error initializing InsightFace: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Initialize database
        print("🔧 Loading face database...")
        self.db = FaceDatabase(database_path)
        print("✅ Face database loaded")
        
        # Recognition settings
        self.similarity_threshold = FACE_RECOGNITION_CONFIG['similarity_threshold']
        self.min_face_size = FACE_RECOGNITION_CONFIG['min_face_size']
        # Lower threshold for small faces
        self.small_face_threshold = 30  # Lower threshold for very small faces
        print(f"🔧 Recognition settings: threshold={self.similarity_threshold}, min_face_size={self.min_face_size}, small_face_threshold={self.small_face_threshold}")
        
        # Pre-load embeddings
        print("🔧 Loading embeddings...")
        self.known_embeddings = []
        self.known_ids = []
        self.known_names = []
        self._load_embeddings()
        print(f"✅ Loaded {len(self.known_embeddings)} embeddings")
        
        # Multi-threading setup
        self._setup_threading()

        # Phase 2: tracker engine (DeepSORT or fallback)
        self._init_tracker()
        
        # Thread-safe statistics (will be reset in _clear_previous_state)
        self.stats = ThreadSafeStats()
        
        # Video processing state
        self.current_frame = None
        self.is_processing = False
        self._current_processing_frame = None  # For quality assessment
        self.video_path = None
        self.frame_lock = threading.Lock()
        
        print("✅ Advanced Flask face recognition system ready!")
        print(f"📊 Loaded {len(self.known_embeddings)} face embeddings")
        print(f"🧵 Multi-threading enabled with {self.config.max_threads} threads")
        print(f"🎮 GPU acceleration: {'Enabled' if self.config.use_gpu else 'Disabled'}")
    
    def _clear_previous_state(self):
        """Clear previous run state including snapshots and statistics"""
        print("🧹 Clearing previous run state...")
        
        # Clear snapshots directory
        snapshots_dir = 'snapshots'
        if os.path.exists(snapshots_dir):
            try:
                import shutil
                shutil.rmtree(snapshots_dir)
                print(f"🗑️  Cleared snapshots directory: {snapshots_dir}")
            except Exception as e:
                print(f"⚠️  Warning: Could not clear snapshots directory: {e}")
        
        # Reset statistics
        self.stats = ThreadSafeStats()
        print("📊 Reset statistics to zero")
        
        # Clear tracks and alarms
        self.tracks.clear()
        self.alarms.clear()
        self.next_track_id = 1
        self.unknown_embeddings_history.clear()
        print("🎯 Cleared all tracks and alarms")
        
        print("✅ Previous run state cleared")
    
    def _setup_gpu(self):
        """Setup GPU configuration and memory management"""
        if self.config.use_gpu and torch.cuda.is_available():
            print(f"🎮 GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"🎮 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(self.config.gpu_memory_fraction)
            print(f"🎮 GPU memory fraction set to: {self.config.gpu_memory_fraction}")
        else:
            print("⚠️  GPU not available, using CPU")
            self.config.use_gpu = False
    
    def _setup_threading(self):
        """Setup multi-threading infrastructure"""
        print("🧵 Setting up multi-threading infrastructure...")
        
        # Frame processing queue
        self.frame_queue = queue.Queue(maxsize=self.config.frame_queue_size)
        
        # Processing threads
        self.threads = []
        self.shutdown_event = threading.Event()
        
        # Start worker threads
        for i in range(self.config.max_threads):
            thread = threading.Thread(
                target=self._worker_thread,
                name=f"Worker-{i}",
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
        
        print(f"✅ Started {len(self.threads)} worker threads")

    def _init_tracker(self):
        """Initialize tracking engine (DeepSORT if available, else simple IoU)."""
        print("🎯 Initializing tracker engine (Phase 2)...")
        self.tracker_type = 'simple_iou'
        self.tracker = None
        if HAVE_DEEPSORT:
            try:
                # Conservative defaults for indoor scenes
                self.tracker = DeepSort(
                    max_age=180,         # keep tracks through longer occlusions
                    n_init=3,            # slightly stricter confirmation
                    max_iou_distance=0.8,
                    max_cosine_distance=0.25,
                    nn_budget=150,
                    embedder='mobilenet',
                )
                self.tracker_type = 'deepsort'
                print("✅ DeepSORT initialized")
            except Exception as e:
                print(f"⚠️  Failed to init DeepSORT, falling back to Simple IoU: {e}")
                self.tracker = None
        if self.tracker is None:
            # Simple IoU tracker state
            self.simple_tracks: Dict[int, Dict] = {}
            self.simple_next_id = 1
            print("✅ Simple IoU tracker initialized")
    
    def _worker_thread(self):
        """Worker thread for processing frames"""
        while not self.shutdown_event.is_set():
            try:
                # Get frame from queue with timeout
                frame_data = self.frame_queue.get(timeout=self.config.processing_timeout)
                if frame_data is None:  # Shutdown signal
                    break
                
                frame, frame_id = frame_data
                self._process_frame_advanced(frame, frame_id)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error in worker thread: {e}")
                try:
                    import traceback
                    traceback.print_exc()
                except Exception:
                    pass
    
    def _process_frame_advanced(self, frame: np.ndarray, frame_id: int):
        """Advanced frame processing with Phase 1 features"""
        start_time = time.time()
        
        try:
            # Store frame for quality assessment
            self._current_processing_frame = frame
            
            # Detect faces with GPU acceleration
            faces = self.detect_faces(frame)
            
            # Debug: Log face detection info
            if len(faces) > 0 and frame_id % 100 == 0:  # Log every 100 frames
                small_faces = sum(1 for f in faces if hasattr(f, 'is_small_face') and f.is_small_face)
                print(f"🔍 Frame {frame_id}: Detected {len(faces)} faces ({small_faces} small)")
            
            # Assess quality for each face and filter
            faces_with_quality = []
            for face in faces:
                quality = self.assess_face_quality(face)
                # Attach quality info to face object for later use
                face.quality_info = quality
                faces_with_quality.append(face)
            
            # Update statistics
            self.stats.increment('faces_detected', len(faces_with_quality))
            
            # Separate faces by quality for different processing
            trackable_faces = [f for f in faces_with_quality if f.quality_info['is_trackable']]
            processable_faces = [f for f in faces_with_quality if f.quality_info['is_good_quality']]
            
            # Phase 2: run tracker association with all trackable faces
            self._update_tracks_with_faces(frame, trackable_faces)
            
            # Update track quality metrics
            for face in faces_with_quality:
                # Find matching track and update quality
                bbox = np.array(face.bbox)
                for tid, td in list(self.tracks.items()):
                    if self._iou(bbox, td.bbox) > 0.5:
                        td.last_quality_score = 1.0 if face.quality_info['is_good_quality'] else 0.0
                        td.last_pose_angles = (
                            face.quality_info['yaw'],
                            face.quality_info['pitch'],
                            face.quality_info['roll']
                        )
                        if face.quality_info['is_good_quality']:
                            td.quality_good_count += 1
                        else:
                            td.quality_good_count = 0
                        break
            
            # Recognition with frame skipping - ONLY process good quality faces
            if frame_id % self.config.frame_skip_interval == 0:
                results = self.recognize_faces(processable_faces)
                self._update_track_identities(results)
                # Trigger unknown snapshotting here when tracks are up-to-date with this frame
                try:
                    self._snapshot_unknowns(frame, results)
                except Exception as e:
                    print(f"⚠️ _snapshot_unknowns error: {e}")
            
            # Update processing time
            processing_time = time.time() - start_time
            self.stats.set('processing_time', processing_time)
            
        except Exception as e:
            print(f"❌ Error processing frame {frame_id}: {e}")
            try:
                import traceback
                traceback.print_exc()
            except Exception:
                pass

    def _update_tracks_with_faces(self, frame: np.ndarray, faces: List):
        """Associate detections to tracks using the active tracker."""
        detections = []
        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox)
            score = float(getattr(f, 'det_score', 1.0))
            # DeepSORT expects [x1,y1,x2,y2], confidence, and optionally feature
            emb = getattr(f, 'embedding', None)
            if emb is not None:
                e = emb.astype(np.float32, copy=True)
                n = float(np.linalg.norm(e)) + 1e-6
                e /= n
            else:
                e = None
            detections.append(((x1, y1, x2, y2), score, e))

        active_ids = set()
        if self.tracker_type == 'deepsort' and self.tracker is not None:
            # Convert to DeepSort input format: list of [xyxy], confidences, features
            boxes = [d[0] for d in detections]
            confs = [d[1] for d in detections]
            feats = [d[2] for d in detections]
            try:
                tracks = self.tracker.update_tracks(boxes, confs, feats)
            except Exception as e:
                print(f"⚠️  DeepSORT update failed, skipping frame: {e}")
                tracks = []
            for t in tracks:
                if not t.is_confirmed():
                    continue
                tid = int(t.track_id)
                ltrb = t.to_ltrb()  # [l,t,r,b]
                bbox = np.array([ltrb[0], ltrb[1], ltrb[2], ltrb[3]])
                active_ids.add(tid)
                self._upsert_track(tid, bbox, confidence=1.0)
                # Update embedding for this track if available
                if t.det_confidence is not None and len(detections) > 0:
                    # best matching detection to this track bbox to grab embedding
                    best_idx = -1; best_i = 0.0
                    for idx, (dbox, _, demb) in enumerate(detections):
                        iou = self._iou(np.array(dbox), bbox)
                        if iou > best_i:
                            best_i = iou; best_idx = idx
                    if best_idx >= 0:
                        _, _, demb = detections[best_idx]
                        if demb is not None:
                            self._update_track_embedding(tid, demb)
        else:
            # Simple IoU matching
            active_ids = self._simple_iou_update(detections)

        # Remove expired tracks
        self._expire_tracks(active_ids)

    def _iou(self, a: np.ndarray, b: np.ndarray) -> float:
        # Convert to numpy arrays and ensure they have at least 4 elements
        a = np.array(a)
        b = np.array(b)
        
        if a.size < 4 or b.size < 4:
            return 0.0
            
        # Extract coordinates as scalars
        xA = max(float(a[0]), float(b[0]))
        yA = max(float(a[1]), float(b[1]))
        xB = min(float(a[2]), float(b[2]))
        yB = min(float(a[3]), float(b[3]))
        
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = max(0, float(a[2]) - float(a[0])) * max(0, float(a[3]) - float(a[1]))
        areaB = max(0, float(b[2]) - float(b[0])) * max(0, float(b[3]) - float(b[1]))
        union = areaA + areaB - inter + 1e-6
        return inter / union

    def _simple_iou_update(self, detections: List[Tuple[Tuple[int,int,int,int], float, Optional[np.ndarray]]]) -> set:
        now = time.time()
        matched = set()
        active_ids = set()
        iou_threshold = 0.2
        # Match existing
        for tid, tr in list(self.simple_tracks.items()):
            best_iou = 0.0; best_idx = -1
            for idx, (bbox, score, emb) in enumerate(detections):
                iou = self._iou(np.array(bbox), tr['bbox'])
                if iou > best_iou:
                    best_iou = iou; best_idx = idx
            if best_idx >= 0 and best_iou >= iou_threshold:
                bbox, score, emb = detections[best_idx]
                tr['bbox'] = np.array(bbox)
                tr['last_seen'] = now
                tr['hit_count'] += 1
                active_ids.add(tid)
                matched.add(best_idx)
                self._upsert_track(tid, tr['bbox'], confidence=score)
        # New tracks
        for idx, (bbox, score, emb) in enumerate(detections):
            if idx in matched:
                continue
            tid = self.simple_next_id; self.simple_next_id += 1
            self.simple_tracks[tid] = {
                'bbox': np.array(bbox),
                'last_seen': now,
                'hit_count': 1,
            }
            active_ids.add(tid)
            self._upsert_track(tid, np.array(bbox), confidence=score)
            if emb is not None:
                self._update_track_embedding(tid, emb)
        return active_ids

    def _upsert_track(self, track_id: int, bbox: np.ndarray, confidence: float):
        now = time.time()
        td = self.tracks.get(track_id)
        if td is None:
            td = TrackData(
                track_id=track_id,
                bbox=bbox.copy(),
                embedding=np.zeros(512, dtype=np.float32),
                confidence=confidence,
                last_seen=now,
                hit_count=1,
            )
            self.tracks[track_id] = td
        else:
            td.update(bbox.copy(), td.embedding, confidence)
        # Update stats
        self.stats.set('active_tracks', len(self.tracks))

    def _expire_tracks(self, active_ids: set):
        timeout = self.config.track_timeout_seconds
        now = time.time()
        to_delete = []
        for tid, td in list(self.tracks.items()):
            if tid not in active_ids and (now - td.last_seen) > timeout:
                to_delete.append(tid)
        for tid in to_delete:
            self.tracks.pop(tid, None)
            if self.tracker_type == 'simple_iou':
                self.simple_tracks.pop(tid, None)
        # Update stats
        self.stats.set('active_tracks', len(self.tracks))

    def _update_track_embedding(self, track_id: int, embedding: np.ndarray):
        """Update per-track embedding on cadence."""
        td = self.tracks.get(track_id)
        if td is None:
            return
        # Update every N frames based on hit_count cadence
        if td.hit_count % max(1, self.config.embedding_update_interval_frames) == 0:
            # L2-normalize embedding for consistent cosine similarity
            e = embedding.astype(np.float32, copy=True)
            norm = float(np.linalg.norm(e)) + 1e-6
            e /= norm
            # Exponential moving average to stabilize appearance
            if np.linalg.norm(td.embedding) < 1e-6:
                td.embedding = e
            else:
                alpha = 0.2  # EMA weight
                td.embedding = (1.0 - alpha) * td.embedding + alpha * e
            # Keep short temporal buffer of normalized embeddings for future re-id
            td.temporal_buffer.append(e)
    
    def _process_face_for_tracking(self, face, frame_id: int):
        """Process face for future tracking integration"""
        # This is Phase 1 preparation - actual tracking will be in Phase 2
        # For now, we'll just log the detection
        bbox = face.bbox
        confidence = face.det_score
        
        # Update statistics
        if confidence > 0.5:  # Only count high-confidence detections
            self.stats.increment('active_tracks')
    
    def _update_track_identities(self, results: List[Tuple[str, str, np.ndarray, float]]):
        """Update track identities based on recognition results with temporal smoothing"""
        now = time.time()
        lock_secs = getattr(self.config, 'identity_lock_seconds', 0.0)
        for person_id, name, bbox, similarity in results:
            # Associate detection to best track via IoU
            x1, y1, x2, y2 = map(int, bbox)
            best_tid = None; best_iou = 0.0
            for tid, td in list(self.tracks.items()):
                iou = self._iou(np.array([x1, y1, x2, y2]), td.bbox)
                if iou > best_iou:
                    best_iou = iou; best_tid = tid
            # If recognized but no reliable track match, still record in stats immediately
            if (best_tid is None or best_iou < 0.3) and person_id != "Unknown":
                self.stats.increment('people_recognized', f"{name} (ID: {person_id})")
                continue
            if best_tid is None:
                continue
            td = self.tracks.get(best_tid)
            if td is None:
                continue

            # Add to confidence history for temporal smoothing
            td.identity_confidence_history.append(float(similarity))
            
            if person_id != "Unknown":
                # Update identity fields
                td.identity = person_id
                td.identity_confidence = float(similarity)
                td.last_known_identity = person_id
                td.last_identity_update_time = now
                self.stats.increment('people_recognized', f"{name} (ID: {person_id})")
            else:
                # Apply identity lock with confidence decay
                if lock_secs > 0 and td.last_known_identity:
                    time_since_known = now - (td.last_identity_update_time or 0.0)
                    if time_since_known <= lock_secs:
                        # Apply confidence decay over time
                        decay_factor = max(0.1, 1.0 - (time_since_known / lock_secs) * 0.8)
                        smoothed_confidence = sum(td.identity_confidence_history) / len(td.identity_confidence_history)
                        
                        if smoothed_confidence > 0.2:  # Still some confidence
                            # Keep the last known identity with decayed confidence
                            td.identity = td.last_known_identity
                            td.identity_confidence = smoothed_confidence * decay_factor
                            continue
                
                # Only set to Unknown if lock period expired or very low confidence
                td.identity = "Unknown"
                td.identity_confidence = float(similarity)
                # Use the track's stabilized embedding (or mean of buffer) for unknown signature
                if np.linalg.norm(td.embedding) > 1e-6:
                    td.last_unknown_embedding = td.embedding.astype(np.float32, copy=True)
                elif len(td.temporal_buffer) > 0:
                    mean_e = np.mean(np.stack(td.temporal_buffer, axis=0), axis=0).astype(np.float32)
                    n = float(np.linalg.norm(mean_e)) + 1e-6
                    td.last_unknown_embedding = (mean_e / n)
                else:
                    td.last_unknown_embedding = np.zeros(512, dtype=np.float32)
    
    def _get_face_embedding_from_bbox(self, bbox: np.ndarray) -> np.ndarray:
        """Extract face embedding from bounding box (placeholder)"""
        # This would extract embedding from the face detection
        # For now, return zero vector as placeholder
        return np.zeros(512, dtype=np.float32)
    
    def cleanup_resources(self):
        """Cleanup GPU and thread resources"""
        print("🧹 Cleaning up resources...")
        
        # Signal shutdown to worker threads
        self.shutdown_event.set()
        
        # Add None to queue to wake up threads
        for _ in range(self.config.max_threads):
            try:
                self.frame_queue.put(None, timeout=1)
            except queue.Full:
                pass
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=2)
        
        # Cleanup GPU memory
        if self.config.use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        print("✅ Resource cleanup completed")
    
    def _load_embeddings(self):
        """Pre-load all embeddings"""
        self.known_embeddings = []
        self.known_ids = []
        self.known_names = []
        
        for person_id, name, embeddings in self.db.get_all_embeddings():
            for embedding in embeddings:
                self.known_embeddings.append(embedding)
                self.known_ids.append(person_id)
                self.known_names.append(name)
    
    def detect_faces(self, image: np.ndarray) -> List:
        """Detect faces in image"""
        faces = self.app_insight.get(image)
        valid_faces = []
        
        for face in faces:
            # Ensure bbox is a proper numpy array
            bbox = np.array(face.bbox)
            if bbox.size < 4:
                continue
                
            face_width = float(bbox[2]) - float(bbox[0])
            face_height = float(bbox[3]) - float(bbox[1])
            face_size = min(face_width, face_height)
            
            # Accept both normal and small faces, but mark them differently
            if face_size >= self.small_face_threshold:
                # Mark small faces for special processing
                if face_size < self.min_face_size:
                    face.is_small_face = True
                else:
                    face.is_small_face = False
                valid_faces.append(face)
        
        return valid_faces
    
    def assess_face_quality(self, face) -> dict:
        """
        Assess face quality based on detection confidence, pose, blur, and occlusion.
        Returns dict with quality metrics and overall pass/fail decision.
        """
        quality_info = {
            'det_score': 0.0,
            'yaw': 0.0,
            'pitch': 0.0,
            'roll': 0.0,
            'sharpness': 1.0,  # Default to good if not available
            'occlusion': 0.0,  # Default to no occlusion
            'is_good_quality': False,
            'is_trackable': True,  # Always trackable unless severely occluded
            'reason': []
        }
        
        reasons = []
        
        # 1. Detection confidence
        det_score = getattr(face, 'det_score', 1.0)
        quality_info['det_score'] = float(det_score)
        if det_score < self.config.min_detection_confidence:
            reasons.append(f"Low detection conf: {det_score:.2f}")
        
        # 2. Pose estimation (yaw, pitch, roll)
        pose = getattr(face, 'pose', None)
        if pose is not None:
            # InsightFace returns pose as [yaw, pitch, roll] in radians or degrees
            # Convert to degrees if needed
            yaw = float(pose[0]) if abs(pose[0]) > 10 else float(np.degrees(pose[0]))
            pitch = float(pose[1]) if abs(pose[1]) > 10 else float(np.degrees(pose[1]))
            roll = float(pose[2]) if abs(pose[2]) > 10 else float(np.degrees(pose[2]))
            
            quality_info['yaw'] = yaw
            quality_info['pitch'] = pitch
            quality_info['roll'] = roll
            
            if abs(yaw) > self.config.max_yaw_angle:
                reasons.append(f"Head turned sideways: yaw={yaw:.1f}°")
            if abs(pitch) > self.config.max_pitch_angle:
                reasons.append(f"Head tilted up/down: pitch={pitch:.1f}°")
        
        # 3. Estimate blur/sharpness from face crop
        try:
            bbox = np.array(face.bbox).astype(int)
            if bbox.size < 4:
                raise ValueError("Invalid bbox")
            x1, y1, x2, y2 = bbox[:4]
            # Get the current frame from processing context
            if hasattr(self, '_current_processing_frame') and self._current_processing_frame is not None:
                h, w = self._current_processing_frame.shape[:2]
                x1 = max(0, min(x1, w-1))
                x2 = max(0, min(x2, w-1))
                y1 = max(0, min(y1, h-1))
                y2 = max(0, min(y2, h-1))
                
                if x2 > x1 and y2 > y1:
                    face_crop = self._current_processing_frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        # Compute Laplacian variance as sharpness metric
                        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
                        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                        # Normalize to 0-1 range (typical values: 0-500, sharp faces > 100)
                        sharpness = min(1.0, laplacian_var / 200.0)
                        quality_info['sharpness'] = sharpness
                        
                        if sharpness < self.config.min_face_sharpness:
                            reasons.append(f"Blurry face: sharpness={sharpness:.2f}")
        except Exception as e:
            pass  # If blur estimation fails, assume it's okay
        
        # 4. Estimate occlusion from landmarks (if available)
        # Avoid boolean evaluation of numpy arrays: do not use `a or b` when `a` may be a numpy array
        lm_106 = getattr(face, 'landmark_2d_106', None)
        if lm_106 is None:
            landmarks = getattr(face, 'kps', None)
        else:
            landmarks = lm_106
        if landmarks is not None:
            # Simple heuristic: check if key landmarks are within face bbox
            # If many landmarks are missing/outside, face is likely occluded
            bbox = np.array(face.bbox)
            if bbox.size >= 4:
                x1, y1, x2, y2 = bbox[:4]
            else:
                x1, y1, x2, y2 = 0, 0, 0, 0
            
            landmarks_array = np.array(landmarks)
            if landmarks_array.ndim == 2 and landmarks_array.size > 0:
                # Count landmarks outside bbox
                outside = 0
                for lm in landmarks_array:
                    # Ensure lm is a valid array with at least 2 elements
                    lm_array = np.array(lm)
                    if lm_array.size >= 2:
                        lx, ly = float(lm_array[0]), float(lm_array[1])
                        if lx < x1 or lx > x2 or ly < y1 or ly > y2:
                            outside += 1
                
                occlusion_ratio = outside / len(landmarks_array) if len(landmarks_array) > 0 else 0.0
                quality_info['occlusion'] = occlusion_ratio
                
                if occlusion_ratio > self.config.max_occlusion_ratio:
                    reasons.append(f"Face occluded: {occlusion_ratio*100:.0f}%")
                    # If > 70% occluded, not even trackable
                    if occlusion_ratio > 0.7:
                        quality_info['is_trackable'] = False
        
        # Overall decision
        quality_info['is_good_quality'] = (
            det_score >= self.config.min_detection_confidence and
            abs(quality_info['yaw']) <= self.config.max_yaw_angle and
            abs(quality_info['pitch']) <= self.config.max_pitch_angle and
            quality_info['sharpness'] >= self.config.min_face_sharpness and
            quality_info['occlusion'] <= self.config.max_occlusion_ratio
        )
        
        quality_info['reason'] = reasons
        return quality_info
    
    def _iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        try:
            # Convert to numpy arrays if not already
            box1 = np.array(box1)
            box2 = np.array(box2)
            
            # Ensure we have valid bounding boxes
            if box1.size < 4 or box2.size < 4:
                return 0.0
            
            # Extract coordinates
            x1_1, y1_1, x2_1, y2_1 = box1[:4]
            x1_2, y1_2, x2_2, y2_2 = box2[:4]
            
            # Calculate intersection area
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)
            
            if x2_i <= x1_i or y2_i <= y1_i:
                return 0.0
            
            intersection = (x2_i - x1_i) * (y2_i - y1_i)
            
            # Calculate union area
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection
            
            if union <= 0:
                return 0.0
            
            return intersection / union
        except Exception:
            return 0.0
    
    def recognize_faces(self, faces: List) -> List[Tuple[str, str, np.ndarray, float]]:
        """Recognize faces using pre-loaded embeddings with temporal smoothing"""
        results = []
        
        for face in faces:
            embedding = face.embedding
            best_similarity = 0
            best_match_id = "Unknown"
            best_match_name = "Unknown"
            
            # Use different thresholds for small vs normal faces
            threshold = self.similarity_threshold
            if hasattr(face, 'is_small_face') and face.is_small_face:
                threshold = max(0.25, self.similarity_threshold - 0.1)  # Lower threshold for small faces
            
            for i, known_embedding in enumerate(self.known_embeddings):
                similarity = cosine_similarity(embedding, known_embedding)
                if similarity > best_similarity and similarity > threshold:
                    best_similarity = similarity
                    best_match_id = self.known_ids[i]
                    best_match_name = self.known_names[i]
            
            # Apply temporal smoothing by finding best matching track
            smoothed_similarity = self._apply_temporal_smoothing(face.bbox, best_similarity)
            
            results.append((best_match_id, best_match_name, face.bbox, smoothed_similarity))
        
        return results
    
    def _apply_temporal_smoothing(self, bbox: np.ndarray, current_similarity: float) -> float:
        """Apply temporal smoothing to recognition confidence"""
        # Find best matching track for temporal smoothing
        x1, y1, x2, y2 = map(int, bbox)
        best_tid = None; best_iou = 0.0
        # Iterate over a snapshot to avoid 'dictionary changed size during iteration'
        for tid, td in list(self.tracks.items()):
            iou = self._iou(np.array([x1, y1, x2, y2]), td.bbox)
            if iou > best_iou:
                best_iou = iou; best_tid = tid
        
        if best_tid is not None and best_iou > 0.3:
            td = self.tracks.get(best_tid)
            if td and len(td.identity_confidence_history) > 0:
                # Use average of last 5 confidence values for smoothing
                smoothed = sum(td.identity_confidence_history) / len(td.identity_confidence_history)
                # Weight current similarity more heavily (70% current, 30% history)
                return 0.7 * current_similarity + 0.3 * smoothed
        
        return current_similarity
    
    def draw_results(self, frame: np.ndarray, results: List[Tuple[str, str, np.ndarray, float]]) -> np.ndarray:
        """Draw bounding boxes and labels"""
        display_frame = frame.copy()
        
        # First, draw track IDs prominently (so they're not hidden behind face recognition boxes)
        # Create a copy to avoid "dictionary changed size during iteration" error
        tracks_copy = dict(self.tracks)
        for tid, td in tracks_copy.items():
            x1, y1, x2, y2 = map(int, td.bbox)
            
            # Draw track bounding box in bright cyan with thick line
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 0), 4)  # Bright cyan, thick line
            
            # Draw track ID label in top-left corner of track box
            label = f"TRACK:{tid}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9
            thickness = 3
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw background rectangle for better visibility
            cv2.rectangle(display_frame, (x1, y1-text_height-15), (x1+text_width+15, y1), (0, 0, 0), -1)
            
            # Draw track ID text in bright yellow
            cv2.putText(display_frame, label, (x1+8, y1-8), font, font_scale, (0, 255, 255), thickness)
        
        # Then draw face recognition results (these will overlay on top)
        for person_id, name, bbox, similarity in results:
            x1, y1, x2, y2 = map(int, bbox)
            
            if person_id != "Unknown":
                color = (0, 255, 0)  # Green
                text1 = f"{name} (ID: {person_id})"
                text2 = f"Confidence: {similarity:.2f}"
            else:
                color = (0, 0, 255)  # Red
                text1 = "Unknown Person"
                text2 = f"Confidence: {similarity:.2f}"
            
            # Draw face recognition box (thinner line so track box shows through)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw text with background in bottom area
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            (text1_width, text1_height), _ = cv2.getTextSize(text1, font, font_scale, thickness)
            (text2_width, text2_height), _ = cv2.getTextSize(text2, font, font_scale, thickness)
            
            # Draw background rectangle at bottom of face box
            cv2.rectangle(display_frame, (x1, y2), (x1 + max(text1_width, text2_width) + 10, y2+50), color, -1)
            cv2.putText(display_frame, text1, (x1+5, y2+20), font, font_scale, (255, 255, 255), thickness)
            cv2.putText(display_frame, text2, (x1+5, y2+40), font, font_scale, (255, 255, 255), thickness)
        
        return display_frame
    
    def add_stats_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add statistics overlay to frame"""
        overlay = frame.copy()
        
        # Status text - handle both video and camera streams
        total_frames = self.stats.get('total_frames', 0) or 0
        frame_count = self.stats.get('frame_count', 0) or 0
        progress = self.stats.get('progress', 0.0) or 0.0
        if total_frames > 0:
            frame_info = f"Frame: {frame_count}/{total_frames} ({progress:.1f}%)"
        else:
            frame_info = f"Frame: {frame_count} (Live Stream)"
        
        fps_value = self.stats.get('fps', 0.0) or 0.0
        faces_detected = self.stats.get('faces_detected', 0) or 0
        people_recognized_set = self.stats.get('people_recognized', set()) or set()
        status_lines = [
            frame_info,
            f"FPS: {fps_value:.1f}",
            f"Faces Detected: {faces_detected}",
            f"People Recognized: {len(people_recognized_set)}"
        ]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        y_offset = 30
        
        for line in status_lines:
            # White text with black outline
            cv2.putText(overlay, line, (10, y_offset), font, font_scale, (255, 255, 255), thickness + 1)
            cv2.putText(overlay, line, (10, y_offset), font, font_scale, (0, 255, 0), thickness - 1)
            y_offset += 30
        
        return overlay
    
    def process_video_stream(self, video_source: str):
        """Process video/camera with multi-threaded architecture"""
        print(f"🎬 Starting advanced video processing for: {video_source}")
        self.video_path = video_source
        self.is_processing = True
        
        # Check if it's a camera stream or video file
        is_rtsp = video_source.startswith('rtsp://') or video_source.startswith('http://')
        is_video_file = not is_rtsp and os.path.exists(video_source)
        
        print(f"📹 Source type - RTSP: {is_rtsp}, Video file: {is_video_file}")
        
        # Configure OpenCV for RTSP with proper settings
        if is_rtsp:
            # Set RTSP transport to TCP for better reliability
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
            cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 15)
            # Set timeouts
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        else:
            cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open video source {video_source}")
            if is_rtsp:
                print("⚠️  Camera might be offline or unreachable")
                print("⚠️  Please check:")
                print("   - Camera is powered on")
                print("   - Network connectivity")
                print("   - RTSP URL is correct")
                print("   - Camera credentials are valid")
            elif is_video_file:
                print("⚠️  Video file exists but cannot be opened")
                print("⚠️  Please check:")
                print("   - File format is supported")
                print("   - File is not corrupted")
                print("   - File permissions are correct")
            self.is_processing = False
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) if not is_rtsp else 15
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_rtsp else 0
        self.stats.set('total_frames', total_frames)
        
        if is_rtsp:
            print(f"📹 Processing camera stream: {video_source}")
            print(f"📹 Target FPS: {fps}")
        else:
            print(f"🎬 Processing video: {video_source}")
            print(f"📹 Total frames: {total_frames}, FPS: {fps}")
        
        start_time = time.time()
        frame_time = 1.0 / fps  # Target time per frame for real-time playback
        
        try:
            frame_count = 0
            while self.is_processing:
                frame_start = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    print("✅ Video processing complete")
                    break
                
                frame_count += 1
                if frame_count % 100 == 0:  # Log every 100 frames
                    print(f"📊 Processed {frame_count} frames, current frame shape: {frame.shape}")
                
                # Update frame count
                self.stats.increment('frame_count')
                
                # Add frame to processing queue for multi-threaded processing
                try:
                    self.frame_queue.put((frame, frame_count), timeout=0.1)
                except queue.Full:
                    # Skip frame if queue is full to maintain real-time performance
                    print("⚠️  Frame queue full, skipping frame")
                    continue
                
                # Process frame for display (immediate processing for streaming)
                faces = self.detect_faces(frame)
                results = self.recognize_faces(faces)
                # Immediately update identities so stats/UI reflect recognition without delay
                try:
                    self._update_track_identities(results)
                except Exception as e:
                    print(f"⚠️ streaming identity update error: {e}")
                
                # Draw results
                display_frame = self.draw_results(frame, results)
                display_frame = self.add_stats_overlay(display_frame)
                
                # Calculate stats
                elapsed = time.time() - start_time
                current_fps = self.stats.get('frame_count') / elapsed if elapsed > 0 else 0
                self.stats.set('fps', current_fps)
                
                if total_frames > 0:
                    progress = (self.stats.get('frame_count') / total_frames) * 100
                    self.stats.set('progress', progress)
                else:
                    self.stats.set('progress', 0)
                
                # Update current frame
                with self.frame_lock:
                    self.current_frame = display_frame.copy()
                
                # Maintain real-time playback speed
                frame_elapsed = time.time() - frame_start
                if frame_elapsed < frame_time:
                    time.sleep(frame_time - frame_elapsed)
                
                # Print progress with enhanced stats
                if self.stats.get('frame_count') % 50 == 0:
                    if total_frames > 0:
                        bar_length = 40
                        filled = int(bar_length * self.stats.get('frame_count') // total_frames)
                        bar = '█' * filled + '░' * (bar_length - filled)
                        print("\r📊 [" + bar + f"] {self.stats.get('progress'):.1f}% | " +
                              f"FPS: {current_fps:.1f} | " +
                              f"Faces: {self.stats.get('faces_detected')} | " +
                              f"Tracks: {self.stats.get('active_tracks')} | " +
                              f"GPU: {self._get_gpu_memory_usage():.1f}%", end='', flush=True)
                    else:
                        print(f"\r📊 Live Stream | Frame: {self.stats.get('frame_count')} | "
                              f"FPS: {current_fps:.1f} | "
                              f"Faces: {self.stats.get('faces_detected')} | "
                              f"Tracks: {self.stats.get('active_tracks')} | "
                              f"GPU: {self._get_gpu_memory_usage():.1f}%", end='', flush=True)
        
        except Exception as e:
            print(f"❌ Error during processing: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cap.release()
            self.is_processing = False
            print("\n🏁 Video processing finished!")
    
    def _get_gpu_memory_usage(self) -> float:
        """Get GPU memory usage percentage"""
        if self.config.use_gpu and torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated(0)
                total = torch.cuda.get_device_properties(0).total_memory
                return (allocated / total) * 100
            except:
                return 0.0
        return 0.0

    def _snapshot_unknowns(self, frame: np.ndarray, results: List[Tuple[str, str, np.ndarray, float]]):
        """Take one-off snapshots for unknown faces using per-track cooldown.
        Requirements:
        - Match detection bbox to an existing track by IoU
        - Track must be warmed up (hit_count >= min_hit_count)
        - Cooldown of 120s per track between unknown snapshots
        - Face must be reasonably large (>= min_face_size)
        """
        now = time.time()
        min_hits = getattr(self.config, 'min_hit_count', 5)  # Reduced from 10 to 5
        min_size = getattr(self, 'small_face_threshold', 30)  # Use small face threshold
        cooldown = 120.0  # Increased to 2 minutes (120 seconds)
        lock_secs = getattr(self.config, 'identity_lock_seconds', 0.0)
 
        for person_id, _name, bbox, _sim in results:
            if person_id != "Unknown":
                # Debug: known, skip
                # print("DBG: known person, skip snapshot")
                continue
            x1, y1, x2, y2 = map(int, bbox)
            if min(x2 - x1, y2 - y1) < min_size:
                print(f"DBG: face too small for snapshot: {(x2-x1)}x{(y2-y1)} < {min_size}")
                continue
            # Find best-matching track by IoU
            best_tid = None; best_iou = 0.0
            for tid, td in list(self.tracks.items()):
                iou = self._iou(np.array([x1, y1, x2, y2]), td.bbox)
                if iou > best_iou:
                    best_iou = iou; best_tid = tid
            if best_tid is None or best_iou < 0.5:  # Reduced from 0.75 to 0.5
                print(f"DBG: no matching track (best_iou={best_iou:.2f}) for bbox {x1,y1,x2,y2}")
                continue
            td = self.tracks.get(best_tid)
            if td is None:
                continue
            # Suppress snapshot during identity lock window if track was recently known
            if lock_secs > 0 and td.last_known_identity and (now - (td.last_identity_update_time or 0.0)) <= lock_secs:
                continue
            if td.hit_count < min_hits:
                print(f"DBG: track {best_tid} not warmed up (hits={td.hit_count} < {min_hits})")
                continue
            
            # QUALITY GATE: More lenient for small faces
            quality_threshold = 2  # Reduced from 3
            if hasattr(td, 'quality_good_count') and td.quality_good_count < quality_threshold:
                print(f"DBG: track {best_tid} insufficient good quality frames (count={td.quality_good_count} < {quality_threshold})")
                continue
            
            # Lower quality threshold for small faces
            min_quality = 0.3  # Reduced from 0.5
            if hasattr(td, 'last_quality_score') and td.last_quality_score < min_quality:
                print(f"DBG: track {best_tid} poor current quality (score={td.last_quality_score:.2f})")
                continue
            
            # Check for duplicate unknown person using embedding comparison
            if self._is_duplicate_unknown(td.last_unknown_embedding):
                print(f"DBG: track {best_tid} is duplicate unknown person, skipping snapshot")
                continue
            # Backfill missing field on legacy TrackData instances
            if not hasattr(td, 'last_unknown_snapshot_time'):
                setattr(td, 'last_unknown_snapshot_time', 0.0)
            if (now - td.last_unknown_snapshot_time) < cooldown:
                print(f"DBG: track {best_tid} cooldown active {(now-td.last_unknown_snapshot_time):.1f}s < {cooldown}s")
                continue
            # Save snapshot crop and add to history
            self._save_unknown_snapshot(best_tid, frame, td.bbox)
            td.last_unknown_snapshot_time = now
            # Add embedding to history for duplicate detection
            self.unknown_embeddings_history.append(td.last_unknown_embedding.copy())

    def _save_unknown_snapshot(self, track_id: int, frame: np.ndarray, bbox: np.ndarray):
        """Save cropped unknown face snapshot to snapshots/unknown_*.jpg"""
        from datetime import datetime
        try:
            # Use absolute path to ensure snapshots are saved in the correct directory
            snapshots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'snapshots')
            os.makedirs(snapshots_dir, exist_ok=True)
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                print(f"DBG: invalid bbox for snapshot: {x1,y1,x2,y2}")
                return
            crop = frame[y1:y2, x1:x2].copy()
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"unknown_{track_id}_{ts}.jpg"
            path = os.path.join(snapshots_dir, filename)
            ok = cv2.imwrite(path, crop)
            if ok:
                print(f"📸 Snapshot saved: {path}")
            else:
                print(f"❌ Failed to save snapshot: {path}")
        except Exception as e:
            print(f"⚠️ _save_unknown_snapshot error: {e}")

    def _is_duplicate_unknown(self, embedding: np.ndarray) -> bool:
        """Check if unknown person is duplicate using embedding comparison"""
        if len(self.unknown_embeddings_history) == 0:
            return False
        
        # Compare with recent unknown embeddings
        for hist_embedding in self.unknown_embeddings_history:
            if np.linalg.norm(embedding) > 0 and np.linalg.norm(hist_embedding) > 0:
                similarity = cosine_similarity(embedding, hist_embedding)
                if similarity > 0.7:  # High similarity threshold for duplicates
                    return True
        return False

# Global recognizer instance
recognizer = None

def generate_frames():
    """Generator function for video streaming"""
    frame_count = 0
    while True:
        if recognizer and recognizer.current_frame is not None:
            with recognizer.frame_lock:
                frame = recognizer.current_frame.copy()
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                frame_count += 1
                if frame_count % 100 == 0:  # Log every 100 frames
                    print(f"📺 Streaming frame {frame_count}, size: {len(frame_bytes)} bytes")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                print("❌ Failed to encode frame to JPEG")
        else:
            # Create a placeholder frame when no video is available
            if recognizer and recognizer.is_processing:
                # Show "Processing..." message
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Processing Video...", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                cv2.putText(placeholder, "Please wait", (150, 300), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                ret, buffer = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                # Show "Ready" message
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Ready to Start", (100, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 165, 0), 3)
                cv2.putText(placeholder, "Click Start button", (120, 300), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                ret, buffer = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS

@app.route('/')
def index():
    """Face Recognition Dashboard"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    """Get current statistics with Phase 1 enhancements"""
    try:
        if recognizer:
            stats = recognizer.stats.get_all()
            return {
                'frame_count': stats.get('frame_count', 0),
                'fps': stats.get('fps', 0.0),
                'faces_detected': stats.get('faces_detected', 0),
                'people_recognized': len(stats.get('people_recognized', [])),
                'people_list': stats.get('people_recognized', []),
                'is_processing': recognizer.is_processing,
                'active_tracks': stats.get('active_tracks', 0),
                'active_alarms': stats.get('active_alarms', 0),
                'gpu_memory_used': stats.get('gpu_memory_used', 0.0),
                'processing_time': stats.get('processing_time', 0.0),
                'progress': stats.get('progress', 0.0),
                'total_frames': stats.get('total_frames', 0),
                'tracker': {
                    'type': getattr(recognizer, 'tracker_type', 'unknown'),
                    'deepsort_available': HAVE_DEEPSORT
                }
            }
        return {
            'frame_count': 0,
            'fps': 0,
            'faces_detected': 0,
            'people_recognized': 0,
            'people_list': [],
            'is_processing': False,
            'active_tracks': 0,
            'active_alarms': 0,
            'gpu_memory_used': 0.0,
            'processing_time': 0.0,
            'progress': 0.0,
            'total_frames': 0,
            'tracker': {
                'type': 'unknown',
                'deepsort_available': HAVE_DEEPSORT
            }
        }
    except Exception as e:
        print(f"❌ Error in get_stats: {e}")
        return {
            'frame_count': 0,
            'fps': 0,
            'faces_detected': 0,
            'people_recognized': 0,
            'people_list': [],
            'is_processing': False,
            'active_tracks': 0,
            'active_alarms': 0,
            'gpu_memory_used': 0.0,
            'processing_time': 0.0,
            'progress': 0.0,
            'total_frames': 0,
            'tracker': {
                'type': 'unknown',
                'deepsort_available': HAVE_DEEPSORT
            },
            'error': str(e)
        }

@app.route('/api/start', methods=['POST'])
def start_processing():
    """Start video processing"""
    global recognizer
    
    try:
        if not recognizer:
            return jsonify({'status': 'error', 'message': 'Recognizer not initialized'})
        
        if recognizer.is_processing:
            return jsonify({'status': 'error', 'message': 'Processing already running'})
        
        # Use default video path (requested by user)
        video_path = '/home/psingh/medgemma/aiims-attendance/recorded_videos/recorded_footage_20251014_164949.mp4'
        
        if not os.path.exists(video_path):
            return jsonify({'status': 'error', 'message': f'Video file not found: {video_path}'})
        
        # Start processing in background thread
        processing_thread = threading.Thread(
            target=recognizer.process_video_stream,
            args=(video_path,),
            daemon=True
        )
        processing_thread.start()
        
        return jsonify({'status': 'success', 'message': 'Processing started'})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/stop', methods=['POST'])
def stop_processing():
    """Stop video processing"""
    try:
        if recognizer:
            recognizer.is_processing = False
            return jsonify({'status': 'success', 'message': 'Processing stopped'})
        return jsonify({'status': 'error', 'message': 'Recognizer not available'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/snapshot', methods=['POST'])
def take_snapshot():
    """Take a snapshot of the current video frame"""
    import os
    import cv2
    from datetime import datetime
    
    if not recognizer or recognizer.current_frame is None:
        return jsonify({'status': 'error', 'message': 'No video frame available'})
    
    try:
        # Create snapshots directory if it doesn't exist
        snapshots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'snapshots')
        os.makedirs(snapshots_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'snapshot_{timestamp}.jpg'
        filepath = os.path.join(snapshots_dir, filename)
        
        # Save current frame
        with recognizer.frame_lock:
            frame = recognizer.current_frame.copy()
        
        success = cv2.imwrite(filepath, frame)
        
        if success:
            return jsonify({
                'status': 'success',
                'filename': filename,
                'image_url': f'/snapshot/{filename}',
                'message': 'Snapshot saved successfully'
            })
        else:
            return jsonify({'status': 'error', 'message': 'Failed to save image'})
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error taking snapshot: {str(e)}'})

@app.route('/snapshot/<filename>')
def serve_snapshot(filename):
    """Serve saved snapshots"""
    from flask import send_from_directory
    import os
    # Use absolute path to snapshots directory
    snapshots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'snapshots')
    return send_from_directory(snapshots_dir, filename)

@app.route('/api/snapshots/unknown')
def list_unknown_snapshots():
    """List all saved unknown snapshots with timestamps"""
    try:
        # Use absolute path to snapshots directory
        snapshots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'snapshots')
        if not os.path.exists(snapshots_dir):
            return {'snapshots': []}
        
        snapshot_files = []
        for f in os.listdir(snapshots_dir):
            if (f.startswith('unknown_') or f.startswith('snapshot_')) and f.endswith('.jpg'):
                filepath = os.path.join(snapshots_dir, f)
                # Try to extract timestamp from filename patterns
                human_readable_timestamp = "Unknown Time"
                try:
                    from datetime import datetime
                    if f.startswith('unknown_'):
                        # unknown_<track>_<YYYYMMDD>_<HHMMSS>.jpg
                        parts = f.replace('.jpg','').split('_')
                        if len(parts) >= 4:
                            dt_object = datetime.strptime(parts[-2] + '_' + parts[-1], '%Y%m%d_%H%M%S')
                            human_readable_timestamp = dt_object.strftime('%Y-%m-%d %H:%M:%S')
                    elif f.startswith('snapshot_'):
                        # snapshot_<YYYYMMDD>_<HHMMSS>.jpg
                        parts = f.replace('snapshot_','').replace('.jpg','').split('_')
                        if len(parts) >= 2:
                            dt_object = datetime.strptime(parts[-2] + '_' + parts[-1], '%Y%m%d_%H%M%S')
                            human_readable_timestamp = dt_object.strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    pass
                
                snapshot_files.append({
                    'filename': f,
                    'url': f'/snapshot/{f}',
                    'timestamp': human_readable_timestamp
                })
        
        # Sort by timestamp, newest first
        snapshot_files.sort(key=lambda x: x['filename'], reverse=True)
        return {'snapshots': snapshot_files}
        
    except Exception as e:
        print(f"❌ Error listing snapshots: {e}")
        return {'snapshots': []}

@app.route('/api/videos')
def list_videos():
    """List all available videos in the recorded_videos folder"""
    try:
        videos_dir = "recorded_videos"
        if not os.path.exists(videos_dir):
            return {'videos': []}
        
        video_files = []
        for f in os.listdir(videos_dir):
            if f.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                filepath = os.path.join(videos_dir, f)
                file_size = os.path.getsize(filepath)
                file_size_mb = file_size / (1024 * 1024)
                
                # Create display name
                display_name = f.replace('.mp4', '').replace('_', ' ').title()
                
                video_files.append({
                    'filename': f,
                    'path': filepath,
                    'display_name': display_name,
                    'size_mb': round(file_size_mb, 1)
                })
        
        # Sort by filename
        video_files.sort(key=lambda x: x['filename'])
        return {'videos': video_files}
        
    except Exception as e:
        print(f"❌ Error listing videos: {e}")
        return {'videos': []}

def test_camera_connection(camera_url: str, timeout: int = 3) -> bool:
    """Test if camera is accessible"""
    print(f"🔍 Testing connection to: {camera_url}")
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
    cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout * 1000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout * 1000)
    
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            print(f"✅ Camera connection successful!")
            return True
        else:
            print(f"❌ Camera opened but cannot read frames")
            return False
    else:
        print(f"❌ Cannot connect to camera")
        return False

def main():
    """Main function with Phase 1 enhancements"""
    global recognizer
    
    print("🌐 ADVANCED FLASK-BASED FACE RECOGNITION SYSTEM - PHASE 1")
    print("=" * 70)
    
    try:
        # Initialize advanced recognizer
        print("🔧 Initializing AdvancedFlaskFaceRecognizer...")
        recognizer = AdvancedFlaskFaceRecognizer()
        print("✅ AdvancedFlaskFaceRecognizer initialized successfully")
        
        print("\n" + "=" * 70)
        print("🌐 WEB SERVER STARTING - PHASE 1 FEATURES")
        print("=" * 70)
        print(f"📺 Access the video stream at:")
        print(f"   http://98.70.41.227:5000")
        print(f"   http://localhost:5000 (if SSH tunneling)")
        print("=" * 70)
        print("🚀 PHASE 1 FEATURES ENABLED:")
        print("   ✅ Multi-threaded processing")
        print("   ✅ GPU acceleration")
        print("   ✅ Enhanced statistics")
        print("   ✅ Track preparation")
        print("   ✅ Memory management")
        print("=" * 70)
        print("⚠️  Make sure port 5000 is open in Azure NSG!")
        print("=" * 70)
        
        # Start Flask server
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup resources on shutdown
        if recognizer:
            recognizer.cleanup_resources()

if __name__ == "__main__":
    main()

"""
Shared database models and utilities
"""
import pickle
import os
from typing import Dict, List, Tuple, Optional
import numpy as np

class FaceDatabase:
    """Manages the face database with optimized operations"""
    
    def __init__(self, database_file: str = 'face_database.pkl'):
        self.database_file = database_file
        self.known_faces = {}  # id -> {'name': str, 'embeddings': [np.ndarray], 'poses': [dict]}
        self.load_database()
    
    def add_person(self, person_id: str, name: str, embeddings: List[np.ndarray], poses: List[dict] = None):
        """Add a new person to the database"""
        if poses is None:
            poses = [{'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}] * len(embeddings)
        
        self.known_faces[person_id] = {
            'name': name,
            'embeddings': embeddings,
            'poses': poses
        }
        self.save_database()
    
    def get_person(self, person_id: str) -> Optional[dict]:
        """Get person data by ID"""
        return self.known_faces.get(person_id)
    
    def delete_person(self, person_id: str) -> bool:
        """Delete a person from the database"""
        if person_id in self.known_faces:
            del self.known_faces[person_id]
            self.save_database()
            return True
        return False
    
    def list_people(self) -> List[Tuple[str, str]]:
        """List all registered people as (id, name) tuples"""
        return [(person_id, data['name']) for person_id, data in self.known_faces.items()]
    
    def get_all_embeddings(self) -> List[Tuple[str, str, List[np.ndarray]]]:
        """Get all embeddings for recognition"""
        return [(person_id, data['name'], data['embeddings']) 
                for person_id, data in self.known_faces.items()]
    
    def save_database(self):
        """Save database to file"""
        try:
            with open(self.database_file, 'wb') as f:
                pickle.dump(self.known_faces, f)
            print(f"💾 Database saved with {len(self.known_faces)} faces")
        except Exception as e:
            print(f"❌ Error saving database: {e}")
    
    def load_database(self):
        """Load database from file"""
        try:
            if os.path.exists(self.database_file):
                with open(self.database_file, 'rb') as f:
                    self.known_faces = pickle.load(f)
                print(f"📂 Loaded {len(self.known_faces)} registered faces")
            else:
                print("📂 No existing database found. Starting fresh.")
        except Exception as e:
            print(f"❌ Error loading database: {e}")
            self.known_faces = {}
    
    def get_stats(self) -> dict:
        """Get database statistics"""
        total_embeddings = sum(len(data['embeddings']) for data in self.known_faces.values())
        return {
            'total_people': len(self.known_faces),
            'total_embeddings': total_embeddings,
            'avg_embeddings_per_person': total_embeddings / len(self.known_faces) if self.known_faces else 0
        }

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_pose_info(face) -> dict:
    """Extract pose information from face"""
    pose_info = {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
    if hasattr(face, 'pose') and face.pose is not None:
        try:
            yaw, pitch, roll = face.pose
            pose_info = {'yaw': float(yaw), 'pitch': float(pitch), 'roll': float(roll)}
        except:
            pass
    return pose_info

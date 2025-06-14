import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import time
from firebase_admin import credentials, db
import firebase_admin
from collections import defaultdict, deque
from sklearn.preprocessing import normalize
import faiss
import uuid
import atexit
import signal
import sys
import warnings
from datetime import datetime
import logging
from config import CONFIG
from data_handler import DataHandler
from training import Trainer
from camera import CameraManager
from postgress_logger import PostgresLogger

cred = credentials.Certificate('faculty-tracker-30135-firebase-adminsdk-j1nux-d744591934.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://faculty-tracker-30135-default-rtdb.firebaseio.com/'
})
faculty_ref = db.reference('faculty_log')
warnings.filterwarnings("ignore", category=FutureWarning)

class AdvancedFaceRecognition:
    def __init__(self):
        print("Initializing face analysis with buffalo_s...")
        self.app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=CONFIG["face_size"], det_thresh=0.4)
        
        self.data_handler = DataHandler(CONFIG["data_file"])
        self.data_handler.load_data()

        

        
        self.trainer = Trainer(self.app, CONFIG)
        
        self.face_tracks = defaultdict(lambda: defaultdict(dict))
        self.empty_frame = np.zeros((CONFIG["display_resolution"][1], CONFIG["display_resolution"][0], 3), dtype=np.uint8)
        self.detected_persons = defaultdict(dict)
        
        self.camera_manager = CameraManager(CONFIG, self.process_frame)

        self.pg_logger = PostgresLogger()

        
        logging.basicConfig(filename=CONFIG['logfile'], level=logging.DEBUG, format='%(message)s')
        atexit.register(self.on_exit)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, sig, frame):
        print(f"Signal {sig} received. Shutting down...")
        self.camera_manager.stop()
        self.on_exit()
        sys.exit(0)

    def on_exit(self):
        print("System shutting down...")
        self.camera_manager.stop()
        self.data_handler.save_data()

    def train_model(self):
        known_embeddings, known_labels = self.trainer.train_model()
        self.data_handler.known_embeddings = known_embeddings
        self.data_handler.known_labels = known_labels
        self.data_handler.save_data()
        print(f"Training complete. Learned {len(self.data_handler.known_labels)} embeddings for {len(set(self.data_handler.known_labels))} identities")

    def should_process_person(self, person_name, camera_idx):
        current_time = time.time()
        from datetime import date
        today = date.today().strftime('%Y%m%d')
        
        if person_name == "Unknown":
            return True
            
        if person_name in self.detected_persons[camera_idx]:
            last_detection = self.detected_persons[camera_idx][person_name]
            if current_time - last_detection < CONFIG["person_detection_timeout"]:
                return False
                
        self.detected_persons[camera_idx][person_name] = current_time
        
        safe_name = person_name.replace('.', '_')
        safe_ip = CONFIG['cameras'][camera_idx]['classroom'].replace('.', '_')
        faculty_ref.child(safe_name).child(safe_ip).child(today).push({
            'time': datetime.fromtimestamp(current_time).strftime('%H:%M:%S')
        })

        ##postgress 
        self.pg_logger.log_detection(
            person_name,
            CONFIG['cameras'][camera_idx]['hikvision_ip'],
            CONFIG['cameras'][camera_idx]['classroom']
        )

        logging.info("Saved logs on database")
        logging.info(f"  DETECTION: COMPUTER DEPARTMENT Person '{person_name}' detected on Camera #{CONFIG['cameras'][camera_idx]['hikvision_ip']}, Time {datetime.fromtimestamp(current_time).strftime('%H:%M:%S')}")
        print(f"DETECTION: Person '{person_name}' detected on Camera #{camera_idx+1}, Time {datetime.fromtimestamp(current_time).strftime('%H:%M:%S')}")
        return True

    def process_frame(self, frame, camera_idx): 
        if frame is None:
            return None
            
        frame_copy = frame.copy()
        faces = []
        
        for scale in CONFIG["detection_scales"]:
            if scale != 1.0:
                scaled_frame = cv2.resize(frame, None, fx=scale, fy=scale)
            else:
                scaled_frame = frame
            scale_faces = self.app.get(scaled_frame)
            for face in scale_faces:
                if face.det_score < CONFIG["recognition_confidence"]:
                    continue
                face.bbox = face.bbox / scale
                faces.append(face)
        
        for face in faces:
            embedding = normalize(face.embedding.reshape(1, -1)).flatten()
            
            # if len(self.data_handler.known_embeddings) > 0:
            #     similarities = np.dot(self.data_handler.known_embeddings, embedding)
            #     best_match_idx = np.argmax(similarities)
            #     max_similarity = similarities[best_match_idx]
            #     dynamic_threshold = min(CONFIG["similarity_threshold"] + 0.1 * (max_similarity - 0.5), 0.55)
            if hasattr(self, 'faiss_index') and self.faiss_index.ntotal > 0:
                embedding_f32 = embedding.astype('float32').reshape(1, -1)
                scores, indices = self.faiss_index.search(embedding_f32, 1)  # top-1
                max_similarity = scores[0][0]
                best_match_idx = indices[0][0]
                dynamic_threshold = min(CONFIG["similarity_threshold"] + 0.1 * (max_similarity - 0.5), 0.55)
            else:
                max_similarity = 0
                best_match_idx = -1
                dynamic_threshold = CONFIG["similarity_threshold"]
            
            face_id = self._get_face_id(face.bbox, embedding, camera_idx)
            bbox = face.bbox.astype(int)
            
            track = self.face_tracks[camera_idx][face_id]
            track['embeddings'].append(embedding)
            # if len(self.data_handler.known_embeddings) > 0:
            track['scores'].append(max_similarity)
            track['labels'].append(best_match_idx)
            
            if len(track['scores']) >= CONFIG["min_frames_for_recognition"] :
                weights = np.linspace(0.5, 1.0, len(track['scores']))
                weights /= weights.sum()
                avg_sim = np.average(track['scores'], weights=weights)
                label_counts = defaultdict(int)
                for idx in track['labels']:
                    label_counts[idx] += 1
                label_idx = max(label_counts, key=label_counts.get)
                label = self.data_handler.known_labels[label_idx] if avg_sim > dynamic_threshold else "Unknown"
                
                process_person = self.should_process_person(label, camera_idx)
                
                if process_person:
                    color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 1)
                    cv2.putText(frame_copy, f"{label} ({avg_sim:.2f})", (bbox[0], bbox[1]-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                else:
                    cv2.rectangle(frame_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (150, 150, 150), 1)
                    cv2.putText(frame_copy, f"Skipped: {label}", (bbox[0], bbox[1]-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        self._clean_tracks(camera_idx)
        return frame_copy

    def _get_face_id(self, bbox, embedding, camera_idx):
        current_time = time.time()
        for face_id, track in self.face_tracks[camera_idx].items():
            if current_time - track.get('last_seen', 0) > 1.0:
                continue
            last_bbox = track['last_bbox']
            if len(track['embeddings']) == 0:
                continue
            last_embed = track['embeddings'][-1]
            spatial_dist = np.linalg.norm(np.array(bbox[:2]) - np.array(last_bbox[:2]))
            feature_sim = np.dot(last_embed, embedding)
            if spatial_dist < 30 and feature_sim > 0.85:
                track['last_bbox'] = bbox
                track['last_seen'] = current_time
                return face_id
        new_id = f"face_{str(uuid.uuid4())[:8]}"
        self.face_tracks[camera_idx][new_id] = {
            'embeddings': deque(maxlen=CONFIG["min_frames_for_recognition"]),
            'scores': deque(maxlen=CONFIG["min_frames_for_recognition"]),
            'labels': deque(maxlen=CONFIG["min_frames_for_recognition"]),
            'last_bbox': bbox,
            'last_seen': current_time
        }
        return new_id

    def _clean_tracks(self, camera_idx):
        current_time = time.time()
        to_remove = [fid for fid, track in self.face_tracks[camera_idx].items() if current_time - track.get('last_seen', 0) > 3]
        for fid in to_remove:
            del self.face_tracks[camera_idx][fid]

    def _clean_detected_persons(self):
        current_time = time.time()
        for camera_idx in self.detected_persons:
            to_remove = []
            for person, last_detected in self.detected_persons[camera_idx].items():
                if current_time - last_detected > CONFIG["person_detection_timeout"]:
                    to_remove.append(person)
            for person in to_remove:
                del self.detected_persons[camera_idx][person]

    def get_total_tracked_persons(self):
        all_persons = set()
        for camera_idx in self.detected_persons:
            for person in self.detected_persons[camera_idx]:
                all_persons.add(person)
        return len(all_persons)

    def run(self):
        print("Starting multi-camera face recognition system...")
        print(f"Person detection timeout set to {CONFIG['person_detection_timeout']} seconds per camera")
        
        
        if not self.data_handler.known_labels:
            print("No known faces found - running training")
            self.train_model()
            
        if os.path.exists('faiss_index.idx'):
            self.faiss_index = faiss.read_index('faiss_index.idx')
        else:
            self.train_model()

        
        self.camera_manager.start_cameras()
        
        num_cameras = len(CONFIG["cameras"])
        grid_cols = 2
        grid_rows = (num_cameras + 1) // 2
        frame_width = CONFIG["display_resolution"][0] // grid_cols
        frame_height = CONFIG["display_resolution"][1] // grid_rows
        
        cv2.namedWindow('Multi-Camera Face Recognition', cv2.WINDOW_NORMAL)
        
        fps_start = time.time()
        fps_count = 0
        fps = 0
        last_cleanup_time = time.time()
        
        camera_frames = [None] * num_cameras
        last_frame_time = [0] * num_cameras
        
        while True:
            combined_frame = self.empty_frame.copy()
            frames_updated = False
            
            current_time = time.time()
            if current_time - last_cleanup_time > 60:
                self._clean_detected_persons()
                last_cleanup_time = current_time
            
            for idx in range(num_cameras):
                try:
                    while not self.camera_manager.frame_queues[idx].empty():
                        frame, cam_idx = self.camera_manager.frame_queues[idx].get_nowait()
                        camera_frames[cam_idx] = frame
                        last_frame_time[cam_idx] = time.time()
                        frames_updated = True
                except Exception as e:
                    print(f"Error getting frame from camera {idx}: {e}")
            
            display_count = 0
            for idx in range(num_cameras):
                if camera_frames[idx] is not None:
                    if time.time() - last_frame_time[idx] > 5.0:
                        row = idx // grid_cols
                        col = idx % grid_cols
                        y_start = row * frame_height
                        x_start = col * frame_width
                        cv2.rectangle(combined_frame, (x_start, y_start), 
                                    (x_start + frame_width, y_start + frame_height), 
                                    (0, 0, 0), -1)
                        cv2.putText(combined_frame, f"Camera #{idx+1}", 
                                (x_start + 10, y_start + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(combined_frame, "Connection Lost", 
                                (x_start + 10, y_start + frame_height//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        continue
                    
                    frame = cv2.resize(camera_frames[idx], (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
                    row = idx // grid_cols
                    col = idx % grid_cols
                    y_start = row * frame_height
                    x_start = col * frame_width
                    combined_frame[y_start:y_start+frame_height, x_start:x_start+frame_width] = frame
                    
                    cv2.putText(combined_frame, f"Camera #{idx+1}", 
                            (x_start + 10, y_start + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    person_count = len(self.detected_persons.get(idx, {}))
                    cv2.putText(combined_frame, f"Tracked: {person_count}", 
                            (x_start + 10, y_start + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    display_count += 1
            
            if display_count > 0:
                fps_count += 1
                if time.time() - fps_start >= 1.0:
                    fps = fps_count / (time.time() - fps_start)
                    fps_count = 0
                    fps_start = time.time()
                
                cv2.putText(combined_frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                total_tracked = self.get_total_tracked_persons()
                cv2.putText(combined_frame, f"Total Unique Persons: {total_tracked}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                cv2.putText(combined_frame, f"Timeout: {CONFIG['person_detection_timeout']}s", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                cv2.putText(combined_frame, "Press 'q' to quit, 't' to retrain, 'c' to clear cache", 
                        (10, CONFIG["display_resolution"][1] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                cv2.imshow('Multi-Camera Face Recognition', combined_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.camera_manager.stop()
                break
            elif key == ord('t'):
                print("Retraining model...")
                self.train_model()
            elif key == ord('c'):
                print("Clearing detected persons cache...")
                self.detected_persons.clear()
            
            time.sleep(0.01)
    def stop(self):
        self.camera_manager.stop()
        
        

if __name__ == "__main__":
    fr = AdvancedFaceRecognition()
    fr.run()
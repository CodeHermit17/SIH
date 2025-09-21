import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from retinaface import RetinaFace
from sklearn.metrics.pairwise import cosine_similarity

class FaceRecognitionEngine:
    def __init__(self):
        """Initializes the FaceAnalysis model."""
        print("Loading InsightFace model... This may take a moment.")
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0)
        print("Model loaded successfully.")

    def get_embedding(self, image):
        """Generates a 512D embedding for a single face image."""
        faces = self.app.get(image)
        if not faces:
            return None
        return faces[0].normed_embedding

    def prepare_known_database(self, known_faces_dir, output_db_dir):
        """Processes all images in the known_faces directory and saves averaged embeddings."""
        os.makedirs(output_db_dir, exist_ok=True)
        for person_name in os.listdir(known_faces_dir):
            person_path = os.path.join(known_faces_dir, person_name)
            if not os.path.isdir(person_path):
                continue

            print(f"[•] Processing {person_name}...")
            embeddings = []
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Resize image for consistency before embedding
                img_resized = cv2.resize(img, (112, 112))
                emb = self.get_embedding(img_resized)
                if emb is not None:
                    embeddings.append(emb)
            
            if embeddings:
                avg_embedding = np.mean(embeddings, axis=0)
                out_path = os.path.join(output_db_dir, f"{person_name}.npy")
                np.save(out_path, avg_embedding)
                print(f"[+] Saved average embedding for {person_name}")
            else:
                print(f"[!] Skipped {person_name}: No valid faces found.")
        print("\n✅ Known faces database is up to date.")

    def detect_and_crop_faces(self, video_path):
        """Detects faces in a video and yields cropped face images."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video file.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(5 * fps) # Process every 5 seconds
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                print(f"-> Scanning video at {frame_count / fps:.2f} seconds...")
                detections = RetinaFace.detect_faces(frame, threshold=0.7)
                if isinstance(detections, dict):
                    for face_info in detections.values():
                        x1, y1, x2, y2 = face_info['facial_area']
                        cropped_face = frame[y1:y2, x1:x2]
                        if cropped_face.size > 0:
                            yield cv2.resize(cropped_face, (112, 112))
            frame_count += 1
        cap.release()

    def compare_embeddings(self, input_embedding, known_db_dir, threshold=0.5):
        """Compares a single input embedding against the known database."""
        if not os.path.exists(known_db_dir) or not os.listdir(known_db_dir):
            return "Unknown", -1 # Return if database is empty

        best_match = "Unknown"
        best_score = -1

        for filename in os.listdir(known_db_dir):
            if filename.endswith('.npy'):
                known_name = os.path.splitext(filename)[0]
                known_emb = np.load(os.path.join(known_db_dir, filename))
                
                score = cosine_similarity([input_embedding], [known_emb])[0][0]
                if score > best_score:
                    best_score = score
                    best_match = known_name
        
        if best_score >= threshold:
            return best_match, best_score
        else:
            return "Unknown", best_score
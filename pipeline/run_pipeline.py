import argparse
import os
from face_lib.engine import FaceRecognitionEngine

DB_DIR = "/home/kp17/Code/Projects/SIH/pipeline/data/output/embeddings/known_db"

def analyze_video(video_path):
    """Runs the full face recognition pipeline on a video."""
    # Validate video file exists
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file does not exist: {video_path}")
        return False
    
    # Check if database directory exists
    if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
        print(f"❌ Error: No known faces database found at: {DB_DIR}")
        print("Please run enroll.py first to add known faces to the database.")
        return False
    
    print(f"🎥 Analyzing video: {video_path}")
    print(f"📚 Using database: {DB_DIR}")
    
    engine = FaceRecognitionEngine()
    
    recognized_people = set()
    face_count = 0
    processed_faces = 0

    # Generator function yields embeddings directly
    embedding_generator = engine.detect_faces_and_embeddings(video_path)

    for input_embedding in embedding_generator:
        face_count += 1
        print(f"  Processing face #{face_count}...")
        
        if input_embedding is not None:
            processed_faces += 1
            print(f"    Got embedding with shape: {input_embedding.shape}")
            # Compare against the known database with debug info
            name, score = engine.compare_embeddings(input_embedding, DB_DIR, threshold=0.6, debug=True)
            
            if name != "Unknown" and name not in recognized_people:
                print(f"✅ Found {name}! (Similarity: {score:.2f})")
                recognized_people.add(name)
            elif name != "Unknown":
                print(f"  Already recognized: {name} (score: {score:.4f})")
        else:
            print(f"  Could not process face #{face_count}")
    
    print(f"\n📊 Statistics:")
    print(f"  Total faces detected: {face_count}")
    print(f"  Faces processed: {processed_faces}")

    print("\n--- Analysis Complete ---")
    if recognized_people:
        print("Recognized individuals in the video:")
        for person in sorted(list(recognized_people)):
            print(f"- {person}")
    else:
        print("No known individuals were recognized in the video.")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a video to recognize known faces.")
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file.')
    args = parser.parse_args()
    
    # Convert to absolute path
    video_path = os.path.abspath(args.video)
    
    success = analyze_video(video_path)
    
    if not success:
        exit(1)

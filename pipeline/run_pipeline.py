import argparse
from face_lib.engine import FaceRecognitionEngine

DB_DIR = "output/embeddings/known_db"

def analyze_video(video_path):
    """Runs the full face recognition pipeline on a video."""
    engine = FaceRecognitionEngine()
    
    recognized_people = set()

    # Generator function yields cropped faces one by one
    face_generator = engine.detect_and_crop_faces(video_path)

    for face_image in face_generator:
        # Get embedding for the detected face
        input_embedding = engine.get_embedding(face_image)
        
        if input_embedding is not None:
            # Compare against the known database
            name, score = engine.compare_embeddings(input_embedding, DB_DIR)
            
            if name != "Unknown" and name not in recognized_people:
                print(f"✅ Found {name}! (Similarity: {score:.2f})")
                recognized_people.add(name)

    print("\n--- Analysis Complete ---")
    if recognized_people:
        print("Recognized individuals in the video:")
        for person in sorted(list(recognized_people)):
            print(f"- {person}")
    else:
        print("No known individuals were recognized in the video.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a video to recognize known faces.")
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file.')
    args = parser.parse_args()
    
    analyze_video(args.video)
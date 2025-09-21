import os
import shutil
from face_lib.engine import FaceRecognitionEngine

KNOWN_FACES_DIR = "data/known_faces"
DB_DIR = "output/embeddings/known_db"

def enroll_new_person():
    """Interactive script to enroll a new person."""
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    
    person_name = input("Enter the name of the person to enroll (e.g., 'john_doe'): ").strip().lower().replace(" ", "_")
    if not person_name:
        print("❌ Name cannot be empty.")
        return

    person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
    if os.path.exists(person_dir):
        print(f"✔️ A directory for '{person_name}' already exists.")
        action = input("Do you want to add more photos? (y/n): ").lower()
        if action != 'y':
            print("Aborting enrollment.")
            return
    else:
        os.makedirs(person_dir)

    print(f"\n✅ A folder has been created at: {person_dir}")
    print("Please add several clear photos of the person to this folder.")
    input("\nPress ENTER when you have finished adding the photos...")

    # Check if photos were added
    if not any(fname.lower().endswith(('.jpg', '.jpeg', '.png')) for fname in os.listdir(person_dir)):
        print("❌ No images found in the directory. Deleting empty folder.")
        shutil.rmtree(person_dir)
        return

    print("\nProcessing images and updating the face database...")
    engine = FaceRecognitionEngine()
    engine.prepare_known_database(KNOWN_FACES_DIR, DB_DIR)

if __name__ == "__main__":
    enroll_new_person()
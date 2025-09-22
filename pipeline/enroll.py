import os
import numpy as np
import cv2
from face_lib.engine import FaceRecognitionEngine

# Input: folder with images of a **single person**
INPUT_DIR = "/home/kp17/Code/Projects/SIH/pipeline/data/known_faces/Manan_shah"
# Output: directory where final .npy will be saved
OUTPUT_DIR = "/home/kp17/Code/Projects/SIH/pipeline/data/output/embeddings/known_db"
# The name of the person (used to name the .npy file)
PERSON_NAME = "manan_shah"

def generate_embedding_for_person():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print(f"❌ No image files found in: {INPUT_DIR}")
        return

    print(f"🧠 Generating embeddings for person: {PERSON_NAME}")
    print(f"📁 From directory: {INPUT_DIR}")
    print(f"💾 Saving to: {os.path.join(OUTPUT_DIR, PERSON_NAME + '.npy')}")

    engine = FaceRecognitionEngine()
    embeddings = []

    for img_name in image_files:
        img_path = os.path.join(INPUT_DIR, img_name)
        print(f"  ➜ Processing {img_name}")

        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"  ❌ Could not read image: {img_path}")
                continue

            embedding = engine.get_embedding(img)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                print(f"  ⚠️ No embedding returned for {img_name}")
        except Exception as e:
            print(f"  ❌ Error processing {img_name}: {e}")

    if embeddings:
        embeddings = np.stack(embeddings)
        avg_embedding = np.mean(embeddings, axis=0)

        out_path = os.path.join(OUTPUT_DIR, f"{PERSON_NAME}.npy")
        np.save(out_path, avg_embedding)
        print(f"\n✅ Saved average embedding to: {out_path}")
    else:
        print("❌ No valid embeddings generated.")

if __name__ == "__main__":
    generate_embedding_for_person()

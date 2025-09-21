# Video Face Recognition Pipeline

This project provides a simple, two-script system to recognize people in videos. It uses the powerful `insightface` library for high-accuracy face recognition.

## Setup

1.  **Clone the repository and navigate into it.**

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create data directories:**
    ```bash
    mkdir -p data/videos
    ```

## Usage

The process is just two simple steps.

### Step 1: Enroll Known Faces

To add people to your recognition database, run the interactive `enroll.py` script.

1.  **Run the script:**
    ```bash
    python enroll.py
    ```
2.  **Enter the person's name** when prompted (e.g., `elon_musk`).
3.  The script will create a folder for them in `data/known_faces/`.
4.  **Add several clear photos** of that person to their new folder.
5.  **Press Enter** in the terminal to confirm. The script will then process the images and update the face database.

Repeat this process for every person you want to be able to recognize.

### Step 2: Analyze a Video

Once your database is ready, you can analyze any video with a single command.

1.  Place your video file in the `data/videos/` directory.
2.  **Run the pipeline:**
    ```bash
    python run_pipeline.py --video data/videos/my_test_video.mp4
    ```

The script will scan the video, detect faces, and print the names of any recognized individuals it finds.
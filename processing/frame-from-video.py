import cv2
import os
from pathlib import Path

# ========= CONFIG =========

# single video OR folder
INPUT_PATH = "videos"

OUTPUT_FOLDER = "frames"

START_NUMBER = 2714

SAVE_EVERY_N_SECONDS = 4

VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv"]

# ==========================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

saved_count = START_NUMBER


def process_video(video_path, saved_count):
    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        print(f"Could not read FPS for {video_path}")
        return saved_count

    frame_interval = int(fps * SAVE_EVERY_N_SECONDS)

    frame_count = 0
    saved_in_video = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = os.path.join(
                OUTPUT_FOLDER,
                f"frame_{saved_count:05d}.jpg"
            )

            cv2.imwrite(filename, frame)

            saved_count += 1
            saved_in_video += 1

        frame_count += 1

    cap.release()

    print(f"{video_path} -> saved {saved_in_video} frames")

    return saved_count


input_path = Path(INPUT_PATH)

# ========= SINGLE VIDEO =========

if input_path.is_file():

    saved_count = process_video(input_path, saved_count)

# ========= FOLDER OF VIDEOS =========

elif input_path.is_dir():

    video_files = []

    for file in sorted(input_path.iterdir()):

        if file.suffix.lower() in VIDEO_EXTS:
            video_files.append(file)

    for video_file in video_files:

        # numbering continues automatically
        saved_count = process_video(video_file, saved_count)

else:
    print("Invalid input path")


print(f"Finished. Total saved frames: {saved_count - START_NUMBER}")

# Code to extract frames from a video and save them as images
import cv2
import os

video_path = "revo_data_store_20260320_081049.mp4"
output_folder = "frames"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)  # frames per second
frame_interval = int(fps)        # save one frame per second

frame_count = 0
saved_count = 0  # Start from the last saved frame number

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print("Done")

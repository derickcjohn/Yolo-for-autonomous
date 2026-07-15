# Yolo26 inference where the segmentation is overlayed on top of the video.
import cv2
import numpy as np
import time
from ultralytics import YOLO
from ultralytics.utils.plotting import colors

# -----------------------
# Paths
# -----------------------
MODEL_PATH = "runs/semantic/train/weights/best.pt"
VIDEO_PATH = "revo-cut.mp4"
OUTPUT_VIDEO = "output_revo-cut.mp4"

# -----------------------
# Load model
# -----------------------
model = YOLO(MODEL_PATH)
n_classes = model.model.nc

ignore_classes = n_classes - 1
# -----------------------
# Open video
# -----------------------
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# Transparency
ALPHA = 0.5

total_inference_time = 0.0
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    start = time.perf_counter()
    # inference
    result = model.predict(
        frame,
        task="semantic",
        conf=0.25,
        verbose=False
    )[0]

    end = time.perf_counter()

    total_inference_time += (end - start)

    overlay = frame.copy()
    # print("result.semantic_mask.data.shape:", result.semantic_mask.data.shape)

    if result.semantic_mask is not None:
        class_map = result.semantic_mask.data.cpu().numpy()

        # Draw each class
        for cls in np.unique(class_map):
            if cls == ignore_classes:
                continue  # Skip the ignored class

            # Class 3 → yellow
            if cls == 3:
                color = (0, 255, 255)  # BGR yellow
            else:
                color = colors(int(cls), bgr=True)

            mask = (class_map == cls)
            overlay[mask] = color

    # Blend with original image
    output_frame = cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0)

    writer.write(output_frame)

cap.release()
writer.release()

avg_time = total_inference_time / frame_count
avg_fps = frame_count / total_inference_time
print(f"Frames processed: {frame_count}")
print(f"Total inference time: {total_inference_time:.3f} s")
print(f"Average inference time/frame: {avg_time*1000:.2f} ms")
print(f"Average inference FPS: {avg_fps:.2f}")

print("Saved to:", OUTPUT_VIDEO)

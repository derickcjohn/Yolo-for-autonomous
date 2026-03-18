import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Input and output video
input_video = "street.mp4"
output_video = "output.mp4"

# Classes you want to keep
classes_to_keep = [0,1,2,3,5,7,9,10,11,12]

cap = cv2.VideoCapture(input_video)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, classes=classes_to_keep)

    # Black frame
    masked_frame = np.zeros_like(frame)

    boxes = results[0].boxes.xyxy.cpu().numpy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)

        # Copy detected region
        masked_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

    out.write(masked_frame)

cap.release()
out.release()

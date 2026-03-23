import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colors

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

# colour = colors()

batch_size = 8
frame_batch = []

def darker(color, factor=0.5):
    return tuple(int(c * factor) for c in color)

while True:

    ret, frame = cap.read()

    if ret:
        frame_batch.append(frame)

    if len(frame_batch) == batch_size or (not ret and len(frame_batch) > 0):

        results = model(frame_batch, classes=classes_to_keep, verbose=False)

        for frame, result in zip(frame_batch, results):

            output = np.zeros_like(frame)

            if result.boxes is not None:

                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)

                for box, cls in zip(boxes, classes):

                    x1, y1, x2, y2 = map(int, box)

                    # ---- Bottom 25% region ----
                    h = y2 - y1
                    y1_new = int(y2 - 0.25 * h)

                    # ---- Color logic ----
                    if cls in [0, 1]:  # person, bicycle
                        color = (0, 0, 255)  # Red (BGR)
                    elif cls in [2, 3, 5, 7]:  # vehicles
                        color = (255, 0, 0)  # Blue (BGR)
                    else:
                        color = colors(cls, True)
                    border_color = darker(color)

                    cv2.rectangle(output, (x1,y1_new), (x2,y2), color, -1)
                    cv2.rectangle(output, (x1,y1_new), (x2,y2), border_color, 3)

            out.write(output)

        frame_batch = []

    if not ret:
        break

cap.release()
out.release()

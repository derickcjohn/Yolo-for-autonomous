# Same as video_mask_batch.py, but only process 1 frame at a time.
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colors

# -------------------- MODELS --------------------
det_model = YOLO("yolo11n.pt")
seg_model = YOLO("yolo11n-seg.pt")

# Input and output video
input_video = "street.mp4"
output_video = "output.mp4"

# Classes you want to keep
# classes_to_keep = [0,1,2,3,5,7,9,10,11,12]
classes_to_keep = [0,1,9,10,11,12]

cap = cv2.VideoCapture(input_video)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# colour = colors()

def darker(color, factor=0.5):
    return tuple(int(c * factor) for c in color)

# -------------------- SEGMENTATION --------------------
def draw_segmentation_on_black(frame, seg_result, canvas):
    """
    Draw segmentation masks on black background
    """
    if seg_result.masks is None:
        return canvas

    masks = seg_result.masks.data.cpu().numpy()
    classes = seg_result.boxes.cls.cpu().numpy().astype(int)

    for mask, cls in zip(masks, classes):
        # Resize mask to frame size
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        if cls in [3]:
            color = (0, 255, 255)
        else:
            color = colors(cls, True)

        canvas[binary_mask == 1] = color

        # -----DRAW BORDER-----
        if cls != 3:  # skip border for vehicles
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, (0, 0, 0), 2)

    return canvas

# -------------------- DETECTION --------------------
def draw_detection_on_canvas(det_result, canvas):
    """
    Draw detection boxes on existing canvas
    """
    if det_result.boxes is None:
        return canvas

    boxes = det_result.boxes.xyxy.cpu().numpy()
    classes = det_result.boxes.cls.cpu().numpy().astype(int)

    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)

        # bottom 25% region
        h = y2 - y1
        y1_new = int(y2 - 0.25 * h)

        if cls in [0, 1]:
            color = (0, 0, 255)
        # elif cls in [2, 3, 5, 7]:
        #     color = (255, 0, 0)
        else:
            color = colors(cls, True)

        border_color = darker(color)

        cv2.rectangle(canvas, (x1, y1_new), (x2, y2), color, -1)
        cv2.rectangle(canvas, (x1, y1_new), (x2, y2), border_color, 3)

    return canvas

# -------------------- MAIN PROCESS --------------------
def process_frame(frame):
    """
    Runs segmentation + detection and combines results
    """
    seg_results = seg_model(frame, verbose=False)[0]
    det_results = det_model(frame, classes=classes_to_keep, verbose=False)[0]

    canvas = np.zeros_like(frame)

    canvas = draw_segmentation_on_black(frame, seg_results, canvas)
    canvas = draw_detection_on_canvas(det_results, canvas)

    return canvas

while True:

    ret, frame = cap.read()
    if not ret:
        break

    output_frame = process_frame(frame)
    out.write(output_frame)

    

cap.release()
out.release()

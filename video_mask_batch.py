# import time
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colors

# -------------------- MODELS --------------------
det_model = YOLO("yolo11n.pt")
seg_model = YOLO("runs/segment/train/weights/best.pt")

# Input and output video
input_video = "street.mp4"
output_video = "output_mask.mp4"

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

batch_size = 8
frame_batch = []

# timing stats
# stats = {
#     'frames': 0,
#     'seg': 0.0,
#     'det': 0.0,
#     'draw_seg': 0.0,
#     'draw_det': 0.0,
#     'total': 0.0,
# }

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
        elif cls in [2, 3, 5, 7]:
            color = (255, 0, 0)
        else:
            color = colors(cls, True)

        border_color = darker(color)

        cv2.rectangle(canvas, (x1, y1_new), (x2, y2), color, -1)
        cv2.rectangle(canvas, (x1, y1_new), (x2, y2), border_color, 3)

    return canvas

# -------------------- MAIN PROCESS --------------------
def process_frame_batch(frame_batch):
    """
    Runs segmentation + detection and combines results
    """
    # seg_start = time.time()
    seg_results = seg_model(frame_batch, verbose=False)
    # seg_end = time.time()
    # stats['seg'] += seg_end - seg_start

    # det_start = time.time()
    det_results = det_model(frame_batch, classes=classes_to_keep, verbose=False)
    # det_end = time.time()
    # stats['det'] += det_end - det_start

    output_frames = []

    for frame, seg_res, det_res in zip(frame_batch, seg_results, det_results):
        canvas = np.zeros_like(frame)

        # draw_seg_start = time.time()
        canvas = draw_segmentation_on_black(frame, seg_res, canvas)
        # stats['draw_seg'] += time.time() - draw_seg_start

        # draw_det_start = time.time()
        canvas = draw_detection_on_canvas(det_res, canvas)
        # stats['draw_det'] += time.time() - draw_det_start

        output_frames.append(canvas)

    return output_frames

while True:

    ret, frame = cap.read()

    if ret:
        frame_batch.append(frame)

    if len(frame_batch) == batch_size or (not ret and len(frame_batch) > 0):

        # batch_start = time.time()
        output_frames = process_frame_batch(frame_batch)
        # batch_end = time.time()

        # stats['frames'] += len(frame_batch)
        # stats['total'] += (batch_end - batch_start)

        for out_frame in output_frames:
            out.write(out_frame)

        frame_batch = []

    if not ret:
        break

cap.release()
out.release()

# if stats['frames'] > 0:
#     print(f"\nProcessed {stats['frames']} frames")
#     print(f"Average segmentation time  : {stats['seg'] / stats['frames']:.4f} sec/frame")
#     print(f"Average detection time     : {stats['det'] / stats['frames']:.4f} sec/frame")
#     print(f"Average segmentation draw  : {stats['draw_seg'] / stats['frames']:.4f} sec/frame")
#     print(f"Average detection draw    : {stats['draw_det'] / stats['frames']:.4f} sec/frame")
#     print(f"Average total batch time   : {stats['total'] / stats['frames']:.4f} sec/frame")
#     print(f"Combined avg (seg+det+draw): {(stats['seg'] + stats['det'] + stats['draw_seg'] + stats['draw_det']) / stats['frames']:.4f} sec/frame")

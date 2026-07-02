# Code to blur faces in a video using a YOLO model for face detection and OpenCV for video processing.
# @author: Derick
import cv2
from ultralytics import YOLO

# Load the face-specific YOLO model
model = YOLO("yolov8n-face.pt") 

input_video = "video-2.mp4"
output_video = "output_blurred.mp4"

cap = cv2.VideoCapture(input_video)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run face detection inference
    results = model(frame, verbose=False)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Extract coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Bound coordinates inside the frame dimensions to prevent slice errors
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_width, x2), min(frame_height, y2)
            
            # Extract face Region of Interest (ROI)
            face_roi = frame[y1:y2, x1:x2]
            
            if face_roi.size > 0:
                # Apply heavy Gaussian Blur (kernel must be odd numbers)
                # Scale kernel size relative to the face size for consistent blur intensity
                ksize_w = int(face_roi.shape[1] * 0.5) | 1  # Ensure odd integer
                ksize_h = int(face_roi.shape[0] * 0.5) | 1
                
                # Keep kernel dimensions within realistic bounds
                ksize_w = max(15, min(ksize_w, 99))
                ksize_h = max(15, min(ksize_h, 99))

                blurred_face = cv2.GaussianBlur(face_roi, (ksize_w, ksize_h), 0)
                
                # Merge blurred ROI back onto original canvas
                frame[y1:y2, x1:x2] = blurred_face

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print("Successfully processed video and saved to", output_video)

# Code to process video frames using SAM3 for semantic segmentation and save results in YOLO format.
import os
import cv2
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor

# ---------------- CONFIG ----------------
frame_dir = "test"
output_img_dir = "datasets/images"
output_lbl_dir = "datasets/labels"

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_lbl_dir, exist_ok=True)

classes = ["crosswalk", "sidewalk", "road"]
class_map = {name: i for i, name in enumerate(classes)}

# SAM3 predictor
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model="sam3.pt",
    half=True,
    save=False,
)

predictor = SAM3SemanticPredictor(overrides=overrides)

# --------------- PROCESS FRAMES ---------------
for img_name in os.listdir(frame_dir):
    if not img_name.lower().endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(frame_dir, img_name)
    image = cv2.imread(img_path)
    h, w = image.shape[:2]

    predictor.set_image(img_path)
    results = predictor(text=classes)

    label_lines = []

    # results[0].masks.data → segmentation masks
    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        for mask, cls_id in zip(masks, class_ids):
            mask = (mask > 0.5).astype(np.uint8)

            # Find contours → polygon
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if len(cnt) < 3:
                    continue

                # Normalize polygon points
                polygon = []
                for point in cnt:
                    x = point[0][0] / w
                    y = point[0][1] / h
                    polygon.append(f"{x:.6f} {y:.6f}")

                line = f"{cls_id} " + " ".join(polygon)
                label_lines.append(line)

    # Save only if at least one object detected
    if len(label_lines) > 0:
        # Save image
        out_img_path = os.path.join(output_img_dir, img_name)
        cv2.imwrite(out_img_path, image)

        # Save label
        label_name = os.path.splitext(img_name)[0] + ".txt"
        out_lbl_path = os.path.join(output_lbl_dir, label_name)

        with open(out_lbl_path, "w") as f:
            for line in label_lines:
                f.write(line + "\n")

print("Done.")

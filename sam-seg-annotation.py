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

target_classes = ["crosswalk", "sidewalk", "road", "vehicle"]
class_map = {
    "crosswalk": 0,
    "sidewalk": 1,
    "pavement": 2,
    "road": 3,
    "vehicle": 4,
    "manhole cover": 5,
    "road marking": 6,
}

prompt_classes = list(class_map.keys())

# SAM3 predictor
overrides = dict(
    conf=0.35,
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
    results = predictor(text=prompt_classes)

    label_lines = []

    # results[0].masks.data → segmentation masks
    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        class_ids_raw = results[0].boxes.cls.cpu().numpy().astype(int)
        names = results[0].names

        # 1. First, merge masks of the same class
        raw_merged = {i: np.zeros((h, w), dtype=np.uint8) for i in set(class_map.values())}
        for mask, raw_id in zip(masks, class_ids_raw):
            target_id = class_map[names[raw_id]]
            binary_mask = (mask > 0.5).astype(np.uint8)
            raw_merged[target_id] = cv2.bitwise_or(raw_merged[target_id], binary_mask)

        
        final_vehicle = raw_merged.get(4, np.zeros((h, w), dtype=np.uint8))
        crosswalk = raw_merged.get(0, np.zeros((h, w), dtype=np.uint8)) & (~final_vehicle)
        sidewalk = raw_merged.get(1, np.zeros((h, w), dtype=np.uint8)) & (~final_vehicle)
        pavement = raw_merged.get(2, np.zeros((h, w), dtype=np.uint8)) & (~final_vehicle)
        road = raw_merged.get(3, np.zeros((h, w), dtype=np.uint8)) & (~final_vehicle)
        manhole = raw_merged.get(5, np.zeros((h, w), dtype=np.uint8)) & (~final_vehicle)
        road_marking = raw_merged.get(6, np.zeros((h, w), dtype=np.uint8)) & (~final_vehicle)

        if np.sum(manhole) > 0:
            # Find individual manhole contours and assign them to road or sidewalk based on location
            manhole_contours, _ = cv2.findContours(manhole, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in manhole_contours:
                single_manhole = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(single_manhole, [cnt], -1, 1, thickness=cv2.FILLED)

                kernel = np.ones((31,31), np.uint8)
                dilated = cv2.dilate(single_manhole, kernel, iterations=1)
                ring = dilated & (~single_manhole)

                context_score = {
                    "road": np.sum(ring & road),
                    "sidewalk": np.sum(ring & sidewalk),
                    "crosswalk": np.sum(ring & crosswalk),
                    "pavement": np.sum(ring & pavement)}

                best_class = max(context_score, key=context_score.get)

                if context_score[best_class] > 0:
                    class_id = class_map[best_class]
                    raw_merged[class_id] = cv2.bitwise_or(raw_merged[class_id], single_manhole)
                else:
                    raw_merged[3] = cv2.bitwise_or(raw_merged[3], single_manhole)

                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 1, thickness=cv2.FILLED)

                # Check overlap with road and sidewalk
                road_overlap = np.sum(mask & road)
                sidewalk_overlap = np.sum(mask & sidewalk)

                if road_overlap > sidewalk_overlap:
                    road = cv2.bitwise_or(road, mask)
                else:
                    sidewalk = cv2.bitwise_or(sidewalk, mask)

            raw_merged[5] = np.zeros((h, w), dtype=np.uint8)

        # Apply semantic remapping first
        road = road | road_marking

        # Crosswalk overrides everything
        sidewalk = sidewalk & (~crosswalk)
        pavement = pavement & (~crosswalk)
        road = road & (~crosswalk)
        
        # -------- RULES --------
        # sidewalk + pavement + road -> sidewalk
        rule1 = sidewalk & pavement & road
        # rule1 =  pavement & road

        # pavement + road -> road
        rule2 = pavement & road & (~sidewalk)
        # rule2 = pavement & road

        # sidewalk + road -> sidewalk
        rule3 = sidewalk & road & (~pavement)
        # rule3 = road & (~pavement)

        # pavement only -> sidewalk
        rule4 = pavement & (~road) & (~sidewalk)
        # rule4 = pavement & (~road)

        # sidewalk only -> sidewalk
        rule5 = sidewalk & (~road)
        rule5 = sidewalk

        # road only -> road
        rule6 = road & (~sidewalk) & (~pavement)
        # rule6 = road & (~pavement)

        # Final outputs
        final_sidewalk = rule1 | rule3 | rule4 | rule5
        final_road = rule2 | rule6
        final_crosswalk = crosswalk

        min_area_ratio = 0.001   # tune this
        min_area = min_area_ratio * (h * w)

        processed_masks = {
            3: final_vehicle,
            0: final_crosswalk, 
            1: final_sidewalk, 
            2: final_road
        }

        # 3. Extract Simplified Polygons
        for target_id, final_mask in processed_masks.items():
            if np.sum(final_mask) == 0:
                continue

            kernel = np.ones((7,7), np.uint8)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
            # Remove tiny regions using connected components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)

            cleaned_mask = np.zeros_like(final_mask)

            for i in range(1, num_labels):  # skip background
                area = stats[i, cv2.CC_STAT_AREA]
                if area > min_area:
                    cleaned_mask[labels == i] = 1

            processed_masks[target_id] = cleaned_mask

            # Find external contours for the cleaned mask
            contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue
                # Simplify geometry to keep YOLO files lightweight
                epsilon = 0.0015 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                if len(approx) < 3: 
                    continue

                # Normalize and format for YOLO
                polygon_points = [f"{p[0][0]/w:.6f} {p[0][1]/h:.6f}" for p in approx]
                label_lines.append(f"{target_id} " + " ".join(polygon_points))

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

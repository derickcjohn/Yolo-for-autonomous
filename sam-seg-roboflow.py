# Code to process images using Roboflow workflow for segmentation and save results in YOLO format.
import os
import cv2
import numpy as np
from shapely.geometry import Polygon
from inference_sdk import InferenceHTTPClient

# ---------------- CONFIG ----------------
frame_dir = "frames"
output_img_dir = "dataset_preprocess/images"
output_lbl_dir = "dataset_preprocess/labels"

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_lbl_dir, exist_ok=True)

class_map = {
    "crosswalk": 0,
    "sidewalk": 1,
    "pavement": 2,
    "road": 3,
    "vehicle": 4,
    "manhole cover": 5,
    "road marking": 6,
    "step": 7,
    "sidewalk border": 8,
    "drain cover": 9,
    "gully grate": 10,
    
}

# Connect to Roboflow
client = InferenceHTTPClient(
  api_url="http://localhost:9001",
  api_key="YlDx8OJor37qPzhHh2Si"
)

# Build classes parameter from class_map
classes_param = ", ".join(class_map.keys())

def polygon_to_mask(points, h, w):
    """
    Convert polygon points to binary mask
    """
    mask = np.zeros((h, w), dtype=np.uint8)

    if len(points) < 3:
        return mask

    pts = np.array([[p["x"], p["y"]] for p in points], dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))

    cv2.fillPoly(mask, [pts], 1)

    return mask
print("Processing frames...")
# --------------- PROCESS FRAMES ---------------
for img_name in os.listdir(frame_dir):
    if not img_name.lower().endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(frame_dir, img_name)
    image = cv2.imread(img_path)
    h, w = image.shape[:2]

    # Run workflow on image
    result = client.run_workflow(
      workspace_name="derick-john-uivii",
      workflow_id="general-segmentation-api-18",
      images={
        "image": img_path
      },
      parameters={
        "classes": classes_param
      },
      
      use_cache=True
    )

    label_lines = []
    predictions = []

        
    # Process results
    if result:
        # Extract masks from workflow result
        # Assuming result contains mask data indexed by class names
        min_area_ratio = 0.001
        min_area = min_area_ratio * (h * w)

        # 1. First, merge masks of the same class from raw predictions
        raw_merged = {i: np.zeros((h, w), dtype=np.uint8) for i in set(class_map.values())}
        
        for item in result:
            preds = item.get("predictions", {}).get("predictions", [])
            predictions.extend(preds)
       # Extract mask data from workflow result
        for pred in predictions:

            class_name = pred.get("class", "").strip()
            
            if class_name not in class_map:
                continue

            target_id = class_map[class_name]

            points = pred.get("points", [])
            binary_mask = polygon_to_mask(points, h, w)

            raw_merged[target_id] = cv2.bitwise_or(
                raw_merged[target_id],
                binary_mask
            )
            if binary_mask.ndim == 3:
                binary_mask = np.squeeze(binary_mask)
            raw_merged[target_id] = cv2.bitwise_or(raw_merged[target_id], binary_mask)

        # 2. Apply remapping rules 
        step = raw_merged.get(7, np.zeros((h, w), dtype=np.uint8))
        sidewalk_border = raw_merged.get(8, np.zeros((h, w), dtype=np.uint8))
        final_vehicle = raw_merged.get(4, np.zeros((h, w), dtype=np.uint8))
        crosswalk = raw_merged.get(0, np.zeros((h, w), dtype=np.uint8)) & (~final_vehicle)
        sidewalk = raw_merged.get(1, np.zeros((h, w), dtype=np.uint8)) & (~final_vehicle & ~step)
        pavement = raw_merged.get(2, np.zeros((h, w), dtype=np.uint8)) & (~final_vehicle & ~step)
        road = raw_merged.get(3, np.zeros((h, w), dtype=np.uint8)) & (~final_vehicle)
        manhole = raw_merged.get(5, np.zeros((h, w), dtype=np.uint8)) & (~final_vehicle)
        road_marking = raw_merged.get(6, np.zeros((h, w), dtype=np.uint8)) & (~final_vehicle)
        drain_cover = raw_merged.get(9, np.zeros((h, w), dtype=np.uint8)) & (~final_vehicle)
        gully_grate = raw_merged.get(10, np.zeros((h, w), dtype=np.uint8)) & (~final_vehicle)
        
        manhole = manhole | drain_cover | gully_grate
        

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

        # pavement + road -> road
        rule2 = pavement & road & (~sidewalk)

        # sidewalk + road -> sidewalk
        rule3 = sidewalk & road & (~pavement)

        # sidewalk + pavement -> sidewalk
        rule4 = sidewalk & pavement & (~road)

        # pavement only -> sidewalk
        rule5 = pavement & (~road) & (~sidewalk)
        # rule4 = pavement & (~road)

        # sidewalk only -> sidewalk
        rule6 = sidewalk & (~road)
        
        # road only -> road
        rule7 = road & (~sidewalk) & (~pavement)
        # rule6 = road & (~pavement)

        # Final outputs
        final_sidewalk = rule1 | rule3 | rule4 | rule5 | rule6
        final_road = rule2 | rule7
        
        processed_masks = {
            3: final_vehicle,
            0: crosswalk, 
            1: final_sidewalk, 
            2: final_road,
            4: sidewalk_border,
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
    
    #print(f"Processed: {img_name}")

print("Done.")

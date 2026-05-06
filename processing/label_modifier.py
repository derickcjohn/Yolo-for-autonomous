"""
YOLO Segmentation Label Batch Editor
-----------------------------------
Works on YOLO segmentation .txt labels.

Processes only files named:
    frame_XXXXX.txt

And only within a selected frame range, for example:
    frame_00205 to frame_01025

Tasks supported:
1. Remap class id (example: 1 -> 2)
2. Remove overlap of one class from another
   Example: subtract class 2 region from class 3 polygons
3. Remove all instances of a class
4 = Remove every Nth frame from images folder
5 = Remove label files if matching image not found
6 = Renumber all files sequentially (frame_00000, frame_00001, ...)
7 = Remove small segments below area threshold
8 = Merge nearby segments of same class (connect small gaps)

Requirements:
    pip install shapely

IMPORTANT:
YOLO segmentation format assumed:
class_id x1 y1 x2 y2 x3 y3 ...

Coordinates are normalized (0 to 1)
"""

import os
import re
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

# =========================
# CONFIG
# =========================

LABEL_FOLDER = r"dataset_preprocess/mapillary-full-yolo.yolov11/train/labels"
IMAGE_FOLDER = r"dataset_preprocess/mapillary-full-yolo.yolov11/train/images"

START_FRAME = 0
END_FRAME   = 4000

# Select task:
TASK = 6

# -------------------------
# TASK 1 CONFIG (remap)
OLD_CLASS = 5
NEW_CLASS = 1

# -------------------------
# TASK 2 CONFIG (subtract overlap)
REMOVE_CLASS = 4      # remove this region
FROM_CLASS   = 1      # from this class

# -------------------------
# TASK 3 CONFIG (delete class)
DELETE_CLASS = 4

# --------------------------------------------------
# TASK 4 : keep only every Nth image frame
KEEP_EVERY_NTH = 2      # every 4th frame

# --------------------------------------------------
MIN_AREA = 1e-4

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# --------------------------------------------------
# TASK 6 : renumber files sequentially
start_index=0

# -------------------------------------------------- 
# TASK 8 CONFIG (merge close segments) 
MERGE_DISTANCE = 0.005   

# =========================
# HELPERS
# =========================

def get_frame_number(filename):
    m = re.match(r"frame_(\d+)", filename)
    if not m:
        return None
    return int(m.group(1))


def in_range(frame_no):
    return START_FRAME <= frame_no <= END_FRAME

def valid_frame_file(filename):
    """
    Accept only frame_XXXXX.txt
    """
    m = re.match(r"frame_(\d+)\.txt$", filename)
    if not m:
        return False, None

    frame_no = int(m.group(1))
    if START_FRAME <= frame_no <= END_FRAME:
        return True, frame_no

    return False, None


def parse_line(line):
    """
    Convert line into class_id and points
    """
    vals = line.strip().split()
    if len(vals) < 7:
        return None

    cls = int(vals[0])
    nums = list(map(float, vals[1:]))

    pts = []
    for i in range(0, len(nums), 2):
        pts.append((nums[i], nums[i + 1]))

    return cls, pts


def polygon_from_pts(pts):
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def poly_to_yolo_lines(cls_id, geom):
    """
    Convert shapely polygon/multipolygon back to YOLO lines
    """
    out = []

    if geom.is_empty:
        return out

    geoms = []

    if isinstance(geom, Polygon):
        geoms = [geom]
    elif isinstance(geom, MultiPolygon):
        geoms = list(geom.geoms)
    else:
        return out

    for g in geoms:
        if g.area < MIN_AREA:
            continue

        coords = list(g.exterior.coords)[:-1]   # remove closing point

        line = [str(cls_id)]
        for x, y in coords:
            line.append(f"{x:.6f}")
            line.append(f"{y:.6f}")

        out.append(" ".join(line))

    return out


# =========================
# TASK 1
# =========================

def task_remap(lines):
    new_lines = []

    for line in lines:
        parsed = parse_line(line)
        if not parsed:
            continue

        cls, pts = parsed

        if cls == OLD_CLASS:
            cls = NEW_CLASS

        coords = []
        for x, y in pts:
            coords.append(f"{x:.6f}")
            coords.append(f"{y:.6f}")

        new_lines.append(str(cls) + " " + " ".join(coords))

    return new_lines


# =========================
# TASK 2
# =========================

def task_remove_overlap(lines):
    objects = []

    for line in lines:
        parsed = parse_line(line)
        if not parsed:
            continue

        cls, pts = parsed
        poly = polygon_from_pts(pts)

        if poly.area < MIN_AREA:
            continue

        objects.append((cls, poly))

    # union of REMOVE_CLASS
    remove_polys = [p for c, p in objects if c == REMOVE_CLASS]

    if not remove_polys:
        return lines

    remove_union = unary_union(remove_polys)

    out_lines = []

    for cls, poly in objects:

        if cls == FROM_CLASS:
            new_poly = poly.difference(remove_union)
            out_lines.extend(poly_to_yolo_lines(cls, new_poly))

        else:
            out_lines.extend(poly_to_yolo_lines(cls, poly))

    return out_lines


# =========================
# TASK 3
# =========================

def task_delete_class(lines):
    new_lines = []

    for line in lines:
        parsed = parse_line(line)
        if not parsed:
            continue

        cls, pts = parsed

        if cls == DELETE_CLASS:
            continue

        coords = []
        for x, y in pts:
            coords.append(f"{x:.6f}")
            coords.append(f"{y:.6f}")

        new_lines.append(str(cls) + " " + " ".join(coords))

    return new_lines


# ==================================================
# LABEL PROCESSING
# ==================================================
def process_labels():
    files = sorted(os.listdir(LABEL_FOLDER))
    count = 0

    for file in files:
        if not file.endswith(".txt"):
            continue
        
        frame_no = get_frame_number(file)
        if frame_no is None or not in_range(frame_no):
            continue
        
        path = os.path.join(LABEL_FOLDER, file)
        
        with open(path, "r") as f:
            lines = f.readlines()
        
        if TASK == 1:
            new_lines = task_remap(lines)
        
        elif TASK == 2:
            new_lines = task_remove_overlap(lines)
        
        elif TASK == 3:
            new_lines = task_delete_class(lines)

        elif TASK == 7:
            new_lines = task_remove_small_segments(lines)

        elif TASK == 8:
            new_lines = task_merge_close_segments(lines)
        
        else:
            return
        
        with open(path, "w") as f:
            for line in new_lines:
                f.write(line + "\n")
        
        count += 1
        print("Processed:", file)
    
    print("Updated label files:", count)

# ==================================================
# TASK 4
# ==================================================
def keep_only_every_nth_images():
    files = sorted(os.listdir(IMAGE_FOLDER))
    count_deleted = 0
    count_kept = 0

    for file in files:
        ext = Path(file).suffix.lower()

        if ext not in IMAGE_EXTS:
            continue

        frame_no = get_frame_number(file)
        if frame_no is None:
            continue

        if not in_range(frame_no):
            continue

        # position relative to selected range
        relative_index = frame_no - START_FRAME + 1

        # keep only every Nth frame
        if relative_index % KEEP_EVERY_NTH != 0:
            count_kept += 1
            print("Kept:", file)
        else:
            path = os.path.join(IMAGE_FOLDER, file)
            os.remove(path)
            count_deleted += 1
            print("Deleted:", file)

    print("Kept image files:", count_kept)
    print("Deleted image files:", count_deleted)

# ==================================================
# TASK 5
# Remove labels if matching image not found
# ==================================================
def remove_labels_without_images():
    image_stems = set()

    # collect all image names without extension
    for file in os.listdir(IMAGE_FOLDER):
        ext = Path(file).suffix.lower()

        if ext in IMAGE_EXTS:
            image_stems.add(Path(file).stem)

    deleted = 0
    kept = 0

    for file in sorted(os.listdir(LABEL_FOLDER)):

        if not file.endswith(".txt"):
            continue

        frame_no = get_frame_number(file)
        if frame_no is None or not in_range(frame_no):
            continue

        stem = Path(file).stem

        if stem not in image_stems:
            path = os.path.join(LABEL_FOLDER, file)
            os.remove(path)
            deleted += 1
            print("Deleted label:", file)
        else:
            kept += 1
            print("Kept label:", file)

    print("Labels kept:", kept)
    print("Labels deleted:", deleted)

# ==================================================
# TASK 6
# Renumber images + labels sequentially
# ==================================================
def renumber_files():
    image_files = []

    # collect valid images
    for file in os.listdir(IMAGE_FOLDER):
        ext = Path(file).suffix.lower()

        if ext not in IMAGE_EXTS:
            continue

        frame_no = get_frame_number(file)
        if frame_no is None:
            continue

        image_files.append((frame_no, file))

    # sort by original frame number
    image_files.sort(key=lambda x: x[0])

    if not image_files:
        print("No image files found.")
        return

    # -------------------------
    # PASS 1 : rename to temp names
    # avoids overwrite conflicts
    # -------------------------
    temp_records = []

    for idx, (old_no, img_file) in enumerate(image_files):
        img_ext = Path(img_file).suffix
        old_img = os.path.join(IMAGE_FOLDER, img_file)

        temp_img = os.path.join(IMAGE_FOLDER, f"__temp__{idx}{img_ext}")
        os.rename(old_img, temp_img)

        lbl_file = Path(img_file).stem + ".txt"
        old_lbl = os.path.join(LABEL_FOLDER, lbl_file)

        temp_lbl = None
        if os.path.exists(old_lbl):
            temp_lbl = os.path.join(LABEL_FOLDER, f"__temp__{idx}.txt")
            os.rename(old_lbl, temp_lbl)

        temp_records.append((idx, temp_img, img_ext, temp_lbl))

    # -------------------------
    # PASS 2 : rename temp -> final
    # -------------------------
    for idx, temp_img, img_ext, temp_lbl in temp_records:

        new_idx = start_index + idx
        new_name = f"frame_{new_idx:05d}"

        final_img = os.path.join(IMAGE_FOLDER, new_name + img_ext)
        os.rename(temp_img, final_img)

        if temp_lbl is not None:
            final_lbl = os.path.join(LABEL_FOLDER, new_name + ".txt")
            os.rename(temp_lbl, final_lbl)

        print(f"Renamed -> {new_name}")

    print("Renumbering complete.")

# =========================
# TASK 7
# Remove small segments
# =========================
def task_remove_small_segments(lines):
    out_lines = []

    for line in lines:
        parsed = parse_line(line)
        if not parsed:
            continue

        cls, pts = parsed
        poly = polygon_from_pts(pts)

        if poly.area < MIN_AREA:
            continue  # drop small segment

        out_lines.extend(poly_to_yolo_lines(cls, poly))

    return out_lines

# =========================
# TASK 8
# Merge nearby segments of same class
# =========================
def task_merge_close_segments(lines):
    class_polys = {}

    # collect polygons per class
    for line in lines:
        parsed = parse_line(line)
        if not parsed:
            continue

        cls, pts = parsed
        poly = polygon_from_pts(pts)

        if poly.area < MIN_AREA:
            continue

        class_polys.setdefault(cls, []).append(poly)

    out_lines = []

    for cls, polys in class_polys.items():

        if not polys:
            continue

        # expand polygons slightly → connect nearby ones
        expanded = [p.buffer(MERGE_DISTANCE) for p in polys]

        # merge
        merged = unary_union(expanded)

        # shrink back
        merged = merged.buffer(-MERGE_DISTANCE)

        # clean geometry
        if not merged.is_valid:
            merged = merged.buffer(0)

        out_lines.extend(poly_to_yolo_lines(cls, merged))

    return out_lines

# ==================================================
# MAIN
# ==================================================

def main():

    if TASK in [1, 2, 3, 7, 8]:
        process_labels()

    elif TASK == 4:
        keep_only_every_nth_images()

    elif TASK == 5:
        remove_labels_without_images()

    elif TASK == 6:
        renumber_files()

    else:
        print("Invalid TASK")

if __name__ == "__main__":
    main()

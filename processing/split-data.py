# Code to split a YOLO dataset into train/val folders, preserving image-label pairs.
#!/usr/bin/env python3
import argparse
import os
import random
import shutil
from pathlib import Path

def split_yolo_dataset(
    src_root: Path,
    dst_root: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
    ext_images=("jpg", "jpeg", "png", "bmp", "tif", "tiff"),
    ext_labels=("txt",),
):
    random.seed(seed)
    src_images = src_root / "images"
    src_labels = src_root / "labels"
    assert src_images.is_dir() and src_labels.is_dir(), "Need images/ and labels/ under src_root"

    dst_root.mkdir(parents=True, exist_ok=True)
    for sub in ("train/images", "train/labels", "val/images", "val/labels"):
        (dst_root / sub).mkdir(parents=True, exist_ok=True)

    # all image files
    img_files = sorted(
        [p for p in src_images.iterdir() if p.suffix.lower().lstrip(".") in ext_images]
    )

    if not img_files:
        raise ValueError("No image files found in src/images")

    random.shuffle(img_files)
    cut = int(len(img_files) * train_ratio)

    def copy_pairs(list_files, split_name):
        for img_path in list_files:
            basename = img_path.stem
            label_path = src_labels / f"{basename}.txt"
            dst_img = dst_root / split_name / "images" / img_path.name
            dst_lbl = dst_root / split_name / "labels" / f"{basename}.txt"

            shutil.copy2(img_path, dst_img)
            if label_path.exists():
                shutil.copy2(label_path, dst_lbl)
            else:
                # optional: skip or create empty label
                (dst_root / split_name / "labels" / f"{basename}.txt").write_text("")
                print(f"Warning: missing label {label_path}")

    copy_pairs(img_files[:cut], "train")
    copy_pairs(img_files[cut:], "val")

    print(f"Split {len(img_files)} images: {cut} train, {len(img_files)-cut} val.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Split YOLO dataset into train/val folders.")
    p.add_argument("--src", type=Path, default=Path("dataset"), help="source root containing images/ labels/")
    p.add_argument("--dst", type=Path, default=Path("dataset_split"), help="destination root")
    p.add_argument("--train-ratio", type=float, default=0.9, help="proportion for training split")
    p.add_argument("--seed", type=int, default=42, help="random seed for reproducible split")
    args = p.parse_args()
    split_yolo_dataset(args.src, args.dst, args.train_ratio, args.seed)

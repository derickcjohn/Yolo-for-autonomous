#!/usr/bin/env python3

import argparse
import random
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


# ----------------------------------------------------
# Command line arguments
# ----------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument(
    "--source-dataset",
    required=True,
    help="Dataset containing images/ and labels/"
)

parser.add_argument(
    "--split-dataset",
    required=True,
    help="Output directory for split dataset"
)

args = parser.parse_args()


SOURCE_DATASET = Path(args.source_dataset)
SPLIT_DATASET = Path(args.split_dataset)

# --------------------------------------------------
# Read Configurations
# --------------------------------------------------
CONFIG_DIR = Path("/app/configs")
with open(CONFIG_DIR / "split-config.yaml", "r") as f:
    split_cfg = yaml.safe_load(f)

with open(CONFIG_DIR / "train-config.yaml", "r") as f:
    train_cfg = yaml.safe_load(f)

with open(CONFIG_DIR / "dataset.yaml") as f:
    dataset_cfg = yaml.safe_load(f)


# ----------------------------------------------------
# Update dataset.yaml path
# ----------------------------------------------------
dataset_cfg["path"] = str(SPLIT_DATASET)

TEMP_DATASET_YAML = CONFIG_DIR / "dataset_runtime.yaml"

with open(TEMP_DATASET_YAML, "w") as f:
    yaml.safe_dump(dataset_cfg, f, sort_keys=False)


# --------------------------------------------------
# Dataset Splitting
# --------------------------------------------------
def split_dataset():

    train_ratio = split_cfg["train_ratio"]
    seed = split_cfg["seed"]

    random.seed(seed)

    src_images = SOURCE_DATASET / "images"
    src_labels = SOURCE_DATASET / "labels"

    if not src_images.exists() or not src_labels.exists():
        raise FileNotFoundError(
            "Dataset must contain images/ and labels/ folders."
        )

    for folder in (
        "train/images",
        "train/labels",
        "val/images",
        "val/labels",
    ):
        (SPLIT_DATASET / folder).mkdir(parents=True, exist_ok=True)

    image_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    )

    images = [
        p for p in src_images.iterdir()
        if p.suffix.lower() in image_extensions
    ]

    if not images:
        raise RuntimeError("No images found.")

    random.shuffle(images)

    split_index = int(len(images) * train_ratio)

    train_images = images[:split_index]
    val_images = images[split_index:]

    def copy_files(image_list, split):

        for img in image_list:

            label = src_labels / f"{img.stem}.txt"

            shutil.copy2(
                img,
                SPLIT_DATASET / split / "images" / img.name,
            )

            if label.exists():
                shutil.copy2(
                    label,
                    SPLIT_DATASET / split / "labels" / label.name,
                )
            else:
                (SPLIT_DATASET / split / "labels" / f"{img.stem}.txt").touch()

    copy_files(train_images, "train")
    copy_files(val_images, "val")

    print(f"Training Images : {len(train_images)}")
    print(f"Validation Images : {len(val_images)}")


# --------------------------------------------------
# YOLO Training
# --------------------------------------------------
def train():

    model = YOLO(train_cfg["model"])

    model.train(

        data=str(TEMP_DATASET_YAML),

        epochs=train_cfg["epochs"],
        imgsz=train_cfg["imgsz"],
        batch=train_cfg["batch"],
        lr0=train_cfg["lr0"],
        lrf=train_cfg["lrf"],
        patience=train_cfg["patience"],

        workers=train_cfg["workers"],
        device=train_cfg["device"],
        optimizer=train_cfg["optimizer"],
        resume=train_cfg["resume"],
        
        project=train_cfg["project"],
        name=train_cfg["run_name"],
        exist_ok=train_cfg["exist_ok"],
        save_dir=train_cfg["save_dir"],

        cache=train_cfg["cache"],
        save=train_cfg["save"],
        save_period=train_cfg["save_period"],
        verbose=train_cfg["verbose"],
    )


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":

    print("\n========== STEP 1 : Splitting Dataset ==========\n")
    split_dataset()

    print("\n========== STEP 2 : Training Model ==========\n")
    train()

    print("\nPipeline Completed Successfully.")
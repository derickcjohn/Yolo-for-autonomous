import cv2
import yaml
import argparse
import tempfile
from pathlib import Path
from datetime import datetime

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


# --------------------------------------------------
# Config
# --------------------------------------------------
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# --------------------------------------------------
# Time conversion
# --------------------------------------------------
def time_to_seconds(time_str):
    """Accept HH:MM:SS, MM:SS or seconds"""

    if time_str is None:
        return None

    try:
        t = datetime.strptime(time_str, "%H:%M:%S")
        return t.hour * 3600 + t.minute * 60 + t.second
    except ValueError:
        pass

    try:
        t = datetime.strptime(time_str, "%M:%S")
        return t.minute * 60 + t.second
    except ValueError:
        pass

    return float(time_str)


# --------------------------------------------------
# Crop Video
# --------------------------------------------------
def crop_video(
    input_path,
    output_path,
    start_time,
    end_time,
):

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(time_to_seconds(start_time) * fps)
    end_frame = int(time_to_seconds(end_time) * fps)

    if start_frame >= end_frame:
        raise ValueError("Invalid crop times")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:

        ret, frame = cap.read()

        if not ret:
            break

        out.write(frame)

    cap.release()
    out.release()


# --------------------------------------------------
# Frame Extraction
# --------------------------------------------------
def process_video(
    video_path,
    output_folder,
    interval_seconds,
    saved_count,
):

    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        print(f"Skipping {video_path}: Invalid FPS")
        return saved_count

    frame_interval = max(1, int(fps * interval_seconds))

    frame_number = 0
    saved = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_number % frame_interval == 0:

            filename = output_folder / f"frame_{saved_count:05d}.jpg"

            cv2.imwrite(str(filename), frame)

            saved += 1
            saved_count += 1

        frame_number += 1

    cap.release()

    print(f"{video_path.name} -> {saved} frames")

    return saved_count


# --------------------------------------------------
# Main Processing
# --------------------------------------------------
def preprocess(input_path, output_path, config):

    input_path = Path(input_path)
    output_path = Path(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    crop_cfg = config["crop"]
    frame_cfg = config["frames"]

    crop_enabled = crop_cfg.get("enabled", False)

    interval = frame_cfg.get("interval_seconds", 2)
    start_number = frame_cfg.get("start_number", 0)

    if input_path.is_file():
        videos = [input_path]

    else:
        videos = sorted(
            f for f in input_path.iterdir()
            if f.suffix.lower() in VIDEO_EXTS
        )

    saved_count = start_number

    with tempfile.TemporaryDirectory() as tmp:

        tmp = Path(tmp)

        for video in videos:

            video_to_process = video

            if crop_enabled:

                if not crop_cfg.get("start") or not crop_cfg.get("end"):
                    raise ValueError(
                        "Crop is enabled but 'start' or 'end' time is missing in config.yaml"
                    )

                cropped = tmp / video.name

                print(
                    f"Cropping {video.name}: "
                    f"{crop_cfg['start']} -> {crop_cfg['end']}"
                )

                crop_video(
                    video,
                    cropped,
                    crop_cfg["start"],
                    crop_cfg["end"],
                )

                video_to_process = cropped

            saved_count = process_video(
                video_to_process,
                output_path,
                interval,
                saved_count,
            )

    print(f"\nFinished. Saved {saved_count - start_number} frames.")


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():

    parser = argparse.ArgumentParser(
        description="Video preprocessing"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input video or folder",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output frames folder",
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Configuration file",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    preprocess(
        args.input,
        args.output,
        config,
    )


if __name__ == "__main__":
    main()
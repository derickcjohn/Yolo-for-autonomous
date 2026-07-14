import cv2
import os
import argparse
from pathlib import Path

VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv"]


def process_video(video_path, output_folder, save_every_n_seconds, saved_count):
    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        print(f"Could not read FPS for {video_path}")
        return saved_count

    frame_interval = max(1, int(fps * save_every_n_seconds))

    frame_count = 0
    saved_in_video = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = os.path.join(
                output_folder,
                f"frame_{saved_count:05d}.jpg"
            )

            cv2.imwrite(filename, frame)

            saved_count += 1
            saved_in_video += 1

        frame_count += 1

    cap.release()

    print(f"{video_path} -> saved {saved_in_video} frames")

    return saved_count


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from a video or all videos in a folder."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input video file or folder containing videos"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Folder where extracted frames will be saved"
    )

    parser.add_argument(
        "--start-number",
        type=int,
        default=0,
        help="Starting frame number (default: 0)"
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=2,
        help="Save one frame every N seconds (default: 2 seconds)"
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    saved_count = args.start_number

    input_path = Path(args.input)

    if input_path.is_file():

        saved_count = process_video(
            input_path,
            args.output,
            args.interval,
            saved_count
        )

    elif input_path.is_dir():

        video_files = sorted(
            f for f in input_path.iterdir()
            if f.suffix.lower() in VIDEO_EXTS
        )

        if not video_files:
            print("No video files found.")
            return

        for video_file in video_files:

            saved_count = process_video(
                video_file,
                args.output,
                args.interval,
                saved_count
            )

    else:
        print("Invalid input path.")
        return

    print(f"Finished. Total saved frames: {saved_count - args.start_number}")


if __name__ == "__main__":
    main()

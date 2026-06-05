# Code to crop a video from one timestamp to another using OpenCV
# Usage: python crop-video.py input_video.mp4 output_video.mp4 start_time end_time
# Example: python crop-video.py input.mp4 output.mp4 00:01:30 00:03:00
# New usage for splitting: python crop-video.py input_video.mp4 output_video.mp4 --split duration_in_minutes
# Example: python crop-video.py input.mp4 output.mp4 --split 5
import cv2
import argparse
from datetime import datetime
import os

def time_to_seconds(time_str):
    """Convert time string (HH:MM:SS or MM:SS) to seconds"""
    try:
        # Try HH:MM:SS format
        time_obj = datetime.strptime(time_str, "%H:%M:%S")
        return int(time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second)
    except ValueError:
        try:
            # Try MM:SS format
            time_obj = datetime.strptime(time_str, "%M:%S")
            return int(time_obj.minute * 60 + time_obj.second)
        except ValueError:
            # Try direct seconds
            return int(float(time_str))

def crop_video(input_path, output_path, start_time, end_time):
    """
    Crop a video from start_time to end_time and save it
    
    Args:
        input_path: Path to input video file
        output_path: Path to output video file
        start_time: Start timestamp (HH:MM:SS, MM:SS, or seconds)
        end_time: End timestamp (HH:MM:SS, MM:SS, or seconds)
    """
    # Open video
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {input_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Convert timestamps to frame numbers
    start_seconds = time_to_seconds(start_time)
    end_seconds = time_to_seconds(end_time)
    
    start_frame = int(start_seconds * fps)
    end_frame = int(end_seconds * fps)
    
    # Validate frame numbers
    if start_frame < 0 or end_frame > total_frames or start_frame >= end_frame:
        print(f"Error: Invalid time range. Video has {total_frames} frames ({total_frames/fps:.2f} seconds)")
        print(f"Requested: {start_seconds}s to {end_seconds}s ({start_frame} to {end_frame} frames)")
        cap.release()
        return False
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not create output video file: {output_path}")
        cap.release()
        return False
    
    # Process video
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = 0
    total_frames_to_process = end_frame - start_frame
    
    print(f"Processing video from {start_seconds}s to {end_seconds}s...")
    print(f"Video properties: {width}x{height} @ {fps} fps")
    print(f"Frames to process: {total_frames_to_process}")
    
    while frame_count < total_frames_to_process:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        out.write(frame)
        frame_count += 1
        
        # Progress indicator
        if frame_count % max(1, total_frames_to_process // 10) == 0:
            print(f"Progress: {frame_count}/{total_frames_to_process} frames")
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"✓ Video saved successfully to: {output_path}")
    print(f"✓ Processed {frame_count} frames of length {frame_count/fps:.2f} seconds")
    return True

def split_video(input_path, output_base_path, duration_minutes):
    """
    Split a video into segments of given duration in minutes
    
    Args:
        input_path: Path to input video file
        output_base_path: Base path for output video files (e.g., output.mp4 will become output_001.mp4, etc.)
        duration_minutes: Duration of each segment in minutes
    """
    # Open video
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {input_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_seconds = total_frames / fps
    segment_duration_seconds = duration_minutes * 60
    segment_frames = int(segment_duration_seconds * fps)
    
    # Set up video writer codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    segment_number = 1
    current_frame = 0
    
    print(f"Splitting video into {duration_minutes}-minute segments...")
    print(f"Video properties: {width}x{height} @ {fps} fps")
    print(f"Total duration: {total_seconds:.2f} seconds")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_base_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    while current_frame < total_frames:
        # Determine output path for this segment
        base_name, ext = os.path.splitext(output_base_path)
        output_path = f"{base_name}_{segment_number:03d}{ext}"
        
        # Calculate end frame for this segment
        end_frame = min(current_frame + segment_frames, total_frames)
        
        # Set up video writer for this segment
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"Error: Could not create output video file: {output_path}")
            cap.release()
            return False
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        
        frames_in_segment = 0
        total_frames_in_segment = end_frame - current_frame
        
        while frames_in_segment < total_frames_in_segment:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            out.write(frame)
            frames_in_segment += 1
            current_frame += 1
            
            # Progress indicator
            if frames_in_segment % max(1, total_frames_in_segment // 10) == 0:
                print(f"Segment {segment_number}: {frames_in_segment}/{total_frames_in_segment} frames")
        
        out.release()
        
        print(f"✓ Segment {segment_number} saved to: {output_path}")
        print(f"✓ Processed {frames_in_segment} frames of length {frames_in_segment/fps:.2f} seconds")
        
        segment_number += 1
    
    cap.release()
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop a video from one timestamp to another or split into segments")
    parser.add_argument("input", help="Input video file path")
    parser.add_argument("output", help="Output video file path (for cropping) or base path (for splitting)")
    parser.add_argument("start", nargs='?', help="Start time (format: HH:MM:SS, MM:SS, or seconds) - required for cropping")
    parser.add_argument("end", nargs='?', help="End time (format: HH:MM:SS, MM:SS, or seconds) - required for cropping")
    parser.add_argument("--split", type=float, help="Split video into segments of given duration in minutes")
    
    args = parser.parse_args()
    
    if args.split:
        if args.start or args.end:
            print("Error: Cannot specify start/end times when using --split")
            exit(1)
        split_video(args.input, args.output, args.split)
    else:
        if not args.start or not args.end:
            print("Error: start and end times are required for cropping")
            exit(1)
        crop_video(args.input, args.output, args.start, args.end)

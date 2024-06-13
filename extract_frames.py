import os
import cv2

def extract_frames(video_path, output_folder, prefix):
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Capture the video
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return
    
    # Frame extraction
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Construct the filename for the frame
        frame_filename = os.path.join(output_folder, f"{prefix}_frame_{frame_number:05d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_number += 1
    
    # Release the video capture object
    cap.release()

# Define source directories
source_dir_A = "/data/daniel/CSE244C/RealVSR/RealVSR/videos/A_videos"
source_dir_B = "/data/daniel/CSE244C/RealVSR/RealVSR/videos/B_videos"

# Define output directories
output_dir_A = "/data/daniel/CSE244C/RealVSR/RealVSR/videos/A_frames"
output_dir_B = "/data/daniel/CSE244C/RealVSR/RealVSR/videos/B_frames"

# Extract frames from A_videos
for video_filename in os.listdir(source_dir_A):
    video_path = os.path.join(source_dir_A, video_filename)
    if os.path.isfile(video_path) and video_filename.endswith(".mov"):
        video_name = os.path.splitext(video_filename)[0]
        extract_frames(video_path, output_dir_A, video_name)

# Extract frames from B_videos
for video_filename in os.listdir(source_dir_B):
    video_path = os.path.join(source_dir_B, video_filename)
    if os.path.isfile(video_path) and video_filename.endswith(".mov"):
        video_name = os.path.splitext(video_filename)[0]
        extract_frames(video_path, output_dir_B, video_name)

print("Frame extraction completed.")

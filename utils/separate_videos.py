import os
import shutil

# Define the source directory and the target directories
source_dir = "/data/daniel/CSE244C/RealVSR/RealVSR/videos"
a_dir = os.path.join(source_dir, "A_videos")
b_dir = os.path.join(source_dir, "B_videos")

# Create target directories if they don't exist
os.makedirs(a_dir, exist_ok=True)
os.makedirs(b_dir, exist_ok=True)

# Iterate over all files in the source directory
for filename in os.listdir(source_dir):
    # Full path to the file
    file_path = os.path.join(source_dir, filename)
    
    # Check if it is a file (and not a directory)
    if os.path.isfile(file_path):
        # Move files based on their suffix
        if filename.endswith("_A.mov"):
            shutil.move(file_path, os.path.join(a_dir, filename))
        elif filename.endswith("_B.mov"):
            shutil.move(file_path, os.path.join(b_dir, filename))

print("Files have been sorted into A_videos and B_videos directories.")

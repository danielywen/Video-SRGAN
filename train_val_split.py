import os
import random
import shutil

def split_data(source_dir, train_dir, val_dir, split_ratio=0.8):
    # Create destination directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get list of files in source directory
    files = os.listdir(source_dir)
    random.shuffle(files)  # Shuffle files randomly
    
    # Calculate split indices
    split_index = int(len(files) * split_ratio)
    
    # Split files into training and validation sets
    train_files = files[:split_index]
    val_files = files[split_index:]
    
    # Copy files to respective directories
    for file in train_files:
        src = os.path.join(source_dir, file)
        dst = os.path.join(train_dir, file)
        shutil.copy(src, dst)
    
    for file in val_files:
        src = os.path.join(source_dir, file)
        dst = os.path.join(val_dir, file)
        shutil.copy(src, dst)

# Example usage
source_directory = "/data/daniel/CSE244C/RealVSR/RealVSR/videos/A_frames"
train_directory = "/data/daniel/CSE244C/RealVSR/RealVSR/videos/train_set"
val_directory = "/data/daniel/CSE244C/RealVSR/RealVSR/videos/val_set"
split_ratio = 0.8  # 80% for training, 20% for validation

split_data(source_directory, train_directory, val_directory, split_ratio)

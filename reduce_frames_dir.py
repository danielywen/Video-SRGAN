import os
import shutil

def extract_files(source_dir, destination_dir, num_files):
    try:
        # Create the destination directory if it does not exist
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        # List all files in the source directory
        files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

        # If there are fewer files than requested, adjust the number of files to move
        num_files = min(num_files, len(files))

        # Move the specified number of files to the destination directory
        for i in range(num_files):
            src_path = os.path.join(source_dir, files[i])
            dest_path = os.path.join(destination_dir, files[i])
            shutil.move(src_path, dest_path)
            print(f"Moved {files[i]} to {destination_dir}")

        print(f"Successfully moved {num_files} files to {destination_dir}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
source_directory = '/data/daniel/CSE244C/RealVSR/RealVSR/videos/A_frames'
destination_directory = '/data/daniel/CSE244C/RealVSR/RealVSR/videos/A_frames_reduced'
number_of_files_to_move = 100

extract_files(source_directory, destination_directory, number_of_files_to_move)

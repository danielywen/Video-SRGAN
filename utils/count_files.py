import os

def count_files_and_directories(directory_path):
    num_files = 0
    num_dirs = 0
    try:
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path):
                num_files += 1
            elif os.path.isdir(item_path):
                num_dirs += 1
        return num_files, num_dirs
    except FileNotFoundError:
        print(f"The directory {directory_path} does not exist.")
        return 0, 0
    except PermissionError:
        print(f"Permission denied to access {directory_path}.")
        return 0, 0

# Example usage
directory_path = '/data/daniel/CSE244C/RealVSR/RealVSR/videos/train_set'
num_files, num_dirs = count_files_and_directories(directory_path)
print(f"There are {num_files} files and {num_dirs} directories in {directory_path}.")

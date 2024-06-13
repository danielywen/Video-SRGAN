import zipfile
import os
import sys

def unzip_file(zip_file_path):
    """
    Unzip a zip file into the same directory where the zip file is located.
    
    Parameters:
    zip_file_path (str): The path to the zip file.
    """
    # Get the directory where the zip file is located
    directory = os.path.dirname(zip_file_path)
    
    # Open the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Extract all the contents into the directory
        zip_ref.extractall(directory)
        print(f"Unzipped '{zip_file_path}' into '{directory}'")

if __name__ == "__main__":
    # Check if the user has provided a zip file path as an argument
    if len(sys.argv) != 2:
        print("Usage: python unzip_script.py <path_to_zip_file>")
        sys.exit(1)
    
    # Get the zip file path from the arguments
    zip_file_path = sys.argv[1]
    
    # Check if the provided path is a valid zip file
    if not zipfile.is_zipfile(zip_file_path):
        print(f"The file '{zip_file_path}' is not a valid zip file.")
        sys.exit(1)
    
    # Unzip the file
    unzip_file(zip_file_path)

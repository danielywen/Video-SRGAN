import os
import shutil
import argparse

def is_image_file(filename):
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    return any(filename.lower().endswith(ext) for ext in IMG_EXTENSIONS)

def gather_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if is_image_file(file):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(output_dir, file)
                
                # Ensure destination filename is unique
                base, extension = os.path.splitext(file)
                counter = 1
                while os.path.exists(dst_path):
                    dst_path = os.path.join(output_dir, f"{base}_{counter}{extension}")
                    counter += 1
                
                shutil.copy(src_path, dst_path)
                print(f"Copied {src_path} to {dst_path}")

def main():
    parser = argparse.ArgumentParser(description='Flatten directory of directories of images into a single directory of images')
    parser.add_argument('--input_dir', default='/data/daniel/CSE244C/RealVSR/RealVSR/GT_test', type=str, help='Path to the input directory containing subdirectories of images')
    parser.add_argument('--output_dir', default='/data/daniel/CSE244C/RealVSR/RealVSR/GT_test_one_dir', type=str, help='Path to the output directory where images will be gathered')

    args = parser.parse_args()

    gather_images(args.input_dir, args.output_dir)
    print(f"Images have been successfully gathered into {args.output_dir}")

if __name__ == "__main__":
    main()

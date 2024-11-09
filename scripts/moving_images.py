import os
import shutil

# Paths to your original images and labels directories
source_images_dir = '../input images/train'  # Directory with all images


# Paths to the destination folders for training data
train_images_dir = '../input images/train/images'


# Make sure the target directories exist
os.makedirs(train_images_dir, exist_ok=True)


# Get list of all image files in the source directory
image_files = [f for f in os.listdir(source_images_dir) if f.endswith(('.jpg', '.png'))]

for image_file in image_files:
    # Define paths for the source image, label, and destination
    source_image_path = os.path.join(source_images_dir, image_file)

    # Define paths for the destination in the train folders
    dest_image_path = os.path.join(train_images_dir, image_file)

    # Move the image file
    shutil.move(source_image_path, dest_image_path)


print("Files have been moved to the train directory.")

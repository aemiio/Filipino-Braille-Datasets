import tensorflow as tf
import numpy as np
import os
from glob import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img

print("hello")
# Define folder paths
images_folder = "../images_joined/images"
augmented_folder = "../augmented_images"

# Number of augmented versions per image
versions_per_images = 5

# Create an ImageDataGenerator for augmentations
datagen = ImageDataGenerator(
    rotation_range=10,  # Rotate images by up to 15 degrees
    width_shift_range=0.1,  # Shift images horizontally
    height_shift_range=0.1,  # Shift images vertically
    shear_range=0.1,  # Shear intensity
    zoom_range=0.1,  # Random zoom
    horizontal_flip=False,  # Braille patterns are directional, so do not flip horizontally
    vertical_flip=False,  # Do not flip vertically
    fill_mode='nearest'  # Fill strategy for newly created pixels
)


def binarize_image(image, threshold=0.5):
    """Convert a grayscale image to binary using a specified threshold."""
    return tf.where(image > threshold, 1.0, 0.0)


def preprocess(image_path, output_filename):
    # Load the image in grayscale and resize it
    image = load_img(image_path, color_mode='grayscale', target_size=(640, 640))
    image = img_to_array(image)  # Convert to a numpy array

    # Normalize the image to [0, 1] range
    image = image / 255.0

    # Binarize the image
    image = binarize_image(image)

    # Reshape for the generator (add batch dimension)
    image = np.expand_dims(image, axis=0)

    # Generate augmented images
    for i in range(versions_per_images):
        augmented_image = datagen.flow(image, batch_size=1)  # Create a generator
        aug_image = next(augmented_image)[0]  # Get the next augmented image

        # Convert back to uint8 and save
        aug_image = (aug_image * 255).astype(np.uint8)  # Scale back to [0, 255]
        output_filename_aug = os.path.join(augmented_folder, f"{os.path.basename(output_filename)}_aug_{i}.jpg")
        save_img(output_filename_aug, aug_image)


# Process all images in the specified folder
for image_path in glob(f"{images_folder}/*.jpg"):
    base_filename = os.path.basename(image_path).split('.')[0]
    output_filename = os.path.join(augmented_folder, f"{base_filename}.jpg")
    preprocess(image_path, output_filename)

print(f"Augmentation complete! Saved to {augmented_folder}.")

import shutil
import cv2
import os
import numpy as np
from imgaug import augmenters as iaa
import random

# Paths
input_folder = '../raw images/train/images'
output_images_folder = '../input images/train/images/'
output_labels_folder = '../input images/train/labels'
labels_folder = '../raw images/train/labels'

# Ensure output folders exist
os.makedirs(output_images_folder, exist_ok=True)
os.makedirs(output_labels_folder, exist_ok=True)

# Define augmentations
rotate_degrees = [2, 3, 4, -4, -3, -2]
brightness_range = [-25, 25]
grayscale_levels = [0.2, 0.5, 0.8]
contrast_range = [0.5, 1.5]
quality_levels = [30, 95]


def adjust_brightness(image, value):
    return cv2.convertScaleAbs(image, alpha=1, beta=value)


def add_salt_and_pepper_noise(image, salt_prob=0.001, pepper_prob=0.001):
    noisy_image = image.copy()
    total_pixels = noisy_image.size

    # Salt noise (white pixels)
    num_salt = int(salt_prob * total_pixels)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in noisy_image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255  # Salt is white

    # Pepper noise (black pixels)
    num_pepper = int(pepper_prob * total_pixels)
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in noisy_image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0  # Pepper is black

    return noisy_image


def apply_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)


def read_yolo_labels(label_path):
    bboxes = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            bboxes.append({
                "class_id": class_id,
                "bbox": [x_center, y_center, width, height]
            })
    return bboxes


def save_yolo_format(image, bboxes, output_image_path, output_labels_path):
    img_name = os.path.basename(output_image_path).replace('.jpg', '')

    # Save image
    cv2.imwrite(output_image_path, image)

    # Save bounding boxes in YOLO format
    with open(output_labels_path, 'w') as f:
        for bbox in bboxes:
            class_id = bbox["class_id"]
            x_center, y_center, width, height = bbox["bbox"]
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def save_random_quality_image(image, img_name):
    for _ in range(3):  # Generate 3 augmented images per original image
        quality = random.randint(quality_levels[0], quality_levels[1])
        output_image_path = os.path.join(output_images_folder, f"{img_name}_quality_{quality}.jpg")
        output_label_path = os.path.join(output_labels_folder, f"{img_name}_quality_{quality}.txt")
        cv2.imwrite(output_image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        shutil.copy(label_path, output_label_path)


# Loop through each image
for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)
    if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    # Print progress
    print(f"Processing image: {img_name}")

    # Load image
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error loading image: {img_path}")
        continue

    label_path = os.path.join(labels_folder, os.path.splitext(img_name)[0] + '.txt')
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        continue

    # Read bounding boxes from the label file
    original_bboxes = read_yolo_labels(label_path)

    # Apply augmentations
    for angle in rotate_degrees:
        print(f"Applying rotation: {angle} degrees")
        rotated_image = iaa.Affine(rotate=angle).augment_image(image)
        output_image_path = os.path.join(output_images_folder, f"{img_name}_rotate_{angle}.jpg")
        output_label_path = os.path.join(output_labels_folder, f"{img_name}_rotate_{angle}.txt")
        save_yolo_format(rotated_image, original_bboxes, output_image_path, output_label_path)

    for _ in range(3):
        brightness = random.randint(brightness_range[0], brightness_range[1])
        print(f"Applying brightness adjustment: {brightness}")
        bright_image = adjust_brightness(image, brightness)
        output_image_path = os.path.join(output_images_folder, f"{img_name}_brightness_{brightness}.jpg")
        output_label_path = os.path.join(output_labels_folder, f"{img_name}_brightness_{brightness}.txt")
        save_yolo_format(bright_image, original_bboxes, output_image_path, output_label_path)

    print("Applying salt and pepper noise")
    salt_pepper_image = add_salt_and_pepper_noise(image)
    output_image_path = os.path.join(output_images_folder, f"{img_name}_salt_pepper_noise.jpg")
    output_label_path = os.path.join(output_labels_folder, f"{img_name}_salt_pepper_noise.txt")
    save_yolo_format(salt_pepper_image, original_bboxes, output_image_path, output_label_path)

    for gray_level in grayscale_levels:
        print(f"Applying grayscale with level: {gray_level}")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image_colored = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        gray_image_colored = cv2.addWeighted(image, gray_level, gray_image_colored, 1 - gray_level, 0)
        output_image_path = os.path.join(output_images_folder, f"{img_name}_gray_{int(gray_level * 100)}.jpg")
        output_label_path = os.path.join(output_labels_folder, f"{img_name}_gray_{int(gray_level * 100)}.txt")
        save_yolo_format(gray_image_colored, original_bboxes, output_image_path, output_label_path)

    print("Applying Gaussian blur")
    blur_image = apply_blur(image)
    output_image_path = os.path.join(output_images_folder, f"{img_name}_blur.jpg")
    output_label_path = os.path.join(output_labels_folder, f"{img_name}_blur.txt")
    save_yolo_format(blur_image, original_bboxes, output_image_path, output_label_path)

    for _ in range(3):
        contrast = random.uniform(contrast_range[0], contrast_range[1])
        print(f"Applying contrast adjustment: {contrast:.2f}")
        contrast_image = iaa.contrast.LinearContrast(contrast).augment_image(image)
        output_image_path = os.path.join(output_images_folder, f"{img_name}_contrast_{contrast:.2f}.jpg")
        output_label_path = os.path.join(output_labels_folder, f"{img_name}_contrast_{contrast:.2f}.txt")
        save_yolo_format(contrast_image, original_bboxes, output_image_path, output_label_path)

    print("Applying random quality adjustments")
    save_random_quality_image(image, img_name)

print("Data augmentation complete. Check the output folders for augmented images and YOLO format bounding boxes.")

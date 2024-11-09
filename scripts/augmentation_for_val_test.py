import cv2
import os
import random
from imgaug import augmenters as iaa

# Paths
input_folder = '../raw images/train/images'
output_images_folder = '../input images/valid/images/'
output_labels_folder = '../input images/valid/labels'
labels_folder = '../raw images/train/labels'

# Ensure output folders exist
os.makedirs(output_images_folder, exist_ok=True)
os.makedirs(output_labels_folder, exist_ok=True)

# Define augmentations
rotate_degrees = [2, -2]
brightness_range = [-25, 25]
contrast_range = [0.5, 1.5]

# Maximum number of augmented images to generate
max_augmented_images = 1500
augmented_images_count = 0


def adjust_brightness(image, value):
    return cv2.convertScaleAbs(image, alpha=1, beta=value)


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


# Get all image filenames in the input folder
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Loop
while augmented_images_count < max_augmented_images:
    # Randomly pick an image
    img_name = random.choice(image_files)
    img_path = os.path.join(input_folder, img_name)

    # Load image
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error loading image: {img_path}")
        continue

    # Find the corresponding label file
    label_path = os.path.join(labels_folder, os.path.splitext(img_name)[0] + '.txt')
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        continue

    # Read bounding boxes from the label file
    original_bboxes = read_yolo_labels(label_path)

    # Randomly select an augmentation
    augmentation_choice = random.choice(['rotate', 'brightness', 'contrast'])

    if augmentation_choice == 'rotate':
        angle = random.choice(rotate_degrees)
        print(f"Applying rotation: {angle} degrees")
        augmented_image = iaa.Affine(rotate=angle).augment_image(image)
        output_image_path = os.path.join(output_images_folder, f"{img_name}_rotate_{angle}.jpg")
        output_label_path = os.path.join(output_labels_folder, f"{img_name}_rotate_{angle}.txt")

    elif augmentation_choice == 'brightness':
        brightness = random.randint(brightness_range[0], brightness_range[1])
        print(f"Applying brightness adjustment: {brightness}")
        augmented_image = adjust_brightness(image, brightness)
        output_image_path = os.path.join(output_images_folder, f"{img_name}_brightness_{brightness}.jpg")
        output_label_path = os.path.join(output_labels_folder, f"{img_name}_brightness_{brightness}.txt")

    elif augmentation_choice == 'contrast':
        contrast = random.uniform(contrast_range[0], contrast_range[1])
        print(f"Applying contrast adjustment: {contrast:.2f}")
        augmented_image = iaa.contrast.LinearContrast(contrast).augment_image(image)
        output_image_path = os.path.join(output_images_folder, f"{img_name}_contrast_{contrast:.2f}.jpg")
        output_label_path = os.path.join(output_labels_folder, f"{img_name}_contrast_{contrast:.2f}.txt")

    # Save the augmented image and labels
    save_yolo_format(augmented_image, original_bboxes, output_image_path, output_label_path)

    # Increment augmented image count
    augmented_images_count += 1

    if augmented_images_count >= max_augmented_images:
        break

print("Data augmentation complete. Check the output folders for augmented images and YOLO format bounding boxes.")

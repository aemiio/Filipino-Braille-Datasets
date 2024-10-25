from PIL import Image
import os

root_directory = 'C:/Users/aemio/Desktop/Filipino Braille Dataset/written braille/canva'


# new_size = (640, 640)


# for dirpath, dirnames, filenames in os.walk(root_directory):
#     for img_file in filenames:
#         if img_file.endswith(".jpg") or img_file.endswith(".png"):
#             img_path = os.path.join(dirpath, img_file)
#             img = Image.open(img_path)


#             img_resized = img.resize(new_size)


#             img_resized.save(img_path)

#             print(f"Resized {img_file} to {new_size} in {dirpath}")

# Initialize a counter for the new filenames
counter = 1

# Walk through all folders and subfolders
for dirpath, dirnames, filenames in os.walk(root_directory):
    for img_file in filenames:
        if img_file.endswith(".jpg") or img_file.endswith(".png"):  # Adjust extensions as needed
            img_path = os.path.join(dirpath, img_file)
            
            # Get the file extension (e.g., '.jpg', '.png')
            file_extension = os.path.splitext(img_file)[1]
            
            # Define the new filename using the counter
            new_filename = f"{counter}{file_extension}"
            new_img_path = os.path.join(dirpath, new_filename)
            
            # Rename the file
            os.rename(img_path, new_img_path)
            
            print(f"Renamed {img_file} to {new_filename}")
            
            # Increment the counter
            counter += 1

from PIL import Image
import os
from collections import Counter
import cv2


# Preprocessing pseudo color of acoustic camera images

def convert_to_grayscale(input_image_path, output_image_path):
    # read acoustic camera images
    image = cv2.imread(input_image_path)

    # Pseudo-color to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # save_gray_images
    cv2.imwrite(output_image_path, gray_image)


# Preprocessing dimensions of acoustic camera images

# Paths for input and output acoustic camera images folders
input_folder = ''
output_folder = ''

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Lists to store acoustic camera images heights and widths
image_heights = []
image_widths = []

# Iterate through all acoustic camera images in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.gif')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Open the acoustic camera image
        img = Image.open(input_path)

        # Get the dimensions of the acoustic camera image
        width, height = img.size
        image_heights.append(height)
        image_widths.append(width)

# Calculate the most common height and width
most_common_height = Counter(image_heights).most_common(1)[0][0]
print('Most common image height:', most_common_height)
most_common_width = Counter(image_widths).most_common(1)[0][0]
print('Most common image width:', most_common_width)

# Resize images to the most common dimensions and save images
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.gif')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Open the acoustic camera image
        img = Image.open(input_path)

        # Resize the acoustic camera images to the most common dimensions
        img = img.resize((most_common_width, most_common_height))

        # Save the resized acoustic image to the output folder
        img.save(output_path)

# Once complete, all images in the output folder will have the most common height and width

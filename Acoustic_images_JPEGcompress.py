from PIL import Image
import os


def compress_image(input_path, output_path, quality=''):
    """
    Compresses a single JPEG acoustic camera image.

    Args:
    - input_path (str): Path to the input acoustic camera image.
    - output_path (str): Path to save the compressed acoustic camera image.
    - quality (int): Compression quality (0-100).

    Returns:
    - bool: True if successful, False otherwise.
    """
    try:
        img = Image.open(input_path)
        img.save(output_path, quality=quality, optimize=True)
        return True
    except Exception as e:
        print(f"Failed to compress acoustic camera image {input_path}: {e}")
        return False


def process_folder(input_folder, output_folder, quality=85):
    """
    Processes all JPEG format of acoustic images in a folder by compressing them.

    Args:
    - input_folder (str): Path to the folder containing input acoustic camera images.
    - output_folder (str): Path to save the compressed acousticc images.
    - quality (int): Compression quality of acoustic images  (0-100).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    num_processed = 0
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            if compress_image(input_path, output_path, quality):
                num_processed += 1

    print(f"Processed {num_processed} images.")


if __name__ == "__main__":
    input_folder = " "  # Path to input acoustic cameras
    output_folder = " "  # Path to save compressed acoustic images
    quality = ''  # Compression quality of acoustic camera images, range (0-100)

    process_folder(input_folder, output_folder, quality)

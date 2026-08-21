import os
import cv2

def resize_image(image, max_width=800):
    """
    Resizes an image proportionally if its width exceeds max_width.
    """
    height, width = image.shape[:2]
    if width > max_width:
        ratio = max_width / float(width)
        new_dimensions = (max_width, int(height * ratio))
        return cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
    return image

def ensure_directory_exists(dir_path):
    """
    Ensures that a specified folder path exists; creates it if missing.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"[System] Created missing directory: '{dir_path}'")

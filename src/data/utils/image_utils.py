import cv2

def load_and_normalize_grayscale_image(img_path):
    """
    Load an image in grayscale, normalize it by dividing by 255, and return the image.
    
    Parameters:
    - img_path (str): Path to the image file.
    
    Returns:
    - numpy.ndarray: Normalized grayscale image.
    """
    return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255

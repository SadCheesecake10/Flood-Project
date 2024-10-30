import numpy as np
import cv2

class CustomTransform:
    """
    Custom transform class for data augmentation.
    
    Parameters:
    - width (int): Width of the transformed image.
    - height (int): Height of the transformed
    
    Returns:
    - transformed image and mask (if available)
    """
    
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def horizontal_flip(self, image, mask=None):
        flipped_image = np.fliplr(image)
        flipped_mask = np.fliplr(mask) if mask is not None else None
        return flipped_image, flipped_mask

    def vertical_flip(self, image, mask=None):
        flipped_image = np.flipud(image)
        flipped_mask = np.flipud(mask) if mask is not None else None
        return flipped_image, flipped_mask

    def random_resize_crop(self, image, mask=None):
        h, w = image.shape[:2]

        x = np.random.randint(0, w - self.width + 1)
        y = np.random.randint(0, h - self.height + 1)

        cropped_image = image[y:y + self.height, x:x + self.width]
        cropped_mask = mask[y:y + self.height, x:x + self.width] if mask is not None else None

        resized_image = cv2.resize(cropped_image, (w, h))
        resized_mask = cv2.resize(cropped_mask, (w, h)) if mask is not None else None

        return resized_image, resized_mask

    def __call__(self, image, mask=None):
        image, mask = self.horizontal_flip(image, mask) if np.random.rand() < 0.5 else (image, mask)
        image, mask = self.vertical_flip(image, mask) if np.random.rand() < 0.5 else (image, mask)
        image, mask = self.random_resize_crop(image, mask)

        return image, mask

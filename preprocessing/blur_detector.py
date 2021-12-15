"""
Module Detect the blur images
"""

import cv2


class BlurDetector:
    """
    Contain code to detect blurness in image
    """
    def detect_blur_in_image(self, image):
        """
        Message:
            Detect blur in image and return result
        Parameters:
            image: Image to check
        Returns:
            bool: Either image is blur or not
        """

        return cv2.Laplacian(image, cv2.CV_64F).var()

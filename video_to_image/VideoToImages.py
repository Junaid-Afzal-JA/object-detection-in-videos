"""
This module contains the converter class.
That convert Video to images.
You can create object of this class and pass it video file
and create a directory containing images.
"""

import os
import cv2
from preprocessing.blur_detector import BlurDetector


class VideoToImages:
    """
    Video to Images converter
    """

    def __init__(self, video_file):
        """
        Message:
            Set the video file to convert
        Parameters:
            self:
            video_file: Video File (.mp4)
        Returns:
            None
        """
        if os.path.isfile(video_file):
            self.video_file = video_file
        else:
            raise FileNotFoundError

    def set_file(self, video_file):
        """
        Message:
            Set the video file to convert
        Parameters:
            self:
            video_file: Video File (.mp4)
        Returns:
            None
        """
        if os.path.isfile(video_file):
            self.video_file = video_file
        else:
            raise FileNotFoundError

    def convert_to_images(self, path_to_save_images):
        """
         Message:
            Convert the video into images
        Parameters:
            path_to_save_images (str): Path of folder to which images save
        Returns:
            None
        """

        blur_detector = BlurDetector()
        video_reader = cv2.VideoCapture(self.video_file)
        if not os.path.exists(path_to_save_images):
            os.mkdir(path_to_save_images)

        image_no = 0
        blur_images_path = path_to_save_images + '/' + 'blur_images'
        os.mkdir(blur_images_path)
        while True:
            ret, image = video_reader.read()
            if ret:
                image_name = f'image{image_no}.jpg'
                x = blur_detector.detect_blur_in_image(image)
                print(x)
                if x > 70:
                    cv2.imwrite(path_to_save_images + '/' + image_name, image)
                else:
                    cv2.imwrite(blur_images_path + '/' + image_name, image)
                image_no += 1
            else:
                break

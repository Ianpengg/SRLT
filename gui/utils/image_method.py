import os
import numpy as np
import cv2
from PIL import Image, ImageEnhance

class ImageMethod:
    def __init__(self, controller):
        self.controller = controller


    def image_to_contrast(self, image):
        image = Image.fromarray(image)
        image = ImageEnhance.Contrast(image)
        image = image.enhance(self.controller.ui.contrast_bar.value()/20)
        image = np.array(image)
        return image

    def image_to_brightness(self, image):
        image = Image.fromarray(image)
        image = ImageEnhance.Brightness(image)
        image = image.enhance(self.controller.ui.brightness_bar.value()/20)
        image = np.array(image)
        image = image 
        return image

    def image_to_threshold(self, image):
        grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, binary_img = cv2.threshold(grayscale_img, self.controller.threshold, 255, cv2.THRESH_BINARY)

        # Convert the binary image to a 3 channel grayscale image
        result_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        
        return result_img


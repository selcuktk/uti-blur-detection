import cv2
import numpy as np

def blurring_gaussian(self, image):
    return cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0)


def blurring_average(self, image):
    return cv2.blur(image, (self.kernel_size, self.kernel_size))


def blurring_median(self, image):
    # medianBlur does not support given type, firstly image translated into uint8, secondly medianBlur function called, finally image is turned into back format
    # part of information is lost in this transformation. But it is not critical since it is an image processing operation rather than being deep learning training data
    image_uint8 = image.astype(np.uint8)
    blurred_uint8 = cv2.medianBlur(image_uint8, self.kernel_size)
    return blurred_uint8.astype(np.float32)


def blurring_bilateral(self, image):
    return cv2.bilateralFilter(image, self.kernel_size, 75, 75)
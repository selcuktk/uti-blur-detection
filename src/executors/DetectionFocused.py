import os
import cv2
import sys
import math
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.helper.executor import Executor
from sdks.novavision.src.base.component import Component
from components.BlurDetection.src.utils.response import build_response_detectionFocused
from components.BlurDetection.src.models.PackageModel import PackageModel


class DetectionFocused(Component):
    """
        The class that performs blurring at detected area in given image.
    """

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.blur_type = self.request.get_param("BlurType")
        self.load_parameters()
        self.image = self.request.get_param("inputImage")
        self.detections = self.request.get_param("inputDetections")

    def load_parameters(self):

        if self.blur_type == "BlurGaussian":
            self.kernel_size = self.request.get_param("KernelSize")
        elif self.blur_type == "BlurAverage":
            self.kernel_size = self.request.get_param("KernelSize")
        elif self.blur_type == "BlurMedian":
            self.kernel_size = self.request.get_param("KernelSize")
        elif self.blur_type == "BlurBilateral":
            self.kernel_size = self.request.get_param("KernelSize")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

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

    def blurring_detections(self, image):
        blurred_image = image.copy()

        for det in self.detections:
            bbox = det.get('boundingBox', {})
            top = int(bbox.get('top', 0))
            left = int(bbox.get('left', 0))
            height = int(bbox.get('height', 0))
            width = int(bbox.get('width', 0))

            y1, y2 = top, top + height
            x1, x2 = left, left + width

            y1, y2 = max(0, y1), min(blurred_image.shape[0], y2)
            x1, x2 = max(0, x1), min(blurred_image.shape[1], x2)

            roi = blurred_image[y1:y2, x1:x2]

            if self.blur_type == "BlurGaussian":
                blurred_roi = self.blurring_gaussian(roi)
            elif self.blur_type == "BlurAverage":
                blurred_roi = self.blurring_average(roi)
            elif self.blur_type == "BlurMedian":
                blurred_roi = self.blurring_median(roi)
            elif self.blur_type == "BlurBilateral":
                blurred_roi = self.blurring_bilateral(roi)
            else:
                raise ValueError(f"Unknown blur type: {self.blur_type}")

            blurred_image[y1:y2, x1:x2] = blurred_roi

        return blurred_image

    def run(self):
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        img.value = self.blurring_detections(img.value)
        self.image = Image.set_frame(img=img, package_uID=self.request.model.uID, redis_db=self.redis_db)
        packageModel = build_response_detectionFocused(context=self)
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()

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
from components.BlurDetection.src.utils.utils import blurring_gaussian, blurring_average, blurring_median, \
    blurring_bilateral


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

    def blurring_detections(self, image):
        blurred_image = image.copy()
        ksize = self.kernel_size

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
                blurred_roi = blurring_gaussian(ksize, roi)
            elif self.blur_type == "BlurAverage":
                blurred_roi = blurring_average(ksize, roi)
            elif self.blur_type == "BlurMedian":
                blurred_roi = blurring_median(ksize, roi)
            elif self.blur_type == "BlurBilateral":
                blurred_roi = blurring_bilateral(ksize, roi)
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


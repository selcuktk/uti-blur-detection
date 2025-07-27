import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.helper.executor import Executor
from sdks.novavision.src.base.component import Component
from components.BlurDetection.src.utils.response import build_response_imageFocused
from components.BlurDetection.src.models.PackageModel import PackageModel
from components.BlurDetection.src.utils.utils import blurring_gaussian, blurring_average, blurring_median, blurring_bilateral


class ImageFocused(Component):
    """
        The class that performs blurring at given image.
    """

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.blur_type = self.request.get_param("BlurType")
        self.load_parameters()
        self.image = self.request.get_param("inputImage")

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

    def blurring(self, image):
        if self.blur_type == "BlurGaussian":
            blurred_image = blurring_gaussian(image)
        elif self.blur_type == "BlurAverage":
            blurred_image = blurring_average(image)
        elif self.blur_type == "BlurMedian":
            blurred_image = blurring_median(image)
        elif self.blur_type == "BlurBilateral":
            blurred_image = blurring_bilateral(image)
        else:
            raise ValueError(f"Unknown blur type: {self.blur_type}")

        return blurred_image

    def run(self):
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        img.value = self.blurring(img.value)
        self.image = Image.set_frame(img=img, package_uID=self.request.model.uID, redis_db=self.redis_db)
        packageModel = build_response_imageFocused(context=self)
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()

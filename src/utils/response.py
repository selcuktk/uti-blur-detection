from sdks.novavision.src.helper.package import PackageHelper
from components.BlurDetection.src.models.PackageModel import PackageModel, PackageConfigs, ConfigExecutor, ImageFocusedOutputs, \
    ImageFocusedResponse, ImageFocusedExecutor, DetectionFocusedOutputs, DetectionFocusedResponse, DetectionFocusedExecutor, OutputImage


def build_response_imageFocused(context):
    outputImage = OutputImage(value=context.image)
    Outputs = ImageFocusedOutputs(outputImage=outputImage)
    imageFocusedResponse = ImageFocusedResponse(outputs=Outputs)
    imageFocusedExecutor = ImageFocusedExecutor(value=imageFocusedResponse)
    executor = ConfigExecutor(value=imageFocusedExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel


def build_response_detectionFocused(context):
    outputImage = OutputImage(value=context.image)
    Outputs = DetectionFocusedOutputs(outputImage=outputImage)
    detectionFocusedResponse = DetectionFocusedResponse(outputs=Outputs)
    detectionFocusedExecutor = DetectionFocusedExecutor(value=detectionFocusedResponse)
    executor = ConfigExecutor(value=detectionFocusedExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel

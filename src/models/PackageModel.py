from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Image, Inputs, Configs, Outputs, Response, Request, Output, Input, \
    Config, Detection


class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"


class InputDetections(Input):
    name: Literal["inputDetections"] = "inputDetections"
    value: List[Detection]
    type: str = "list"

    class Config:
        title = "Detections"


class OutputImage(Output):
    name: Literal["outputImage"] = "outputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"


class KernelSize(Config):
    """
    Low blur level:
    Mediam blur level
    High Blur level:
    """
    name: Literal["KernelSize"] = "KernelSize"
    value: int = Field(ge=3, le=21, default=5)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    @validator("value")
    @classmethod()
    def check_odd_value(cls, v:int):
        value = v.get('value')
        if value is not None and value % 2 == 0:
            raise ValueError("Kernel size must be an odd number.")
        return v

    class Config:
        title = "Kernel Size"



class BlurGaussian(Config):
    kernelSize: KernelSize
    name: Literal["BlurGaussian"] = "BlurGaussian"
    value: Literal["BlurGaussian"] = "BlurGaussian"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Gaussian"


class BlurAverage(Config):
    kernelSize: KernelSize
    name: Literal["BlurAverage"] = "BlurAverage"
    value: Literal["BlurAverage"] = "BlurAverage"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Average"


class BlurMedian(Config):
    kernelSize: KernelSize
    name: Literal["BlurMedian"] = "BlurMedian"
    value: Literal["BlurMedian"] = "BlurMedian"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Median"


class BlurBilateral(Config):
    kernelSize: KernelSize
    name: Literal["BlurBilateral"] = "BlurBilateral"
    value: Literal["BlurBilateral"] = "BlurBilateral"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Bilateral"


class BlurType(Config):
    """
        Blur Type can be selected from here
    """
    name: Literal["BlurType"] = "BlurType"
    value: Union[BlurGaussian, BlurAverage, BlurMedian, BlurBilateral]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Blur Type"


class ImageFocusedInputs(Inputs):
    inputImage: InputImage


class DetectionFocusedInputs(Inputs):
    inputImage: InputImage
    inputDetections: InputDetections


class ImageFocusedConfigs(Configs):
    blurType: BlurType


class DetectionFocusedConfigs(Configs):
    blurType: BlurType


class ImageFocusedOutputs(Outputs):
    outputImage: OutputImage


class DetectionFocusedOutputs(Outputs):
    outputImage: OutputImage


class ImageFocusedRequest(Request):
    inputs: Optional[ImageFocusedInputs]
    configs: ImageFocusedConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class DetectionFocusedRequest(Request):
    inputs: Optional[DetectionFocusedInputs]
    configs: DetectionFocusedConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class ImageFocusedResponse(Response):
    outputs: ImageFocusedOutputs


class DetectionFocusedResponse(Response):
    outputs: DetectionFocusedOutputs


class ImageFocusedExecutor(Config):
    name: Literal["ImageFocused"] = "ImageFocused"
    value: Union[ImageFocusedRequest, ImageFocusedResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Image Focused Blur"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class DetectionFocusedExecutor(Config):
    name: Literal["DetectionFocused"] = "DetectionFocused"
    value: Union[DetectionFocusedRequest, DetectionFocusedResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Detection Focused Blur"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[DetectionFocusedExecutor, ImageFocusedExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {
            "target": "value"
        }


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["component"] = "component"
    name: Literal["BlurDetection"] = "BlurDetection"

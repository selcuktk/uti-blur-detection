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

class High(Config):
    name: Literal["High"] = "High"
    value: Literal["High"] = "High"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "High"


class Medium(Config):
    name: Literal["Medium"] = "Medium"
    value: Literal["Medium"] = "Medium"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Medium"


class Low(Config):
    name: Literal["Low"] = "Low"
    value: Literal["Low"] = "Low"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Low"


class BlurLevel(Config):
    name: Literal["BlurLevel"] = "BlurLevel"
    value: Union[Low, Medium, High]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Blur Level"


class KernelSize(Config):
    name: Literal["KernelSize"] = "KernelSize"
    value: float = Field(default=5, ge=0, le=10)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "KernelSize"


class Default(Config):
    blurLevel: BlurLevel
    name: Literal["Default"] = "Default"
    value: Literal["Default"] = "Default"
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Default"


class Customized(Config):
    kernelSize: KernelSize
    name: Literal["Customized"] = "Customized"
    value: Literal["Customized"] = "Customized"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Customized"


class Gaussian(Config):
    name: Literal["Gaussian"] = "Gaussian"
    value: Union[Customized, Default]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Gaussian"


class DetectionFocusedInputs(Inputs):
    inputImage: InputImage
    inputDetections: InputDetections


class ImageFocusedInputs(Inputs):
    inputImage: InputImage


class DetectionFocusedConfigs(Configs):
    gaussian: Gaussian


class ImageFocusedConfigs(Configs):
    gaussian: Gaussian


class DetectionFocusedOutputs(Outputs):
    outputImage: OutputImage


class ImageFocusedOutputs(Outputs):
    outputImage: OutputImage


class DetectionFocusedRequest(Request):
    inputs: Optional[DetectionFocusedInputs]
    configs: DetectionFocusedConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class ImageFocusedRequest(Request):
    inputs: Optional[ImageFocusedInputs]
    configs: ImageFocusedConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class DetectionFocusedResponse(Response):
    outputs: DetectionFocusedOutputs


class ImageFocusedResponse(Response):
    outputs: ImageFocusedOutputs


class DetectionFocusedExecutor(Config):
    name: Literal["DetectionFocused"] = "DetectionFocused"
    value: Union[DetectionFocusedRequest, DetectionFocusedResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "DetectionFocused"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ImageFocusedExecutor(Config):
    name: Literal["ImageFocused"] = "ImageFocused"
    value: Union[ImageFocusedRequest, ImageFocusedResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "ImageFocused"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[ImageFocusedExecutor, DetectionFocusedExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["component"] = "component"
    name: Literal["BlurDetection"] = "BlurDetection"

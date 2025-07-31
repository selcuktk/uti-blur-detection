
<img width="740" height="198" alt="novavision" src="https://github.com/user-attachments/assets/6a38785a-807c-4d5c-b70d-d95050288e53" />
<br><br>

![blur](https://github.com/user-attachments/assets/4c932c6f-748a-4549-992c-d44f1082c21e)

The Blur Component is a configurable image preprocessing module designed to apply various types of blurring effects to images. 
## Build with
- OpenCV
- Wsl
- Redis


## Features

- ✅ Supports multiple blur types:
    - Gaussian Blur
    - Average Blur
    - Median Blur
    - Bilateral Filter
- 🎛️ Configurable **kernel size** for tuning the blur intensity
- 📦 Easily integrates with existing Novavision-ai packages 


## Usage

There are 2 executors in Blur. 

- ImageFocused: Blurs the given image completely 
- DetectionFocused: Blurs the detections on the given image 

1   . ImageFocused: It takes 1 input as InputImage and gives 1 output as OutputImage. Use of the ImageFocused can be seen at the image below.

![if_input](https://github.com/user-attachments/assets/b74abf1b-9c14-4a98-b876-5bd4f651e59c)


2   . DetectionFocused: It takes 2 input as InputImage and InputDetections and gives 1 output as OutputImage. Use of the DetectionFocused can be seen at the image below.

![detection_workflow](https://github.com/user-attachments/assets/3a977105-203a-43a0-bdf8-0d123d5014b9)


## Supported Configurations

| Parameter     | Type | Description                                           | Default      |
| ------------- | ---- | ----------------------------------------------------- | ------------ |
| `BlurType`   | str  | Type of blur: `"Gaussian"`, `"Average"`, `"Median"`, `"Bilateral"` | `"Gaussian"` |
| `KernelSize` | int  | Size of the kernel (must be odd) (<52)                 | `25`          |

## Screenshots
1   .  ImageFocused with **BlurType:** Average & **KernelSize:** 51 
<br><br>
![if_average_51](https://github.com/user-attachments/assets/36c88595-878e-4304-89a4-bb7f16c248ec) 
<br><br><br><br>
2.  DetectionFocused with **BlurType:** Gaussian & **KernelSize:** 51
<br><br>
![df_gaussian_51](https://github.com/user-attachments/assets/ef140c6a-59f8-4b79-84fe-8633cc460037)


# About the Project
The acoustic camera is a type of 2D forward-looking sonar (FLS) that can operate in low-visibility underwater environments.
However, the resolution of acoustic_images captured by acoustic cameras is very low, which is a common problem in sonar image processing.
Different from the previous CNN-based acoustic_image_super-resolution (SR) reconstruction, this project tested the reconstruction performance based on GAN.
Ultimately, this project will serve the preprocessing of acoustic_camera_images together with the association, denoising and matching blocks.

# Prerequisites
opencv-python=4.8.1

basicsr=1.4.2

facexlib=0.3.0

gfpgan=1.3.8

matplotlib=3.7.3

numpy=1.24.4

Pillow=10.2.0

torch=1.12.1

torchvision=0.13.1

tqdm=4.66.1

# Installation
Considering the complexity of acoustic camera image attributes, the model initialization and data flow pipeline are based on [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), please refer to it for initialization.

# Usage
Utilities are defined in python scripts used as modules.
Unlike optical sensors, acoustic cameras generate grayscale images by emitting sound waves and analyzing the time delay, intensity, and phase difference of the echoes. Therefore, it is necessary to introduce a high-order degradation process, in which the noise and quality loss parts are critical. The training and fine-tuning parameters of the model can be set according to the manual.

# Future Improvements
Acoustic camera image association, image denoising, image SR reconstruction and feature matching are the preprocessing steps of our schedule, serving acoustic camera applications such as underwater mosaicking, 3D reconstruction, and SLAM.

# Acknowledgements
The analysis and processing of acoustic images refer to [aris-file-sdk](https://github.com/SoundMetrics/aris-file-sdk), and the SR reconstruction pipeline is implemented based on [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).
And we borrowed some code from [wgan-gp](https://github.com/EmilienDupont/wgan-gp).

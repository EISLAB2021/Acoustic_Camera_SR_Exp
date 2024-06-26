### Acoustic camera (AC) preprocessing external scripts

from PIL import Image, ImageEnhance
import os
from collections import Counter
import cv2
import numpy as np
import pywt
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# Can freely choose to preprocess the acoustic camera images before feeding them into the SR model
# For example, noise suppression, contrast enhancement and sharpening

# Preprocessing pseudo color of acoustic camera images

def convert_to_grayscale(input_image_path, output_image_path):
    # read acoustic camera images
    image = cv2.imread(input_image_path)

    # Pseudo-color to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # save_gray_images
    cv2.imwrite(output_image_path, gray_image)


# Enhance contrast of acoustic camera images

def enhance_contrast_pillow(ACimage, factor=1.2):
    """
    By Pillow's inner function
    :param image: Input acoustic camera image
    :param factor: Contrast enhancement factor
    """
    enhancer = ImageEnhance.Contrast(ACimage)
    enhanced_image = enhancer.enhance(factor)
    return enhanced_image


def enhance_contrast_hist_eq(ACimage):
    """
    By histogram equalization
    :param image: Input acoustic camera image
    """
    image_array = np.array(ACimage)
    if len(image_array.shape) == 2:  # Grayscale image
        equalized_image_array = cv2.equalizeHist(image_array)
    else:  # Color image
        y_cr_cb = cv2.cvtColor(image_array, cv2.COLOR_RGB2YCrCb)
        y_cr_cb[:, :, 0] = cv2.equalizeHist(y_cr_cb[:, :, 0])
        equalized_image_array = cv2.cvtColor(y_cr_cb, cv2.COLOR_YCrCb2RGB)
    return Image.fromarray(equalized_image_array)


def enhance_contrast_clahe(ACimage, clip_limit=1.5, tile_grid_size=(8, 8)):
    """
    By CLAHE (Contrast Limited Adaptive Histogram Equalization).
    :param image: Input acoustic camera image
    :param clip_limit: One threshold for contrast limiting
    :param tile_grid_size: The size of grid for histogram equalization
    """
    image_array = np.array(ACimage)
    if len(image_array.shape) == 2:  # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        clahe_image_array = clahe.apply(image_array)
    else:  # Color image
        y_cr_cb = cv2.cvtColor(image_array, cv2.COLOR_RGB2YCrCb)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        y_cr_cb[:, :, 0] = clahe.apply(y_cr_cb[:, :, 0])
        clahe_image_array = cv2.cvtColor(y_cr_cb, cv2.COLOR_YCrCb2RGB)
    return Image.fromarray(clahe_image_array)


# Noise Suppression of acoustic camera images

# 1. Mean filter
def mean_filter(img, kernel_size=(3, 3)):
    return cv2.blur(img, kernel_size)


# 2. Gaussian filter
def gaussian_filter(img, kernel_size=(5, 5), sigma=1):
    return cv2.GaussianBlur(img, kernel_size, sigma)


# 3. Median filter
def median_filter(img, kernel_size=5):
    return cv2.medianBlur(img, kernel_size)


# 4. Bilateral filter
def bilateral_filter(img, d=5, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


# 5. Anisotropic diffusion filter
def anisotropic_diffusion(img, alpha=0.125, kappa=0.05, niters=5):
    return cv2.ximgproc.anisotropicDiffusion(img, alpha, kappa, niters)


# 6. Wavelet denoising
def wavelet_denoise(image_path, threshold=30, wavelet='db2'):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    coeffs = pywt.dwt2(image, wavelet)
    cA, (cH, cV, cD) = coeffs
    cA = pywt.threshold(cA, threshold, mode='soft')
    cH = pywt.threshold(cH, threshold, mode='soft')
    cV = pywt.threshold(cV, threshold, mode='soft')
    cD = pywt.threshold(cD, threshold, mode='soft')
    denoised_coeffs = (cA, (cH, cV, cD))
    return pywt.idwt2(denoised_coeffs, wavelet)


# 7. Total Variation Denoising
def total_variation(img: torch.Tensor) -> torch.Tensor:
    # reference:[1]https://en.wikipedia.org/wiki/Total_variation
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(img)}")
    if len(img.shape) < 3 or len(img.shape) > 4:
        raise ValueError(f"Expected input tensor to be of ndim 3 or 4, but got {len(img.shape)}.")
    pixel_dif1 = img[..., 1:, :] - img[..., :-1, :]
    pixel_dif2 = img[..., :, 1:] - img[..., :, :-1]
    reduce_axes = (-3, -2, -1)
    res1 = pixel_dif1.abs().sum(dim=reduce_axes)
    res2 = pixel_dif2.abs().sum(dim=reduce_axes)
    return res1 + res2


class TotalVariation(nn.Module):
    def forward(self, img_path_or_tensor):
        if isinstance(img_path_or_tensor, str):
            img = Image.open(img_path_or_tensor).convert('RGB')
            img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
        elif isinstance(img_path_or_tensor, torch.Tensor):
            img = img_path_or_tensor
        else:
            raise TypeError(f"Input type is not supported. Got {type(img_path_or_tensor)}")
        return total_variation(img)


# usage demo
# img = cv2.imread('ACimage.png')
#
# # Convert Pseudo Color to grayscale
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # Apply various denoising techniques on acoustic camera images
# img_mean = mean_filter(img)
# img_gaussian = gaussian_filter(img)
# img_median = median_filter(img)
# img_bilateral = bilateral_filter(img)
# img_anisotropic = anisotropic_diffusion(img)
# denoised_image_wavelet = wavelet_denoise('demo.png')
#
# # Total Variation Denoising
# tv = TotalVariation()
# output_tv = tv('ACimage.png')

# #  Visualization of denoised acoustic camera
# cv2.imshow('Mean Filter on acoustic camera images', img_mean)
# cv2.imshow('Gaussian Filter on acoustic camera images', img_gaussian)
# cv2.imshow('Median Filter on acoustic camera images', img_median)
# cv2.imshow('Bilateral Filter on acoustic camera images', img_bilateral)
# cv2.imshow('Anisotropic Diffusion on acoustic camera images', img_anisotropic)
# cv2.imshow('Wavelet Denoise on acoustic camera images', denoised_image_wavelet)
# cv2.imshow('Total Variation Denoise on acoustic camera images', output_tv.permute(1, 2, 0).numpy())


# Sharpening of acoustic camera images

def simple_kernel_sharpening(ACimage):
    """
    Sharpens the image using a simple kernel.

    :param image: Input acoustic camera image
    :return: Sharpened acoustic camera image
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(ACimage, -1, kernel)
    return sharpened


def unsharp_masking(ACimage, sigma=1.0, strength=1.5):
    """
    Sharpens the acoustic camera image using unsharp masking technique.

    :param image: Input acoustic camera image
    :param sigma: Gaussian blur sigma
    :param strength: Strength of sharpening
    :return: Sharpened acoustic camera image
    """
    blurred = cv2.GaussianBlur(ACimage, (0, 0), sigma)
    sharpened = cv2.addWeighted(ACimage, 1 + strength, blurred, -strength, 0)
    return sharpened


def laplacian_sharpening(ACimage):
    """
    Sharpens the acoustic camera image using Laplacian kernel.

    :param image: Input acoustic camera image
    :return: Sharpened acoustic camera image
    """
    laplacian = cv2.Laplacian(ACimage, cv2.CV_64F)
    sharpened = cv2.convertScaleAbs(ACimage - laplacian)
    return sharpened


# Visualization

# Apply different sharpening techniques
# simple_sharpened = simple_kernel_sharpening(image)
# unsharp_sharpened = unsharp_masking(image)
# laplacian_sharpened = laplacian_sharpening(image)


# Preprocessing dimensions of acoustic camera images

# Paths for input and output acoustic camera images folders
input_folder = ''
output_folder = ''

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Lists to store acoustic camera images heights and widths
image_heights = []
image_widths = []

# Iterate through all acoustic camera images in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.gif')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Open the acoustic camera image
        img = Image.open(input_path)

        # Get the dimensions of the acoustic camera image
        width, height = img.size
        image_heights.append(height)
        image_widths.append(width)

# Calculate the most common height and width
most_common_height = Counter(image_heights).most_common(1)[0][0]
print('Most common image height:', most_common_height)
most_common_width = Counter(image_widths).most_common(1)[0][0]
print('Most common image width:', most_common_width)

# Resize images to the most common dimensions and save images
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.gif')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Open the acoustic camera image
        img = Image.open(input_path)

        # Resize the acoustic camera images to the most common dimensions
        img = img.resize((most_common_width, most_common_height))

        # Save the resized acoustic image to the output folder
        img.save(output_path)

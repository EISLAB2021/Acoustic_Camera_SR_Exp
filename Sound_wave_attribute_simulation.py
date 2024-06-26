import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

def generate_speckle_noise(ACimage, shape_parameter='', scale_parameter=''):
    """
    Generate speckle noise and add it to the acoustic camera image

    :param image: Input one acoustic camera image
    :param shape_parameter: Shape parameter for the gamma distribution
    :param scale_parameter: Scale parameter for the noise, controlling noise intensity
    :return: Acoustic image with added complex noise
    """
    # Generate random noise following a Gamma distribution
    gamma = np.random.gamma(shape_parameter, scale_parameter, ACimage.shape)
    gamma = gamma.reshape(ACimage.shape[0], ACimage.shape[1]).astype('uint8')

    # Multiply the noise with the input acoustic camera (AC) imagery
    noisy_AC_image = ACimage + ACimage * gamma

    # Clip the image values to be between 0 and 255
    noisy_AC_image = np.clip(noisy_AC_image, 0, 255)
    noisy_AC_image = noisy_AC_image.astype(np.uint8)

    return noisy_AC_image

def random_add_speckle_noise_on_AC_IMG(ACimage, shape_parameter='', scale_parameter='', clip=True, rounds=False):
    """
    Generate speckle noise and add it to the acoustic camera image.

    :param image: Input acoustic camera (AC )image
    :param shape_parameter: Shape parameter for the noise
    :param scale: Scale parameter for the noise, controls noise intensity on acoustic camera
    :param clip: Whether to clip the output values
    :param rounds: Whether to round the output values
    :return: AC Image with added noise
    """
    # Generate random noise following Gamma distribution
    gamma = np.random.gamma(shape_parameter, scale_parameter, ACimage.shape)
    gamma = gamma.reshape(ACimage.shape[0], ACimage.shape[1], ACimage.shape[2], ACimage.shape[3])
    gamma = torch.from_numpy(gamma).to(ACimage.device)

    # Multiply noise with the input acoustic camera image
    image = ACimage * 255.0
    out = image + image * gamma

    # Clip the acoustic camera image values to the range [0, 255]
    if clip and rounds:
        out = torch.clamp((out).round(), 0, 255) / 255.
    elif clip:
        out = torch.clamp(out, 0, 255) / 255.
    elif rounds:
        out = (out).round() / 255.

    return out.float()

def generate_gaussian_noise(ACimage, mean=0, sigma=''):
    """
    Generate Gaussian noise and add it to the image

    :param image: Input image
    :param mean: Mean of the Gaussian distribution
    :param sigma: Standard deviation of the Gaussian distribution
    :return: Image with added noise
    """
    gaussian = np.random.normal(mean, sigma, ACimage.shape)
    noisy_image = ACimage + gaussian
    noisy_image = np.clip(noisy_image, 0, 255)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image


def generate_poisson_noise(ACimage):
    """
    Generate Poisson noise and add it to the image

    :param image: Input acoustic camera image
    :return: AC Image with added noise
    """
    vals = len(np.unique(ACimage))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy_image = np.random.poisson(ACimage * vals) / float(vals)
    noisy_image = np.clip(noisy_image, 0, 255)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image


def plot_histogram(image, title):
    """
    Plot the histogram of the acoustic camera image

    :param image: Input acoustic camera image
    :param title: Title of the plot
    """
    plt.figure()
    plt.hist(image.ravel(), bins=256, range=[0, 256])
    plt.title(title)
    plt.xlabel('Pixel Intensity on the AC')
    plt.ylabel('Frequency Value')
    plt.show()


def plot_fourier_magnitude(ACimage, title):
    """
    Plot the magnitude of the Fourier Transform of the acoustic camera image

    :param image: Input acoustic camera image
    :param title: Title of the plot
    """
    # Compute the 2D Fourier Transform
    f = np.fft.fft2(ACimage)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    plt.figure()
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


# Read the AC image
image = cv2.imread('tire.png', 0)  # GRAY

# noise control
noise_user_parameter_m = ''
noise_user_parameter_n = ''

# Add speckle noise on AC image
noisy_image = generate_speckle_noise(image, shape_parameter=noise_user_parameter_m,
                                     scale_parameter=noise_user_parameter_n)

# Add Gaussian noise on AC image

noise_user_parameter_sigma = ''
gaussian_noisy_image = generate_gaussian_noise(image, mean=0, sigma=noise_user_parameter_sigma)

# Add Poisson noise on AC image
poisson_noisy_image = generate_poisson_noise(image)

# Display the raw acoustic camera image and the noisy added image
plt.figure(figsize=(10, 5))
# Raw AC image
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Noisy added AC image
plt.subplot(2, 2, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image')
plt.axis('off')

# Display the histograms
plot_histogram(image, 'Histogram of Raw AC Image')
plot_histogram(noisy_image, 'Histogram of Noisy added AC Image')

# Display the Fourier magnitude spectra
plot_fourier_magnitude(image, 'Fourier Magnitude of Raw AC Image')
plot_fourier_magnitude(noisy_image, 'Fourier Magnitude of Noisy added AC Image')

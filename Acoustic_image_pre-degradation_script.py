# A simple preprocessing script for external degradation of acoustic camera images
# Can add various degradation sub-modules along this pipeline

import cv2
import torch
import numpy as np
import torchvision
from DiffJPEG import DiffJPEG  # refer to released DiffJPEG
import matplotlib.pyplot as plt
import torch.nn.functional as F


def img_to_tensor(ACimages, convert_bgr_to_rgb=True, to_float32=True):
    """
    Convert numpy array or list of arrays to PyTorch tensor(s).

    Args:
        ACimages (numpy.ndarray or list): Input acoustic camera (AC) image or list of images.
        convert_bgr_to_rgb (bool): Convert BGR to RGB format if True.
        to_float32 (bool): Convert image data to float32.

    Returns:
        torch.Tensor or list: Converted tensor(s) of acoustic camera images.
    """

    def _to_tensor(image, convert_bgr_to_rgb, to_float32):
        if image.shape[2] == 3 and convert_bgr_to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if to_float32:
            image = image.astype(np.float32) / 255.0
        tensor_image = torch.from_numpy(image.transpose(2, 0, 1))
        return tensor_image

    if isinstance(ACimages, list):
        return [_to_tensor(img, convert_bgr_to_rgb, to_float32) for img in ACimages]
    else:
        return _to_tensor(ACimages, convert_bgr_to_rgb, to_float32)


def generate_speckle_noise(ACimage, shape_parameter=0.4, scale_parameter=0.6):
    """
    Generate speckle noise and add it to the acoustic camera image.

    Args:
        ACimage (numpy.ndarray): Input acoustic camera (AC) image.
        shape_parameter (float): Shape parameter for gamma distribution.
        scale_parameter (float): Scale parameter for gamma distribution.

    Returns:
        numpy.ndarray: Noisy acoustic camera image.
    """
    gamma = np.random.gamma(shape_parameter, scale_parameter, ACimage.shape[:2]).astype(ACimage.dtype)
    gamma = np.expand_dims(gamma, axis=-1)
    noisy_AC_image = ACimage + ACimage * gamma
    noisy_AC_image = np.clip(noisy_AC_image, 0, 255).astype(np.uint8)

    return noisy_AC_image


# Free to add your own custom degradation sub-modules here,

# Can refer to the acoustic image processing scripts we provide


class AcousticCameraDatadegration:
    def __init__(self, img_path):
        """
        Initialize the AcousticCameraDatadegration class.

        Args:
            img_path (str): Path to the input acoustic camera image.
        """
        super(AcousticCameraDatadegration, self).__init__()
        self.path = img_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.jpeg_range = [20, 60]  # example range, adjust as needed
        self.jpeger = DiffJPEG(differentiable=False).to(self.device)

    @torch.no_grad()
    def load_image(self):
        """
        Load and preprocess the input acoustic camera image.

        Returns:
            torch.Tensor: Preprocessed AC image tensor.
            str: Image path.
        """
        gt_path = self.path
        img_gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)  # Assuming color image
        img_gt = img_to_tensor(img_gt)
        return img_gt, gt_path

    @torch.no_grad()
    def synthesis(self):
        """
        Perform image synthesis with noise and JPEG degradation.

        Returns:
            LR: Degraded low resolution acoustic camera image tensor.
        """
        img_gt, gt_path = self.load_image()
        img_gt = img_gt.unsqueeze(0).to(self.device)

        for _ in range(2):
            img_gt_np = img_gt.mul(255).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
            noisy_img_gt_np = generate_speckle_noise(img_gt_np[0], shape_parameter=0.4, scale_parameter=0.6)  # uer set
            img_gt = torch.tensor(noisy_img_gt_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img_gt = img_gt.to(self.device)
            img_gt = torch.clamp(img_gt, -0.5, 1.5)  # user set
            img_gt = F.relu(img_gt)
            jpeg_p = img_gt.new_zeros(img_gt.size(0)).uniform_(*self.jpeg_range)
            img_gt = self.jpeger(img_gt, quality=jpeg_p.to(self.device))

        LR_ACimage = torch.clamp((img_gt * 255.0).round(), 0, 255) / 255.0
        LR_ACimage = LR_ACimage.squeeze(0).contiguous()

        return LR_ACimage


if __name__ == '__main__':
    LR_ACimg = AcousticCameraDatadegration('ACimage.jpg')  # Acoustic camera input
    result = LR_ACimg.synthesis()

    # Save the processed image
    torchvision.utils.save_image(result, 'degraded_LR_ACimage.jpg')

    # Visualizing statistical feature maps on degraded LR acoustic camera images
    # Read the processed image
    processed_image = cv2.imread('degraded_LR_ACimage.jpg')

    # Grayscale Histogram
    gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))  # Keep pseudo color display
    plt.title('LR acoustic Image')

    plt.subplot(2, 2, 2)
    plt.hist(gray_image.ravel(), bins=256, range=(0, 255))
    plt.title('Grayscale Histogram on acoustic image')

    # Gradient Map
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    plt.subplot(2, 2, 3)
    plt.imshow(gradient_magnitude, cmap='gray')
    plt.title('Gradient Map on acoustic image')

    # Fourier Transform Image
    fft_image = np.fft.fftshift(np.fft.fft2(gray_image))
    fft_image = np.log(np.abs(fft_image) + 1)
    plt.subplot(2, 2, 4)
    plt.imshow(fft_image, cmap='jet')
    plt.title('Fourier Transform')

    # Adjust subplot layout
    plt.tight_layout()

    # Save the entire image analysis result
    plt.savefig('LR_ACimage_analysis.png')

    # Optionally, display the image analysis result
    plt.show()

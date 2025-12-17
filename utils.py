import numpy as np
from scipy.ndimage import (gaussian_filter, median_filter, convolve)        
    
def lpf(img: np.ndarray, sigma_lpf: float=5.0) -> np.ndarray:
    """
    Apply low-pass Gaussian filter to a numpy image
    Set Gaussian sigma with self.sigma_lpf

    img: input image (3D array)

    Returns: filtered image
    """
    return gaussian_filter(img, sigma=sigma_lpf)

def conv3d(h, img):
    """
    3D convolution with nearest edge padding

    parameter: img: input 3D image
    parameter: h: 3D convolution kernel
    returns: 3D convolved image
    """
    return convolve(img, h, mode='nearest')

def noise_sigma_map(noise_img: np.ndarray, signal_mask: np.ndarray):
    """
    Create a local noise sigma map using MAD robust estimator on the residual image
    Approximate noise residual distribution as N(0, sigma_n)
    """
    noise_img = noise_img * signal_mask
    sigma_img = 1.4826 * median_filter(np.abs(noise_img), size=5)

    return sigma_img

def signal_mask_otsu(img_noisy: np.ndarray):
    """
    Create a signal mask using Otsu thresholding

    img_noisy: input noisy image (3D array)

    Returns: signal mask (boolean 3D array)
    """
    import ants
    img_ai = ants.from_numpy(img_noisy)
    signal_mask_ai = ants.segmentation.otsu_segmentation(img_ai, k=1)
    signal_mask = signal_mask_ai.numpy() == 1

    return signal_mask
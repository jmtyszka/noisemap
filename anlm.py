"""
Adaptive Non-Local Means (ANLM) denoising and noise estimation for Rician MRI data.
Based on the method of 
"""

import ants
import numpy as np
from scipy.ndimage import median_filter

from .utils import lpf

def anlm_denoise(img_noisy: np.ndarray, sigma_lpf: float=5.0):

    img_noise_ai = ants.from_numpy(img_noisy)

    # Create a signal mask using Otsu thresholding
    signal_mask_ai = ants.segmentation.otsu_segmentation(img_noise_ai, k=1)
    signal_mask = signal_mask_ai.numpy() == 1
    
    denoised_ants_ai = ants.denoise_image(
        image=img_noise_ai,
        mask=signal_mask_ai,
        noise_model='Rician',
        shrink_factor=2,
        p=1,
        r=2
    )
    
    img_denoised = denoised_ants_ai.numpy()
    img_noise = img_noisy - img_denoised

    # Local MAD noise sigma map estimation
    # Scale MAD to sigma_n for Gaussian white noise approximation x 1.4826
    abs_noise = np.abs(img_noise)
    sigma_n = median_filter(abs_noise, size=5) * 1.4826

    snr = np.zeros_like(img_noisy)
    snr[signal_mask] = img_denoised[signal_mask] / (sigma_n[signal_mask] + 1e-12)

    signal_mask = signal_mask
    img_denoised = img_denoised
    img_noise = img_noise
    img_sigma_n_lpf = lpf(sigma_n, sigma_lpf=sigma_lpf)
    img_snr_lpf = lpf(snr, sigma_lpf=sigma_lpf)

    return img_denoised, img_noise, img_sigma_n_lpf, img_snr_lpf, signal_mask
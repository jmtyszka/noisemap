"""
Adaptive Non-Local Means (ANLM) denoising and noise estimation for Rician MRI data.
Based on the method of 
"""

import ants
import numpy as np
from scipy.ndimage import median_filter

from .utils import lpf, signal_mask_otsu, snr_map

def anlm_est(img_noisy: np.ndarray, sigma_lpf: float=10.0):

    # Division-by-zero insurance
    small_float = 1e-12

    print("\nRunning ANLM Rician noise estimation ...")

    signal_mask, mask_thresh, percent_coverage = signal_mask_otsu(img_noisy)
    print(f"Airspace mask threshold {mask_thresh:0.2f}, coverage {percent_coverage:0.2f} %")

    # Convert to ANTs image for ANLM denoising    
    img_noisy_ai = ants.from_numpy(img_noisy)
    signal_mask_ai = ants.from_numpy(signal_mask.astype(np.uint8))
    
    print("ANLM denoising ...")
    img_denoised_ai = ants.denoise_image(
        image=img_noisy_ai,
        mask=signal_mask_ai,
        noise_model='Rician',
        p=1,
        r=2
    )
    
    # Extract denoised image as numpy array
    img_denoised = img_denoised_ai.numpy()

    # Estimate noise sigma and SNR maps from noisy and denoised images
    print("Estimating noise sigma and SNR maps...")
    img_snrmap, img_sigmamap, img_noise = snr_map(img_noisy, img_denoised, signal_mask)

    return img_denoised, img_noise, img_sigmamap, img_snrmap, signal_mask
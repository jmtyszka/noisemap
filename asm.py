"""
Adaptive Soft Matching (ASM) denoising and noise estimation for Rician MRI data.
Based on the method of Pierrick Coupé, José V. Manjón, Montserrat Robles, and Louis D. Collins. 
Adaptive Multiresolution Non-Local Means Filter for 3D MR Image Denoising. IET Image Processing, 6(5):558–568, July 2012. 
Implemented in the DiPy package.
"""

import numpy as np
from dipy.denoise.nlmeans import non_local_means, estimate_sigma
from dipy.denoise.adaptive_soft_matching import adaptive_soft_matching
from scipy.ndimage import median_filter

from .utils import noise_sigma_map

def asm_est(img_noisy: np.ndarray, signal_mask: np.ndarray):
    """
    ASM denoising and noise estimation for Rician MRI data.

    """

    # Estimate noise sigma from a signal-free region using MAD
    noise_roi = img_noisy[~signal_mask]
    mad = np.median(np.abs(noise_roi - np.median(noise_roi)))

    sigma_dipy = estimate_sigma(img_noisy, N=32)[0]
    print(f"DiPy global sigma_n estimate [UNUSED] {sigma_dipy:0.1f}")
    
    sigma_mad = mad / 0.6745  # Convert MAD to standard deviation
    print(f"Air Region MAD sigma_n estimate {sigma_mad:0.1f}")

    print("NLM denoising with small patches...")
    den_small = non_local_means(
        img_noisy, sigma=sigma_mad, mask=signal_mask, patch_radius=1, block_radius=1, rician=True
    )

    print("NLM denoising with large patches...")
    den_large = non_local_means(
        img_noisy, sigma=sigma_mad, mask=signal_mask, patch_radius=2, block_radius=1, rician=True
    )

    print("Adaptive soft matching...")
    img_denoised = adaptive_soft_matching(img_noisy, den_small, den_large, sigma_mad)

    noise_img = img_noisy - img_denoised
    sigma_img = noise_sigma_map(noise_img, signal_mask)

    return img_denoised, noise_img, sigma_img, sigma_mad
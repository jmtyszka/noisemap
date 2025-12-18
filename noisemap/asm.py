"""
Adaptive Soft Matching (ASM) denoising and noise estimation for Rician MRI data.
Based on the method of Pierrick Coupé, José V. Manjón, Montserrat Robles, and Louis D. Collins. 
Adaptive Multiresolution Non-Local Means Filter for 3D MR Image Denoising. IET Image Processing, 6(5):558–568, July 2012. 
Implemented in the DiPy package.
"""

import numpy as np
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.denoise.adaptive_soft_matching import adaptive_soft_matching
from scipy.ndimage import median_filter

from .utils import signal_mask_otsu, airspace_noise_est, noise_sigma_map

def asm_est(img_noisy: np.ndarray):
    """
    ASM denoising and noise estimation for Rician MRI data.
    """

    print("\nRunning ASM Rician noise estimation ...")

    # Estimate signal mask
    signal_mask, _ = signal_mask_otsu(img_noisy)

    # Estimate noise sigma using airspace method
    sigma_n = airspace_noise_est(img_noisy)
    print(f"Airspace sigma_n estimate {sigma_n:0.1f}")

    sigma_n_dipy = estimate_sigma(img_noisy, N=32)[0]
    print(f"DiPy sigma_n estimate [UNUSED] {sigma_n_dipy:0.1f}")

    print("NLM denoising with small patches...")
    den_small = nlmeans(
        img_noisy, sigma=sigma_n, mask=signal_mask, patch_radius=1, block_radius=1, rician=True
    )

    print("NLM denoising with large patches...")
    den_large = nlmeans(
        img_noisy, sigma=sigma_n, mask=signal_mask, patch_radius=2, block_radius=1, rician=True
    )

    print("Adaptive soft matching...")
    img_denoised = adaptive_soft_matching(img_noisy, den_small, den_large, sigma_n)
    img_noise = img_noisy - img_denoised
    img_sigmamap = noise_sigma_map(img_noise, signal_mask)
    
    # Image SNR map estimation
    img_snrmap = img_denoised / (img_sigmamap + 1e-12)

    return img_denoised, img_noise, img_sigmamap, img_snrmap, signal_mask
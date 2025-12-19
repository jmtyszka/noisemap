"""
Adaptive Soft Coefficient Matching (ASCM) denoising and noise estimation for Rician MRI data.

Based on the method of Coupé et al.:
Coupé, P., Manjón, J. V., Robles, M., & Collins, L. D. (2012).
Adaptive Multiresolution Non-Local Means Filter for 3D MR Image Denoising.
IET Image Processing, 6(5), 558–568. https://doi.org/10.1049

Implemented in the DiPy package
https://docs.dipy.org/dev/examples_built/preprocessing/denoise_ascm.html
"""

import numpy as np
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.denoise.adaptive_soft_matching import adaptive_soft_matching

from .utils import signal_mask_otsu, airspace_noise_est, snr_map

def ascm_est(img_noisy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ASCM denoising and noise estimation for Rician MRI data.
    """

    # Division-by-zero insurance
    small_float = 1e-12

    print("\nRunning ASCM Rician noise estimation ...")

    # Estimate signal mask
    signal_mask, mask_thresh, percent_coverage = signal_mask_otsu(img_noisy)
    print(f"Signal mask threshold {mask_thresh:0.2f}, coverage {percent_coverage:0.2f} %")

    # Estimate noise sigma using airspace method
    sigma_n = airspace_noise_est(img_noisy)
    print(f"Airspace sigma_n estimate {sigma_n:0.1f}")

    sigma_n_dipy = estimate_sigma(img_noisy, disable_background_masking=True, N=32)[0]
    print(f"DiPy sigma_n estimate [UNUSED] {sigma_n_dipy:0.1f}")

    print("NLM denoising with small patches...")
    den_small = nlmeans(
        img_noisy, sigma=sigma_n, mask=signal_mask, patch_radius=1, block_radius=1, rician=True
    )

    print("NLM denoising with large patches...")
    den_large = nlmeans(
        img_noisy, sigma=sigma_n, mask=signal_mask, patch_radius=2, block_radius=1, rician=True
    )

    print("Adaptive soft coefficient matching...")
    img_denoised = adaptive_soft_matching(img_noisy, den_small, den_large, sigma_n)

    # Estimate noise sigma and SNR maps from noisy and denoised images
    print("Estimating noise sigma and SNR maps...")
    img_snrmap, img_sigmamap, img_noise = snr_map(img_noisy, img_denoised, signal_mask)

    return img_denoised, img_noise, img_sigmamap, img_snrmap, signal_mask
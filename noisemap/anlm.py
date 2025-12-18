"""
Adaptive Non-Local Means (ANLM) denoising and noise estimation for Rician MRI data.
Based on the method of 
"""

import ants
import numpy as np
from scipy.ndimage import median_filter

from .utils import lpf, signal_mask_otsu, noise_sigma_map

def anlm_est(img_noisy: np.ndarray, sigma_lpf: float=10.0):

    print("\nRunning ANLM Rician noise estimation ...")

    signal_mask, _ = signal_mask_otsu(img_noisy)
    signal_mask_ai = ants.from_numpy(signal_mask.astype(np.uint8))

    img_noise_ai = ants.from_numpy(img_noisy)
    
    print("  ANLM denoising ...")
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

    # Noise sigma map estimation
    print("  Estimating noise sigma map ...")
    img_sigmamap = noise_sigma_map(img_noise, signal_mask)

    # Low pass filter sigma map
    img_sigmamap = lpf(img_sigmamap, sigma_spat=sigma_lpf)

    img_snrmap = np.zeros_like(img_noisy)
    img_snrmap[signal_mask] = img_denoised[signal_mask] / (img_sigmamap[signal_mask] + 1e-12)

    return img_denoised, img_noise, img_sigmamap, img_snrmap, signal_mask
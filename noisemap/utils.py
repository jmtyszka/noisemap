import numpy as np
from scipy.ndimage import (gaussian_filter, median_filter, convolve)    
import skimage.filters as skf
    
def lpf(img: np.ndarray, sigma_spat: float=10.0) -> np.ndarray:
    """
    Spatial domain low-pass Gaussian filter of a 3D numpy array

    img: input image (3D array)
    sigma_spat: spatial domain Gaussian sigma for low-pass filtering

    Returns: filtered image
    """
    return gaussian_filter(img, sigma=sigma_spat)

def conv3d(h, img):
    """
    3D convolution with nearest edge padding

    parameter: img: input 3D image
    parameter: h: 3D convolution kernel
    returns: 3D convolved image
    """
    return convolve(img, h, mode='nearest')

def snr_map(
        img_noisy: np.ndarray,
        img_denoised: np.ndarray,
        signal_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate local noise sigma from the noisy and denoised images within the signal mask.
    Assumes a Rician noise model, so the raw sigma should be corrected for SNR bias.

    img_noisy: input noisy image (3D array)
    img_denoised: input denoised image (3D array)
    signal_mask: boolean signal mask (3D array)
    """

    # Division-by-zero insurance
    small_float = 1e-12

    # Check for empty signal mask
    if np.sum(signal_mask) == 0:
        signal_mask, mask_thresh, percent_coverage = signal_mask_otsu(img_denoised)
        print(f"Warning: Empty signal mask provided")
        print(f"Generated new mask with threshold {mask_thresh:0.2f}, coverage {percent_coverage:0.2f} %")

    # Signed noise residual within signal mask
    img_noise = (img_noisy - img_denoised) * signal_mask
    
    # Estimate local noise sigma from local median of absolute noise residuals
    # Use pre-simulated lookup table for SNR-dependent median to sigma conversion

    snr_lut = [0.00000000, 0.40816327, 0.81632653, 1.02040816, 1.32653061, 2.14285714, 5.0000000]
    mad_lut = [1.17743669, 0.81755008, 0.59415234, 0.58440642, 0.60464279, 0.65227609, 0.6716113]

    # Kernel size for median filtering
    k = 5

    # Median filter absolute noise residuals to estimate local noise level
    img_noise_medfilt = median_filter(np.abs(img_noise), size=k)

    # Initial sigma_n map estimation within signal mask using Gaussian (high SNR) assumption
    img_sigmamap = img_noise_medfilt / 0.6745
    img_snrmap = img_denoised / (img_sigmamap + 0.1) * signal_mask

    # TODO: Implement SNR-dependent correction using lookup table and interpolation
    # from scipy.interpolate import PchipInterpolator
    # pchip_interp = PchipInterpolator(snr_lut, mad_lut, extrapolate=True)
    # Iterative sigma and SNR map correction for Rician bias for SNR-depedent sigma/MAD factor
    # n_iter = 3
    # for it in range(n_iter):
    #     img_sigmamap = img_noise_medfilt / 0.6745
    #     # Limit correction to SNR < 5
    #     correction_factors = pchip_interp(img_snrmap)
    #     img_sigmamap = img_noise_medfilt / correction_factors
    #     img_snrmap = img_denoised / (img_sigmamap + small_float) * signal_mask

    return img_snrmap, img_sigmamap, img_noise

def signal_mask_otsu(img_noisy: np.ndarray, nclasses: int=4) -> tuple[np.ndarray, float, float]:
    """
    Create a signal mask using multilevel Otsu thresholding (k=4) via ANTsPy.
    Final signal mask is all non-zero labels.

    img_noisy: input noisy image (3D array)
    nclasses: number of classes for multilevel Otsu thresholding

    Returns: signal_mask: signal mask (boolean 3D array)
    Returns: thresh: threshold value used for mask generation
    """

    # Sample non-zero voxels for Otsu thresholding
    t1_sample = img_noisy[img_noisy > 0].flatten()

    # Multilevel Otsu thresholds
    otsu_thresh = skf.threshold_multiotsu(t1_sample, classes=nclasses)
    mask_thresh = otsu_thresh[0]

    # Create signal mask
    signal_mask = img_noisy >= mask_thresh

    # Compute percent coverage
    percent_coverage = 100.0 * np.sum(signal_mask) / signal_mask.size

    return signal_mask, mask_thresh, percent_coverage

def airspace_noise_est(img_noisy: np.ndarray):
    """
    Magnitude of complex-valued Gaussian noise follows a Rayleigh distribution in signal-free regions.
    Median of Rayleigh distribution is related to sigma_n by: sigma_n = median / sqrt(2 * log(2))

    img_noisy: input noisy image (3D array)

    Returns: estimated noise sigma (scalar)
    """

    signal_mask, mask_thresh, percent_coverage = signal_mask_otsu(img_noisy)
    print(f"Airspace mask threshold {mask_thresh:0.2f}, coverage {percent_coverage:0.2f} %")

    noise_sample = img_noisy[~signal_mask].flatten()
    median_noise = np.median(noise_sample)
    sigma_n = median_noise / np.sqrt(2 * np.log(2))

    return sigma_n
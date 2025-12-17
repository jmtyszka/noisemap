"""
Aja-Fernandez homomorphic noise estimation
"""

import numpy as np
from scipy.special import iv

from .utils import (lpf, conv3d)

def approxI1_I0(z):
    """
    Approximate the ratio I1(z)/I0(z) for the modified Bessel functions.
    Vectorized for numpy arrays.
    """
    z = np.asarray(z)
    M = np.zeros_like(z, dtype=np.float64)
    z8 = 8.0 * z
    Mn = 1 - 3.0 / z8 - 15.0 / 2.0 / (z8 ** 2 + 1e-12) - (3 * 5 * 21) / 6.0 / (z8 ** 3 + 1e-12)
    Md = 1 + 1.0 / z8 + 9.0 / 2.0 / (z8 ** 2 + 1e-12) + (25 * 9) / 6.0 / (z8 ** 3 + 1e-12)
    M = Mn / (Md + 1e-12)
    
    # For z < 1.5, use the true Bessel ratio
    cont = z < 1.5
    if np.any(cont):
        M[cont] = iv(1, z[cont]) / iv(0, z[cont])
    
    # For z == 0, set to 0
    M[z == 0] = 0.0
    
    return M

def em_ml_rice3D(img, em_niter=5, em_ksize=3):
    """
    EM implementation of Maximum Likelihood for Rician data (3D).

    img: Input data (Rician image)
    niter: Number of EM iterations
    ksize: kernel size for local averaging (integer)
    returns: img_denoised, img_sigma_n
    """

    # Kernel array weights
    h = np.ones((em_ksize, em_ksize, em_ksize)) / em_ksize**3

    # Initialize a_k and sigma_k2
    a_k = np.sqrt(np.sqrt(np.maximum(2 * conv3d(h, img**2)**2 - conv3d(h, img**4), 0)))
    sigma_k2 = 0.5 * np.maximum(conv3d(h, img**2) - a_k**2, 0.01)
    
    for _ in range(niter):
        a_k = np.maximum(conv3d(h, approxI1_I0(a_k * img / sigma_k2) * img), 0)
        sigma_k2 = np.maximum(0.5 * conv3d(h, np.abs(img)**2) - a_k**2 / 2, 0.01)

    img_denoised = a_k
    img_sigma_n = np.sqrt(sigma_k2)

    return img_denoised, img_sigma_n

def correct_rice_gauss(self, snr):
    """
    Rician-Gaussian correction for noise estimation

    snr: signal SNR (scalar or array)
    Returns: correction curve Fc
    """
    
    snr = np.asarray(snr, dtype=np.float64)

    # Coefficients for Rician-Gaussian correction polynomial
    snr_coeffs = [
        -0.289549906258443,
        -0.0388922575606330,
        0.409867108141953,
        -0.355237628488567,
        0.149328280945610,
        -0.0357861117942093,
        0.00497952893859122,
        -0.000374756374477592,
        1.18020229140092e-05
    ]

    Fc = (
        snr_coeffs[0] +
        snr_coeffs[1] * snr +
        snr_coeffs[2] * snr**2 +
        snr_coeffs[3] * snr**3 +
        snr_coeffs[4] * snr**4 +
        snr_coeffs[5] * snr**5 +
        snr_coeffs[6] * snr**6 +
        snr_coeffs[7] * snr**7 +
        snr_coeffs[8] * snr**8
    )
    
    Fc = Fc * (snr <= 7)

    return Fc

def rice_homomorphic_est(img_noisy: np.ndarray):
    """
    Noise estimation in SENSE MR using a homomorphic approach.

    PARAMETER: img_noisy: 3D noisy magnitude data
    """

    # Low pass filter Gaussian sigma
    sigma_lpf = 5.0

    # EM parameters
    em_niter = 5
    em_ksize = 3

    # Estimate SNR from data using EM
    img_denoised, img_sigma_n = em_ml_rice3D(img_noisy, em_niter=em_niter, em_ksize=em_ksize)
    img_snr = img_denoised / (img_sigma_n + 1e-12)

    # Apply low-pass filter to sigma_n and SNR maps
    img_noise = img_noisy - img_denoised
    img_sigma_n_lpf = lpf(img_sigma_n, sigma_lpf=sigma_lpf)
    img_snr_lpf = lpf(img_snr, sigma_lpf=sigma_lpf)

    return img_denoised, img_noise, img_sigma_n_lpf, img_snr_lpf
"""
Two noise mapping algorithms for 3D magnitude MR images.

1. Homomorphic noise estimation for SENSE MR images in Python.
Based on the method of Aja-Fernandez et al., "Noise estimation in SENSE MR images: The homomorphic approach", IEEE TMI, 2009.

2. Adaptive Non-Local Means (ANLM) noise estimation for MR images.
Based on the method of Manjon et al., "Adaptive non-local means denoising of MR images with spatially varying noise levels", JMRI, 2010.
"""

import os
import os.path as op
from networkx import sigma
import numpy as np
from scipy.special import iv
from scipy.ndimage import (gaussian_filter, median_filter, convolve)
import nibabel as nib
import ants

class NoiseMap:

    def __init__(self, nifti_path):
        """
        Initialize with a Nifti image path. Loads the image data as a numpy array.
        """
        print(f'Loading NIfTI image: {nifti_path}')
        self.nifti_path = nifti_path
        self.img_nii = nib.load(nifti_path)
        self.img = self.img_nii.get_fdata()

        # Init noise and SNR maps
        self.signal_mask = None
        self.img_denoised = None
        self.img_noise = None
        self.img_sigma_n_lpf = None
        self.img_snr_lpf = None
        
        # Default estimation method
        self.estimation_method = 'homomorphic'

        # --------
        # Aja-Fernandez homomorphic noise estimation parameters
        # --------

        # Low pass filter Gaussian sigma
        self.sigma_lpf = 5.0

        # Coefficients for Rician-Gaussian correction polynomial
        self.snr_coeffs = [
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

        # EM parameters
        self.em_niter = 5
        self.em_ksize = 3

    #
    # Homomorphic methods
    #

    @staticmethod
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
    
    @staticmethod
    def conv3d(h, img):
        """
        3D convolution with nearest edge padding

        parameter: img: input 3D image
        parameter: h: 3D convolution kernel
        returns: 3D convolved image
        """
        return convolve(img, h, mode='nearest')

    def em_ml_rice3D(self, img):
        """
        EM implementation of Maximum Likelihood for Rician data (3D).

        img: Input data (Rician image)
        niter: Number of EM iterations
        ksize: kernel size for local averaging (integer)
        returns: img_denoised, img_sigma_n
        """

        niter = self.em_niter
        ksize = self.em_ksize
        h = np.ones((ksize, ksize, ksize)) / ksize**3

        # Initialize a_k and sigma_k2
        a_k = np.sqrt(np.sqrt(np.maximum(2 * self.conv3d(h, img**2)**2 - self.conv3d(h, img**4), 0)))
        sigma_k2 = 0.5 * np.maximum(self.conv3d(h, img**2) - a_k**2, 0.01)
        
        for _ in range(niter):
            a_k = np.maximum(self.conv3d(h, self.approxI1_I0(a_k * img / sigma_k2) * img), 0)
            sigma_k2 = np.maximum(0.5 * self.conv3d(h, np.abs(img)**2) - a_k**2 / 2, 0.01)

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

        c = self.snr_coeffs

        Fc = (
            c[0] +
            c[1] * snr +
            c[2] * snr**2 +
            c[3] * snr**3 +
            c[4] * snr**4 +
            c[5] * snr**5 +
            c[6] * snr**6 +
            c[7] * snr**7 +
            c[8] * snr**8
        )
        
        Fc = Fc * (snr <= 7)

        return Fc

    def rice_homomorf_est(self, img_noisy: np.ndarray):
        """
        Noise estimation in SENSE MR using a homomorphic approach.

        PARAMETER: img_noisy: 3D noisy magnitude data
        """
        img_noisy = self.img

        # Estimate SNR from data using EM
        img_denoised, img_sigma_n = self.em_ml_rice3D(img_noisy)
        img_snr = img_denoised / (img_sigma_n + 1e-12)

        # Apply low-pass filter to sigma_n and SNR maps
        self.img_denoised = img_denoised
        self.img_noise = img_noisy - img_denoised
        self.img_sigma_n_lpf = self.lpf(img_sigma_n, sigma_lpf=self.sigma_lpf)
        self.img_snr_lpf = self.lpf(img_snr, sigma_lpf=self.sigma_lpf)

    #
    # ANLM methods
    #

    def anlm_denoise(self, img_noisy: np.ndarray):

        img_noisy = self.img

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

        self.signal_mask = signal_mask
        self.img_denoised = img_denoised
        self.img_noise = img_noise
        self.img_sigma_n_lpf = self.lpf(sigma_n, sigma_lpf=self.sigma_lpf)
        self.img_snr_lpf = self.lpf(snr, sigma_lpf=self.sigma_lpf)

    #
    # Common methods
    #
        
    def estimate(self, method:str='homomorphic'):
        """
        Run noise estimation on the loaded Nifti image.
        SNR: signal-to-noise ratio (optional, if None, estimated from data)
        method: estimation method (optional, default 'homomorphic')
        """

        # Save estimation method for output file naming
        self.estimation_method = method

        # 3D scalar images only for now
        assert self.img.ndim == 3, "Input image must be 3D scalar"
        img3d = self.img

        match method.lower():
            case 'homomorphic':
                # Run homomorphic Rician noise estimation
                self.rice_homomorf_est(img3d)
            case 'anlm':
                # Run ANLM denoising and noise estimation
                self.anlm_denoise(img3d)
            case _:
                raise ValueError(f"Unknown estimation method: {method}")
            
    
    def lpf(self, img: np.ndarray, sigma_lpf: float=5.0) -> np.ndarray:
        """
        Apply low-pass Gaussian filter to a numpy image
        Set Gaussian sigma with self.sigma_lpf

        img: input image (3D array)

        Returns: filtered image
        """
        return gaussian_filter(img, sigma=sigma_lpf)

    def save_maps(self, out_dir=None):
        """
        Save the estimated noise and SNR maps as NIfTI files.
        out_dir: output directory (if None, saves in the same directory as input)
        """

        if out_dir is None:
            # Save in same directory as input with method-specific subdirectory
            out_dir = op.join(op.dirname(self.nifti_path), f"noisemap_{self.estimation_method}")
        
        # Safe create output directory if it doesn't exist
        os.makedirs(out_dir, exist_ok=True)

        print(f"Saving noise maps to {out_dir}")

        # Save denoised image
        denoised_nii = nib.Nifti1Image(self.img_denoised, affine=self.img_nii.affine, header=self.img_nii.header)
        denoised_path = op.join(out_dir, self.nifti_path.replace(".nii.gz", "_denoised.nii.gz"))
        nib.save(denoised_nii, denoised_path)
        print(f"  Saved denoised image to {op.basename(denoised_path)}")

        # Save noise image
        noise_nii = nib.Nifti1Image(self.img_noise, affine=self.img_nii.affine, header=self.img_nii.header)
        noise_path = op.join(out_dir, self.nifti_path.replace(".nii.gz", "_noise.nii.gz"))
        nib.save(noise_nii, noise_path)
        print(f"  Saved noise image to {op.basename(noise_path)}")

        # Save Sigma_n map
        sigma_n_nii = nib.Nifti1Image(self.img_sigma_n_lpf, affine=self.img_nii.affine, header=self.img_nii.header)
        sigma_n_path = op.join(out_dir, self.nifti_path.replace(".nii.gz", "_sigma.nii.gz"))
        nib.save(sigma_n_nii, sigma_n_path)
        print(f"  Saved noise sigma map to {op.basename(sigma_n_path)}")

        # Save SNR map
        snr_nii = nib.Nifti1Image(self.img_snr_lpf, affine=self.img_nii.affine, header=self.img_nii.header)
        snr_path = op.join(out_dir, self.nifti_path.replace(".nii.gz", "_snr.nii.gz"))
        nib.save(snr_nii, snr_path)
        print(f"  Saved SNR map to {op.basename(snr_path)}")

        if self.signal_mask is not None:
            # Save signal mask
            signal_mask_nii = nib.Nifti1Image(self.signal_mask.astype(np.uint8), affine=self.img_nii.affine, header=self.img_nii.header)
            signal_mask_path = op.join(out_dir, self.nifti_path.replace(".nii.gz", "_mask.nii.gz"))
            nib.save(signal_mask_nii, signal_mask_path)
            print(f"  Saved signal mask to {op.basename(signal_mask_path)}")
    
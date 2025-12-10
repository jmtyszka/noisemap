import os
import os.path as op
import numpy as np
from scipy.special import iv
from scipy.ndimage import (gaussian_filter, convolve)
import nibabel as nib

class NoiseMap:

    def __init__(self, nifti_path):
        """
        Initialize with a Nifti image path. Loads the image data as a numpy array.
        """
        self.nifti_path = nifti_path
        self.img_nii = nib.load(nifti_path)
        self.img = self.img_nii.get_fdata()

        # Init noise and SNR maps
        self.img_sigma_n_lpf = None
        self.img_snr_lpf = None
        
        # Default estimation method
        self.estimation_method = 'homomorphic'

        # --------
        # Aja-Fernandez homomorphic noise estimation parameters
        # --------

        # Low pass filter Gaussian sigma
        self.sigma_lpf = 4.8

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

    @staticmethod
    def approxI1_I0(z):
        """
        Approximate the ratio I1(z)/I0(z) for the modified Bessel functions.
        Vectorized for numpy arrays.
        """
        z = np.asarray(z)
        M = np.zeros_like(z, dtype=np.float64)
        z8 = 8.0 * z
        Mn = 1 - 3.0 / z8 - 15.0 / 2.0 / (z8 ** 2) - (3 * 5 * 21) / 6.0 / (z8 ** 3)
        Md = 1 + 1.0 / z8 + 9.0 / 2.0 / (z8 ** 2) + (25 * 9) / 6.0 / (z8 ** 3)
        M = Mn / Md
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
        Rician-Gaussian correction for noise estimation.
        snr: signal SNR (scalar or array)
        coef_mat_path: path to .mat file with 'Coefs' variable
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
    
    def lpf(self, img: np.ndarray) -> np.ndarray:

        """
        Apply low-pass Gaussian filter to a numpy image
        img: input image (3D array)
        Returns: filtered image
        """
        return gaussian_filter(img, sigma=self.sigma_lpf)

    def rice_homomorf_est(self, img_noisy: np.ndarray, snr=None):
        """
        Noise estimation in SENSE MR using a homomorphic approach.

        PARAMETER: img_noisy: noisy data (3D array)
        PARAMETER: snr: signal-to-noise ratio (optional, if None, estimated from data)
        """
        img_noisy = np.asarray(img_noisy, dtype=np.float64)

        if snr is None or (isinstance(snr, (int, float)) and snr == 0):
            # Estimate SNR from data using EM
            img_denoised, img_sigma_n = self.em_ml_rice3D(img_noisy)
            img_snr = img_denoised / (img_sigma_n + 1e-12)
        else:
            # If SNR is provided, estimate Sigma_n from SNR and In
            img_snr = snr * np.ones_like(img_noisy)
            img_sigma_n = img_noisy / (snr + 1e-12)

        # Apply low-pass filter to sigma_n and SNR maps
        self.img_sigma_n_lpf = self.lpf(img_sigma_n)
        self.img_snr_lpf = self.lpf(img_snr)

    def estimate(self, snr=None, method=None):
        """
        Run noise estimation on the loaded Nifti image.
        SNR: signal-to-noise ratio (optional, if None, estimated from data)
        method: estimation method (optional, default 'homomorphic')
        """

        # Save estimation method for output file naming
        if method is not None:
            self.estimation_method = method

        # 3D scalar images only for now
        assert self.img.ndim == 3, "Input image must be 3D scalar"
        img3d = self.img

        match method:
            case 'homomorphic':
                # Run homomorphic Rician noise estimation
                self.rice_homomorf_est(img3d, snr=snr)
            case 'ANLM':
                raise NotImplementedError("ANLM method not implemented yet")
            case _:
                raise ValueError(f"Unknown estimation method: {method}")
            
    
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

        print(f"Saving noise and SNR maps to {out_dir}")

        # Save Sigma_n map
        sigma_n_nii = nib.Nifti1Image(self.img_sigma_n_lpf, affine=self.img_nii.affine, header=self.img_nii.header)
        sigma_n_path = op.join(out_dir, self.nifti_path.replace(".nii.gz", "_sigma.nii.gz"))
        nib.save(sigma_n_nii, sigma_n_path)
        print(f"Saved noise sigma map to {sigma_n_path}")

        # Save SNR map
        snr_nii = nib.Nifti1Image(self.img_snr_lpf, affine=self.img_nii.affine, header=self.img_nii.header)
        snr_path = op.join(out_dir, self.nifti_path.replace(".nii.gz", "_snr.nii.gz"))
        nib.save(snr_nii, snr_path)
        print(f"Saved SNR map to {snr_path}")
    
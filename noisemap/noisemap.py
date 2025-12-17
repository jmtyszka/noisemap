"""
Three denoising/noise estimation algorithms are implemented for 3D magnitude MR images.
For denoising algorithms, the residual difference between noisy and denoised images is used to
estimate local noise sigma maps and SNR maps using the local MAD approximation for Gaussian white noise.
Rician noise modeling and corrections are used where possible.

1. Homomorphic noise estimation for SENSE MR images in Python.
Based on the method of Aja-Fernandez et al., "Noise estimation in SENSE MR images: The homomorphic approach", IEEE TMI, 2009.

2. Adaptive Non-Local Means (ANLM) denoising for MR images.
Based on the method of Manjon et al., "Adaptive non-local means denoising of MR images with spatially varying noise levels", JMRI, 2010.

3. Adaptive Soft Matching (ASM) denoising for MR images.
Based on the method of Pierrick Coupé, José V. Manjón, Montserrat Robles, and Louis D. Collins. Adaptive Multiresolution Non-Local Means
Filter for 3D MR Image Denoising. IET Image Processing, 6(5):558–568, July 2012. implemented in the DiPy package.
"""

import os
import os.path as op
from networkx import sigma
import numpy as np
from scipy.special import iv
import nibabel as nib
import ants

from homomorphic import rice_homomorphic_est
from anlm import anlm_est
from asm import asm_est

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
        img_noisy = self.img

        match method.lower():
            case 'homomorphic':
                # Run homomorphic Rician noise estimation
                img_denoised, img_noise, img_sigma, img_snr, signal_mask = rice_homomorphic_est(img_noisy)
                self.img_denoised = img_denoised
                self.img_noise = img_noise
                self.img_sigma_n_lpf = img_sigma
                self.img_snr_lpf = img_snr
                self.signal_mask = signal_mask
            case 'anlm':
                # Run ANLM denoising and noise estimation
                img_denoised, img_noise, img_sigma, img_snr, signal_mask = anlm_est(img_noisy)
                self.img_denoised = img_denoised
                self.img_noise = img_noise
                self.img_sigma_n_lpf = img_sigma
                self.img_snr_lpf = img_snr
                self.signal_mask = signal_mask
            case 'asm':
                # Run ASM denoising and noise estimation
                img_denoised, img_noise, img_sigma, sigma_mad = asm_est(img_noisy)
                self.img_denoised = img_denoised
                self.img_noise = img_noise
                self.img_sigma_n_lpf = img_sigma
                self.signal_mask = img_noisy > np.max(img_noisy) * 0.05
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
    
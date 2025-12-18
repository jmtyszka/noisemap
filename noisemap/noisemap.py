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

from .homomorphic import rice_homomorphic_est
from .anlm import anlm_est
from .asm import asm_est

class NoiseMap:

    def __init__(self, nifti_path, method='homomorphic', out_dir=None):
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
        self.img_sigmamap = None
        self.img_snrmap = None
        
        # Default estimation method
        if method in ['homomorphic', 'anlm', 'asm']:
            self.estimation_method = method
        else:
            self.estimation_method = 'homomorphic'

        if out_dir is None:
            # Save in same directory as input with method-specific subdirectory
            self.out_dir = op.join(op.dirname(self.nifti_path), f"noisemap_{self.estimation_method}")
        else:
            self.out_dir = out_dir
        
        # Safe create output directory if it doesn't exist
        print(f"Output directory set to {self.out_dir}")
        os.makedirs(self.out_dir, exist_ok=True)
      
    def estimate(self):
        """
        Run noise estimation on the loaded Nifti image.
        SNR: signal-to-noise ratio (optional, if None, estimated from data)
        method: estimation method (optional, default 'homomorphic')
        """

        # 3D scalar magnitude images only for now
        assert (self.img.ndim == 3) & (self.img >= 0).all(), "Input image must be 3D scalar magnitude data"
        img_noisy = self.img

        match self.estimation_method.lower():
            case 'homomorphic':
                # Run homomorphic Rician noise estimation
                img_denoised, img_noise, img_sigmamap, img_snrmap, signal_mask = rice_homomorphic_est(img_noisy)
                self.img_denoised = img_denoised
                self.img_noise = img_noise
                self.img_sigmamap = img_sigmamap
                self.img_snrmap = img_snrmap
                self.signal_mask = signal_mask
            case 'anlm':
                # Run ANLM denoising and noise estimation
                img_denoised, img_noise, img_sigmamap, img_snrmap, signal_mask = anlm_est(img_noisy)
                self.img_denoised = img_denoised
                self.img_noise = img_noise
                self.img_sigmamap = img_sigmamap
                self.img_snrmap = img_snrmap
                self.signal_mask = signal_mask
            case 'asm':
                # Run ASM denoising and noise estimation
                img_denoised, img_noise, img_sigmamap, img_snrmap, signal_mask = asm_est(img_noisy)
                self.img_denoised = img_denoised
                self.img_noise = img_noise
                self.img_sigmamap = img_sigmamap
                self.img_snrmap = img_snrmap
                self.signal_mask = signal_mask
            case _:
                raise ValueError(f"Unknown estimation method: {self.estimation_method}")

    def save_maps(self):
        """
        Save the estimated noise and SNR maps as NIfTI files to the output directory.
        The following files are saved:
        - Denoised image: *_denoised.nii.gz
        - Noise image: *_noise.nii.gz
        - Noise sigma map: *_sigma.nii.gz
        - SNR map: *_snr.nii.gz
        - Signal mask (if available): *_mask.nii.gz
        """

        # Save denoised image
        denoised_nii = nib.Nifti1Image(self.img_denoised, affine=self.img_nii.affine, header=self.img_nii.header)
        denoised_path = op.join(self.out_dir, self.nifti_path.replace(".nii.gz", "_denoised.nii.gz"))
        nib.save(denoised_nii, denoised_path)
        print(f"Saved denoised image to {op.basename(denoised_path)}")

        # Save noise image
        noise_nii = nib.Nifti1Image(self.img_noise, affine=self.img_nii.affine, header=self.img_nii.header)
        noise_path = op.join(self.out_dir, self.nifti_path.replace(".nii.gz", "_noise.nii.gz"))
        nib.save(noise_nii, noise_path)
        print(f"Saved noise image to {op.basename(noise_path)}")

        # Save Sigma_n map
        sigma_n_nii = nib.Nifti1Image(self.img_sigmamap, affine=self.img_nii.affine, header=self.img_nii.header)
        sigma_n_path = op.join(self.out_dir, self.nifti_path.replace(".nii.gz", "_sigma.nii.gz"))
        nib.save(sigma_n_nii, sigma_n_path)
        print(f"Saved noise sigma map to {op.basename(sigma_n_path)}")

        # Save SNR map
        snr_nii = nib.Nifti1Image(self.img_snrmap, affine=self.img_nii.affine, header=self.img_nii.header)
        snr_path = op.join(self.out_dir, self.nifti_path.replace(".nii.gz", "_snr.nii.gz"))
        nib.save(snr_nii, snr_path)
        print(f"Saved SNR map to {op.basename(snr_path)}")

        if self.signal_mask is not None:
            # Save signal mask
            signal_mask_nii = nib.Nifti1Image(self.signal_mask.astype(np.uint8), affine=self.img_nii.affine, header=self.img_nii.header)
            signal_mask_path = op.join(self.out_dir, self.nifti_path.replace(".nii.gz", "_mask.nii.gz"))
            nib.save(signal_mask_nii, signal_mask_path)
            print(f"Saved signal mask to {op.basename(signal_mask_path)}")
    
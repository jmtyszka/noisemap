# noisemap
Spatially variant noise estimation for MRI

## Installation
1. Clone git repository locally (`noisemap/`)
2. Install with pip
```
$ cd noisemap/
$ pip install .
```

## Usage
```
$ noisemap -i <mag_image>.nii.gz [-m <algorithm>]
# noisemap -i T1w.nii.gz
```
Currently the default and only supported value for <algorithm> is "anlm" (Adaptive Non-Local Means). An output folder will be created named <mag_image>_anlm/
containing the following (ANLM output shown):

```
noisemap_anlm/
├── T1w_denoised.nii.gz
├── T1w_mask.nii.gz
├── T1w_noise.nii.gz
├── T1w_sigma.nii.gz
└── T1w_snr.nii.gz

_denoised : ANLM denoised version of original image  
_mask : signal/computation mask used during denoising  
_noise : residual noise image (original - denoised)  
_sigma : Local Gaussian noise sigma map estimated from _noise  
_snr : Voxel-wise SNR estimated from _denoised and _sigma  

```

## Algorithms

### Adaptive Non-local Means (ANLM)

Python wrapper for ANTs ImageDenoise function implemented in antspyx. Estimates noise residual from the original and denoised images within regions with signal support (Otsu threshold).

Based on:
J. V. Manjon, P. Coupe, Luis Marti-Bonmati, D. L. Collins, and M. Robles.
Adaptive Non-Local Means Denoising of MR Images With Spatially Varying Noise Levels
Journal of Magnetic Resonance Imaging, 31:192-203, June 2010.
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


## Algorithms

### Aja-Fernández Homomorphic

Extension of original 2D Matlab scripts (Matlab Central) to python with 3D support.

#### References

Aja-Fernández, S., Pieciak, T. & Vegas-Sánchez-Ferrero, G. Spatially variant noise estimation in MRI: a homomorphic approach. Med. Image Anal. 20, 184–197 (2015).

https://www.mathworks.com/matlabcentral/fileexchange/48762-noise-estimator-for-sense-mri


### Adaptive Non-local Means

Python wrapper for ANTs ImageDenoise function implemented in antspyx. Estimates noise residual from the original and denoised images within regions with signal support (Otsu threshold)
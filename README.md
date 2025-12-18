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
$ noisemap -i <mag_image>.nii.gz -m <algorithm>
# noisemap -i T1w.nii.gz -m anlm
```
<algorithm> can be homomorphic, anlm or asm. An output folder will be created named <mag_image>_<algorithm>/
containing the following (anlm output shown):
```
```


## Algorithms

### Global noise sigma estimate
Spatial average noise sigma estimates assuming a Rayleigh (magnitude of N(0, sigma)) distribution
Makes use of property of Rayleigh distribution: mode(|N(0, sigma)) = sigma
Alternative to use of median(|N(0, sigma)) = sigma * 2 sqrt(2)


### Aja-Fernández Homomorphic

Extension of original 2D Matlab scripts (Matlab Central) to python with 3D support.

#### References

Aja-Fernández, S., Pieciak, T. & Vegas-Sánchez-Ferrero, G. Spatially variant noise estimation in MRI: a homomorphic approach. Med. Image Anal. 20, 184–197 (2015).

https://www.mathworks.com/matlabcentral/fileexchange/48762-noise-estimator-for-sense-mri


### Adaptive Non-local Means (ANLM)

Python wrapper for ANTs ImageDenoise function implemented in antspyx. Estimates noise residual from the original and denoised images within regions with signal support (Otsu threshold)

### Adaptive soft
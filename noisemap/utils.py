import numpy as np
import ants
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

def fourier_lpf(img: np.ndarray, sigma_freq: float=5.0) -> np.ndarray:
    """
    Frequency domain low-pass Gaussian filter of a 3D numpy array
    - reimplement lpf MATLAB function from homomorphic_sense_tbx with MODO=1

    img: input image (3D array)
    sigma_freq: frequency domain Gaussian sigma in voxels

    Returns: filtered image

    ORIGINAL MATLAB CODE:

    if MODO==1
        [Mx,My]=size(I);
        h=fspecial('gaussian',size(I),sigma);
        h=h./max(h(:));

        if (Mx==1)||(My==1) %1D
            lRnF=fftshift(fft(I));
            %Filtering
            lRnF2=lRnF.*h;
            If=real(ifft(fftshift(lRnF2)));

        else %2D
            lRnF=fftshift(fft2(I));
            %Filtering
            lRnF2=lRnF.*h;
            If=real(ifft2(fftshift(lRnF2)));
        end

        return np.real(img_filtered)
    """

    # Create Gaussian kernel in frequency domain
    size = img.shape
    h = fspecial_gaussian_3d(size, sigma_freq)
    h = h / np.max(h)

    # FFT of input image
    img_fft = np.fft.fftn(img)
    img_fft_shifted = np.fft.fftshift(img_fft)

    # Filtering in frequency domain
    img_fft_filtered = img_fft_shifted * h

    # Inverse FFT to get filtered image
    img_ifft_shifted = np.fft.ifftshift(img_fft_filtered)
    img_filtered = np.fft.ifftn(img_ifft_shifted)

    return np.real(img_filtered)

def fspecial_gaussian_3d(size, sigma):
    """
    Create a 3D Gaussian kernel similar to MATLAB's fspecial('gaussian', ...)

    size: tuple of 3 ints, size of the kernel
    sigma: standard deviation of the Gaussian

    Returns: 3D numpy array representing the Gaussian kernel
    """
    m, n, o = [(ss - 1) / 2 for ss in size]
    y, x, z = np.ogrid[-m:m+1, -n:n+1, -o:o+1]
    h = np.exp(-(x*x + y*y + z*z) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def conv3d(h, img):
    """
    3D convolution with nearest edge padding

    parameter: img: input 3D image
    parameter: h: 3D convolution kernel
    returns: 3D convolved image
    """
    return convolve(img, h, mode='nearest')

def noise_sigma_map(noise_img: np.ndarray, signal_mask: np.ndarray) -> np.ndarray:
    """
    NOTE : This function takes as input the residual noise image (img_noisy - img_denoised)
    Create a local noise sigma map using MAD robust estimator on the residual image
    Approximate noise residual distribution as N(0, sigma_n)
    """

    noise_img = noise_img * signal_mask
    sigma_img = median_filter(np.abs(noise_img), size=5) / np.sqrt(2 * np.log(2))

    return sigma_img

def signal_mask_otsu(img_noisy: np.ndarray, nclasses: int=4):
    """
    Create a signal mask using multilevel Otsu thresholding (k=4) via ANTsPy.
    Final signal mask is all non-zero labels.

    img_noisy: input noisy image (3D array)
    nclasses: number of classes for multilevel Otsu thresholding

    Returns: signal_mask: signal mask (boolean 3D array)
    Returns: thresh: threshold value used for masking
    """

    t1_sample = img_noisy[img_noisy > 0].flatten()

    # Multilevel Otsu thresholds
    otsu_thresh = skf.threshold_multiotsu(t1_sample, classes=nclasses)
    thresh = otsu_thresh[0]

    signal_mask = img_noisy >= thresh

    return signal_mask, thresh

def airspace_noise_est(img_noisy: np.ndarray):
    """
    Magnitude of complex-valued Gaussian noise follows a Rayleigh distribution in signal-free regions.
    Median of Rayleigh distribution is related to sigma_n by: median = sigma_n * sqrt(2 * log(2))

    img_noisy: input noisy image (3D array)

    Returns: estimated noise sigma (scalar)
    """

    signal_mask, _ = signal_mask_otsu(img_noisy)

    noise_sample = img_noisy[~signal_mask].flatten()
    median_noise = np.median(noise_sample)
    sigma_n = median_noise / np.sqrt(2 * np.log(2))

    return sigma_n
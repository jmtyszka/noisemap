import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, ifft2, fftshift, ifftshift, dctn, idctn

def lpf(I, sigma, MODO=2):
    """
    Low pass filter for images. Supports DFT and DCT modes.
    I: Input image (2D array)
    sigma: standard deviation of Gaussian window
    MODO: 1 for DFT filtering, 2 for DCT filtering (default)
    """
    I = np.asarray(I)
    if MODO == 1:
        # DFT filtering
        Mx, My = I.shape
        # Create Gaussian filter in spatial domain, then FFT to freq domain
        x = np.arange(-Mx//2, Mx//2)
        y = np.arange(-My//2, My//2)
        X, Y = np.meshgrid(x, y, indexing='ij')
        h = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        h = h / np.max(h)
        lRnF = fftshift(fft2(I))
        lRnF2 = lRnF * h
        If = np.real(ifft2(ifftshift(lRnF2)))
        return If
    elif MODO == 2:
        # DCT filtering
        Mx, My = I.shape
        # Double size for filter as in MATLAB code
        h = np.exp(-((np.arange(2*Mx) - Mx)**2 + (np.arange(2*My) - My)**2) / (2 * (sigma*2)**2))
        h = h / np.max(h)
        h = h[:Mx, :My]
        I_dct = dctn(I, norm='ortho')
        I_dct_filt = I_dct * h
        If = idctn(I_dct_filt, norm='ortho')
        return If
    else:
        raise ValueError('MODO must be 1 (DFT) or 2 (DCT)')

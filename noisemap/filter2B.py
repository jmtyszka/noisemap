import numpy as np
from scipy.signal import convolve2d

def filter2B(h, I):
    """
    2D convolution with border correction. h must be NxN, with N odd.
    Equivalent to MATLAB's filter2B.
    """
    h = np.asarray(h)
    I = np.asarray(I)
    Mx, My = h.shape
    if (Mx % 2 == 0) or (My % 2 == 0):
        raise ValueError('h size must be odd')
    Nx = (Mx - 1) // 2
    Ny = (My - 1) // 2
    # Pad array with edge values (replicate)
    It = np.pad(I, ((Nx, Nx), (Ny, Ny)), mode='edge')
    # 2D convolution, 'valid' to match MATLAB's output size
    I2 = convolve2d(It, h, mode='valid')
    return I2

import numpy as np
from scipy.io import loadmat
from .rice_homomorf_est import rice_homomorf_est

def example():
    """
    Example usage of the homomorphic SENSE toolbox in Python.
    Loads example data and runs noise estimation.
    """
    # Load Mapa_grappa.mat
    mapa_data = loadmat('Mapa_grappa.mat')
    Mapa = mapa_data['Mapa']
    # Load mri.mat
    mri_data = loadmat('mri.mat')
    I = mri_data['I']
    # Simulate noisy input
    np.random.seed(0)
    randn_real = np.random.randn(256, 256)
    randn_imag = np.random.randn(256, 256)
    In = np.abs(I + Mapa * randn_real + Mapa * 1j * randn_imag)
    SNR = I / Mapa
    # Estimation with known SNR
    MapaR, MapaG = rice_homomorf_est(In, SNR, 3.4, 2)
    # Estimation with unknown SNR
    MapaR2, MapaG2 = rice_homomorf_est(In, 0, 3.4, 2)
    # Visualization (optional, requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(np.hstack([Mapa, MapaR, MapaR2, MapaG2]), cmap='viridis')
        plt.colorbar()
        plt.title('Noise Maps')
        plt.show()
    except ImportError:
        print('matplotlib not installed, skipping visualization')
    return Mapa, MapaR, MapaR2, MapaG2

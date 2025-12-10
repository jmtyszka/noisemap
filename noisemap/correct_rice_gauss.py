import numpy as np
from scipy.io import loadmat

def correct_rice_gauss(SNR, coef_mat_path='coef_SNR_8.mat'):
    """
    Rician-Gaussian correction for noise estimation.
    SNR: signal SNR (scalar or array)
    coef_mat_path: path to .mat file with 'Coefs' variable
    Returns: correction curve Fc
    """
    SNR = np.asarray(SNR)
    mat = loadmat(coef_mat_path)
    Coefs = mat['Coefs'].flatten()
    Fc = (Coefs[0] + Coefs[1]*SNR + Coefs[2]*SNR**2 + Coefs[3]*SNR**3 +
          Coefs[4]*SNR**4 + Coefs[5]*SNR**5 + Coefs[6]*SNR**6 +
          Coefs[7]*SNR**7 + Coefs[8]*SNR**8)
    Fc = Fc * (SNR <= 7)
    return Fc

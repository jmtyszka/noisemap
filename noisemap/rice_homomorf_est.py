import numpy as np
from .em_ml_rice2D import em_ml_rice2D
from .lpf import lpf

def rice_homomorf_est(In, SNR=None, LPF_val=4.8, Modo=2):
    """
    Noise estimation in SENSE MR using a homomorphic approach.
    In: noisy Rician data (2D array)
    SNR: signal-to-noise ratio (optional, if None, estimated from data)
    LPF_val: threshold for LPF (default 4.8)
    Modo: calculation mode for local mean (1: local mean, 2: ML estimation, 3: no subtraction)
    Returns: MapaR, MapaG
    """
    In = np.asarray(In, dtype=np.float64)
    if SNR is None or (isinstance(SNR, (int, float)) and SNR == 0):
        # Estimate SNR from data using EM
        Signal, Sigma_n = em_ml_rice2D(In, N=5, Ws=(3,3))
        SNR = Signal / (Sigma_n + 1e-12)
    # Apply low-pass filter
    MapaR = lpf(SNR, LPF_val, Modo)
    MapaG = lpf(Sigma_n, LPF_val, Modo)
    return MapaR, MapaG

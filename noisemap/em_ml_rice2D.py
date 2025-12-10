import numpy as np
from scipy.special import iv
from .filter2B import filter2B
from .approxI1_I0 import approxI1_I0

def em_ml_rice2D(In, N, Ws):
    """
    EM implementation of Maximum Likelihood for Rician data (2D).
    In: Input data (Rician image)
    N: Number of EM iterations
    Ws: Window size (tuple, e.g., (3,3))
    Returns: Signal, Sigma_n
    """
    In = np.asarray(In, dtype=np.float64)
    Mask = np.ones(Ws) / np.prod(Ws)
    a_k = np.sqrt(np.sqrt(np.maximum(2 * filter2B(Mask, In**2)**2 - filter2B(Mask, In**4), 0)))
    sigma_k2 = 0.5 * np.maximum(filter2B(Mask, In**2) - a_k**2, 0.01)
    for _ in range(N):
        a_k = np.maximum(filter2B(Mask, approxI1_I0(a_k * In / sigma_k2) * In), 0)
        sigma_k2 = np.maximum(0.5 * filter2B(Mask, np.abs(In)**2) - a_k**2 / 2, 0.01)
    Signal = a_k
    Sigma_n = np.sqrt(sigma_k2)
    return Signal, Sigma_n

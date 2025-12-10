import numpy as np
from scipy.special import iv

def approxI1_I0(z):
    """
    Approximate the ratio I1(z)/I0(z) for the modified Bessel functions.
    Vectorized for numpy arrays.
    """
    z = np.asarray(z)
    M = np.zeros_like(z, dtype=np.float64)
    z8 = 8.0 * z
    Mn = 1 - 3.0 / z8 - 15.0 / 2.0 / (z8 ** 2) - (3 * 5 * 21) / 6.0 / (z8 ** 3)
    Md = 1 + 1.0 / z8 + 9.0 / 2.0 / (z8 ** 2) + (25 * 9) / 6.0 / (z8 ** 3)
    M = Mn / Md
    # For z < 1.5, use the true Bessel ratio
    cont = z < 1.5
    if np.any(cont):
        M[cont] = iv(1, z[cont]) / iv(0, z[cont])
    # For z == 0, set to 0
    M[z == 0] = 0.0
    return M

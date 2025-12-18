"""
Aja-Fernandez et al. homomorphic spatially variant noise estimation

Algorithm proposed in:
    Spatially variant noise estimation in MRI: A homomorphic approach
    S Aja-Fernández, T Pieciak, G Vegas-Sánchez-Ferrero
    Medical Image Analysis, 2014
"""

import numpy as np
from scipy.special import iv
from scipy.special import digamma as psi

from .utils import (fourier_lpf, conv3d)

def approxI1_I0(z):
    """
    Approximate the ratio I1(z)/I0(z) for the modified Bessel functions.
    Vectorized for numpy arrays.
    """

    z = np.asarray(z)
    M = np.zeros_like(z, dtype=np.float64)
    z8 = 8.0 * z

    # Divide-by-zero safe powers of z8
    min_float = 1e-12
    z8_1 = z8 + min_float
    z8_2 = z8 ** 2 + min_float
    z8_3 = z8 ** 3 + min_float
    
    Mn = 1 - 3.0 / z8_1 - 15.0 / 2.0 / z8_2 - (3 * 5 * 21) / 6.0 / z8_3
    Md = 1 + 1.0 / z8_1 + 9.0 / 2.0 / z8_2 + (25 * 9) / 6.0 / z8_3

    M = Mn / (Md + min_float)
    
    # For z < 1.5, use the true Bessel ratio
    cont = z < 1.5
    if np.any(cont):
        M[cont] = iv(1, z[cont]) / iv(0, z[cont])
    
    # For z == 0, set to 0
    M[z == 0] = 0.0
    
    return M

def em_ml_rice3D(img, em_niter=5, em_ksize=3):
    """
    EM implementation of Maximum Likelihood for Rician data (3D).

    img: Input data (Rician image)
    niter: Number of EM iterations
    ksize: kernel size for local averaging (integer)
    returns: img_denoised, img_sigma_n
    """

    # Kernel array weights
    h = np.ones((em_ksize, em_ksize, em_ksize)) / em_ksize**3

    # Initialize a_k and sigma_k2
    a_k = np.sqrt(np.sqrt(np.maximum(2 * conv3d(h, img**2)**2 - conv3d(h, img**4), 0)))
    sigma_k2 = 0.5 * np.maximum(conv3d(h, img**2) - a_k**2, 0.01)
    
    for _ in range(em_niter):
        a_k = np.maximum(conv3d(h, approxI1_I0(a_k * img / sigma_k2) * img), 0)
        sigma_k2 = np.maximum(0.5 * conv3d(h, np.abs(img)**2) - a_k**2 / 2, 0.01)

    img_denoised = a_k
    img_sigma_n = np.sqrt(sigma_k2)

    return img_denoised, img_sigma_n

def correct_rice_gauss(snr):
    """
    Rician-Gaussian correction for noise estimation

    snr: signal SNR (scalar or array)
    Returns: correction curve Fc
    """
    
    snr = np.asarray(snr, dtype=np.float64)

    # Coefficients for Rician-Gaussian correction polynomial
    snr_coeffs = [
        -0.289549906258443,
        -0.0388922575606330,
        0.409867108141953,
        -0.355237628488567,
        0.149328280945610,
        -0.0357861117942093,
        0.00497952893859122,
        -0.000374756374477592,
        1.18020229140092e-05
    ]

    Fc = (
        snr_coeffs[0] +
        snr_coeffs[1] * snr +
        snr_coeffs[2] * snr**2 +
        snr_coeffs[3] * snr**3 +
        snr_coeffs[4] * snr**4 +
        snr_coeffs[5] * snr**5 +
        snr_coeffs[6] * snr**6 +
        snr_coeffs[7] * snr**7 +
        snr_coeffs[8] * snr**8
    )
    
    Fc = Fc * (snr <= 7)

    return Fc

def rice_homomorphic_est(img_noisy: np.ndarray):
    """
    Homomorphic Rician noise estimation adapted for 3D scalar magnitude data
    Use the following original MATLAB parameters:
    1. Low pass filter Gaussian sigma in Fourier domain = 4.8 voxels
    2. Estimate SNR from data using EM (niter = 10, ksize = 3) [SNR=0, Modo 2]
    
    PARAMETER: img_noisy: 3D noisy magnitude data
    """

    print("\nRunning homomorphic Rician noise estimation ...")

    # Divide-by-zero safe small float
    small_float = 1e-12
    
    # Digamma (psi) function based scaling factor
    digamma_sf = np.sqrt(2.0) * np.exp(-psi(1) / 2.0)

    # Low pass filter Gaussian sigma in the frequency domain in voxels
    sigma_freq = 4.8

    # EM parameters
    em_niter = 10
    em_ksize = 3

    # Estimate mean and noise using EM

    # Original 2D MATLAB code in comments for reference
    # [M2 Sigma_n]=em_ml_rice2D(In, 10, [3,3]);
    # M2 -> img_denoised
    # Sigma_n -> img_sigma_n
    # In -> img_noisy
    print("  Estimating SNR using EM maximum likelihood ...")
    img_denoised, img_sigma_n = em_ml_rice3D(img_noisy, em_niter=em_niter, em_ksize=em_ksize)

    # Unused -> skip
    # % Low pass filter of noise to avoid high frequancy components
    # Sigma_n2 = lpf(Sigma_n,1.2);
    # % Local Mean
    # M1 = filter2B(ones(5)./25, In);

    # %SNR estimation (using RIcian EM)
    # if (length(SNR)==1)&&(SNR==0)
    #     SNR=M2./Sigma_n;
    # end
    # SNR -> img_snr_init
    img_snr_init = img_denoised / (img_sigma_n + small_float)

    # Homomorphic filtering

    # Skip Gaussian noise estimation

    # Rician noise estimation
    # if Modo==1
    #     LocalMean=M1;
    # elseif Modo==2
    #     LocalMean=M2;
    # else
    #     LocalMean=0;
    # end
    # Modo 2: use EM denoised image as local mean
    # LocalMean -> img_denoised (Modo 2)

    # Rn=abs(In-LocalMean);
    img_noise = img_noisy - img_denoised
    Rn_rice = np.abs(img_noise)

    # lRn=log(Rn.*(Rn~=0)+0.001.*(Rn==0));
    lRn_rice = np.log(Rn_rice * (Rn_rice != 0) + 0.001 * (Rn_rice == 0))

    # LPF2=lpf((lRn),LPF);
    # LPF2 -> lRn_rice_lpf
    lRn_rice_lpf = fourier_lpf(lRn_rice, sigma_freq=sigma_freq)

    # Fc1=correct_rice_gauss(SNR);
    print("  Applying Rician-Gaussian correction ...")
    Fc1 = correct_rice_gauss(img_snr_init)

    # LPF1=LPF2-Fc1;
    # LPF1 -> lRn_rice_corrected
    lRn_rice_corrected = lRn_rice_lpf - Fc1

    # LPF1=lpf((LPF1),LPF+2,2);
    # Original MATLAB increase frequency domain sigma by 2 for second LPF
    # LPF1 -> lRn_rice_corrected_lpf
    lRn_rice_corrected_lpf = fourier_lpf(lRn_rice_corrected, sigma_freq=sigma_freq + 2.0)

    # Mapa1=exp(LPF1);
    # Mapa1 -> Rn_rice_corrected_lpf
    # MapaR=Mapa1.*2./sqrt(2).*exp(-psi(1)./2);
    # MapaR -> img_sigmamap
    print("  Computing final Rician noise sigma map ...")
    img_sigmamap = digamma_sf * np.exp(lRn_rice_corrected_lpf)
    
    # Final SNR map
    img_snr = img_denoised / (img_sigmamap + small_float)

    # No spatial masking used for homomorphic estimation
    signal_mask = None

    return img_denoised, img_noise, img_sigmamap, img_snr, signal_mask

"""
ORIGINAL MATLAB CODE from Matlab Central

function [MapaR MapaG]=rice_homomorf_est(In,SNR,LPF,Modo)
%
% RICE_HOMOMORF_EST Noise estimation in SENSE MR 
%  [MapaR MapaG Sigma_n2 Sigma_n]=rice_homomorf_est(In,Tipo,SNR,LPF)
%  estimates the variable map of noise out of SENSE magnetic resonance
%  data using a homomorphic approach.
%
% V1.0
%
% USAGE:
%
%   NoiseMap=rice_homomorf_est(In);
%   NoiseMap=rice_homomorf_est(In,SNR,4.5);
%
% GENERAL:
%
%   [MapaR MapaG]=rice_homomorf_est(In,SNR,LPF)
%
% INPUTS:
%       In:     noisy rician data
%
% OPTIONAL INPUTS
%
%       SNR:    Signal to noise ratio
%               =0 estimated from data using EM
%       LPF     Threshold for LPF 
%               ==0 default value (4.8)
%       Modo    Calculation of the local mean to be substracted to the image
%               1: Local mean
%               2: ML estimation of signal and noise (Default)
%               3: No substraction of the mean
%
% OUTPUT
%       MapaR:  Rician Map
%       MapaG:  Gaussian Map
%
% Algorithm proposed in:
%
%       Spatially variant noise estimation in MRI: A homomorphic approach
%       S Aja-Fernández, T Pieciak, G Vegas-Sánchez-Ferrero
%       Medical Image Analysis, 2014
%
%
% Santiago Aja-Fernandez (V1.0)
% LPI 
% www.lpi.tel.uva.es/~santi
% sanaja@tel.uva.es
% LPI Valladolid, Spain
% Original: 06/07/2014, 
% Release   16/12/2014

if( nargin<1 )
    error('At least the input image and modos shouls be provided');
end
if( nargin<2 )
    SNR=0;
    LPF=4.8;
    Modo=2;
end
if( nargin<3 )
    LPF=4.8;
    Modo=2;
end
if( nargin<4 )
    Modo=2;
end

% Euler gamma:
eg=0.5772156649015328606;

%Prior Values------------

%Estimate mean and noise using EM
[M2 Sigma_n]=em_ml_rice2D(In,10,[3,3]);
%Low pass filter of noise to avoid high frequancy components
Sigma_n2=lpf(Sigma_n,1.2);
% Local Mean
M1=filter2B(ones(5)./25,In);


%SNR estimation (using RIcian EM)
if (length(SNR)==1)&&(SNR==0)
    SNR=M2./Sigma_n;
end

%Homomorfic filtering-------------------
%Gauss----------------------
Rn=abs(In-M1);
lRn=log(Rn.*(Rn~=0)+0.001.*(Rn==0));
LPF2=lpf((lRn),LPF);
Mapa2=exp(LPF2);
MapaG=Mapa2.*2./sqrt(2).*exp(-psi(1)./2);

%Rician-------------------------
if Modo==1
    LocalMean=M1;
elseif Modo==2
    LocalMean=M2;
else
    LocalMean=0;
end

Rn=abs(In-LocalMean);
lRn=log(Rn.*(Rn~=0)+0.001.*(Rn==0));
LPF2=lpf((lRn),LPF);
Fc1=correct_rice_gauss(SNR);
LPF1=LPF2-Fc1;
LPF1=lpf((LPF1),LPF+2,2);
Mapa1=exp(LPF1);
MapaR=Mapa1.*2./sqrt(2).*exp(-psi(1)./2);
"""
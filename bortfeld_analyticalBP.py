""" 
This file was tested in Python 2.7, Numpy 1.8.0, Scipy 0.13.3
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, pbdv

def cal_bragg_nostraggle(E, z):
    """
    Calculates the Bragg peak based on the Bortfeld equation without range straggling. (Bortfeld 1996)
    :param E: scalar. This is the initial energy of the proton beam, in MeV.
    :param z: a numpy array of positions to calculate the dose. The unit is centimeter.
    :return: (dose, index of bragg peak: the maximum value of the dose vector)
    """
    p = 1.77  # Unitless
    alpha = 2.2e-3  # Unitless
    g = 0.6  # Unitless
    beta = 0.012  # 1 / cm
    phiIN = 0.0001  # Primary fluence
    rho = 1.00  # gm / cm3(Mass density of the medium)
    R = alpha * E ** p  # cm(Range, neglecting straggling)
    # phi = phiIN * (1 + beta * (R - z)) / (1 + beta * R)
    a1 = phiIN / (rho * p * (alpha ** (1 / p)) * (1 + beta * R))
    a2 = phiIN * (beta + g * beta * p) / (rho * p * (alpha ** (1 / p)) * (1 + beta * R))
    pdd = a1 * (R - z) ** ((1 / p) - 1) + a2 * ((R - z) ** (1 / p))
    pdd[z > R] = 0  # unit is MeV/g, to get grey multiply by 10^9 * e
    pdd = np.array(pdd)
    return pdd, np.argmax(pdd)

def cal_bragg_straggle(E, z, straggle):
    """
    Calculates the Bragg peak based on the Bortfeld equation considering range straggling.
    This uses parabolic cylinder function, which is a fit to the Gaussian convolution with non-straggled curve (Bortfeld).
    :param E: scalar. Initial beam energy in MeV.
    :param z: a numpy array of positions to calculate the dose. The unit is centimeter.
    :param straggle: scalar. Initial energy sigma in MeV.
    :return: (dose, index of bragg peak: the maximum value of the dose vector)
    """
    p = 1.77  # Unitless
    alpha = 2.2e-3  # Unitless
    g = 0.6  # Unitless
    beta = 0.012  # 1 / cm
    phiIN = 0.0001  # Primary fluence
    # phiIN = 0.1
    rho = 1.00  # gm / cm3(Mass density of the medium)
    R = alpha * E ** p  # cm(Range, neglecting straggling)

    sigmamono = 0.012 * (R ** 0.935)    # 2nd March, remember to test with or without brackets same result.
    sigmaEzero = straggle
    sigmatotal = np.sqrt(sigmamono ** 2 + (sigmaEzero ** 2) * (alpha ** 2) * (p ** 2) * (E ** (2 * p - 2)))
    eta = (R - z) * 1. / sigmatotal
    epsilon = 0.0
    # print 'eta = ', eta
    pdd = []
    denom = np.sqrt(2 * np.pi) * rho * p * (alpha ** (1 / p)) * (1 + beta * R)
    for zd in eta:
        para1 = pbdv(-1 / p, -zd)[0]
        para2 = pbdv(-1 / p - 1, -zd)[0]
        numerator = phiIN * (np.exp(-(zd ** 2) / 4) * (sigmatotal ** (1 / p)) * gamma(1 / p))
        add1 = 1./sigmatotal * para1
        add2 = (beta / p + g * beta + epsilon / R) * para2
        pdd_zd = (numerator / denom) * (add1 + add2)
        pdd.append(pdd_zd)
    pdd = np.array(pdd)
    dose_nostraggle, _ = cal_bragg_nostraggle(E=E, z=z)
    nan_index = np.argwhere(~np.isfinite(pdd))
    pdd[nan_index] = dose_nostraggle[nan_index]
    return pdd, np.argmax(pdd)

z=np.arange(0, 20, 0.1)             # Create depth coordinate vector. (Start, end, step size) in centimeters.
b, _ = cal_bragg_straggle(E=150, z=z, straggle=3)
plt.figure()
plt.plot(z, b)
plt.xlabel('Depth (centimeter)')
plt.ylabel('Dose (arbitrary unit)')
plt.show()

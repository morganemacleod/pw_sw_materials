import numpy as np
from scipy.interpolate import interp1d
#import sys
#sys.path.insert(0, '/home/antonija/Documents/MM-athena-public-version-master/analysis')
from Constants import Constants
c=Constants()
#############################################################
# constants

#temperature = 10000.0

# mass fractions of hydrogen, helium, and metals
Xuni = 0.738
Yuni = 0.248
Zuni = 1.0 - Xuni - Yuni

# hydrogen recombination rate
alpha = 2.59e-13

# helium recombination rates
alpha1 = 6.23e-14
alpha3 = 2.1e-13

# hydrogen photoionization rate and cross section
phi0_1au = 7.32376502602e-7 #@ 1au
sigma_photo_nu0 = 2.98367621486e-18  #2.98367621486e-18 #6.3e-18 # cm^2

# helium photoionization rates and cross sections
phi1_1au = 2.75e-7
phi3_1au = 0.0001268

sigma_photo_nu1 = 7.82000000e-18
sigma_photo_nu3 = 4.86096491e-18

# helium collisional coefficients
q13a = 4.5e-20
q31a = 2.6e-8
q31b = 4.0e-9
Q31 = 5.0e-10

coll_T = np.concatenate((np.arange(2000, 21000, 1000), np.array([30000, 40000, 50000])), axis=0)
q13_values = np.array([1.31e-58, 4.82e-42, 8.89e-34, 7.92e-29, 1.56e-25, 3.49e-23, 2.0e-21, 4.64e-20, 5.73e-19, 4.41e-18, 2.41e-17, 1.01e-16, 3.44e-16, 9.92e-16, 2.50e-15, 5.64e-15, 1.16e-14, 2.21e-14, 3.94e-14, 1.47e-12, 8.51e-12, 2.37e-11])
q31a_values = np.array([1.48e-9, 5.68e-9, 1.07e-8, 1.53e-8, 1.91e-8, 2.22e-8, 2.46e-8, 2.66e-8, 2.81e-8, 2.89e-8, 2.94e-8, 2.96e-8, 2.98e-8, 2.97e-8, 2.96e-8, 2.94e-8, 2.92e-8, 2.89e-8, 2.86e-8, 2.43e-8, 2.01e-8, 1.64e-8])
q31b_values = np.array([1.28e-11, 1.65e-10, 5.81e-10, 1.23e-9, 2.02e-9, 2.87e-9, 3.75e-9, 4.61e-9, 5.45e-9, 6.09e-9, 6.67e-9, 7.18e-9, 7.65e-9, 8.07e-9, 8.45e-9, 8.8e-9, 9.11e-9, 9.39e-9, 9.65e-9, 1.14e-8, 1.23e-8, 1.30e-8])

q13a_func = interp1d(coll_T, q13_values, bounds_error=False, fill_value = (q13_values[0], q13_values[-1]))
q31a_func = interp1d(coll_T, q31a_values, bounds_error=False, fill_value = (q31a_values[0], q31a_values[-1]))
q31b_func = interp1d(coll_T, q31b_values, bounds_error=False, fill_value = (q31b_values[0], q31b_values[-1]))

def q13a_approx_func(T):
    logT = np.where(T<5.e4,np.log10(T),4.69897)
    logq = (-11137.665388772013 + 12004.264662098845*logT 
            -5248.4474574580945*logT**2 + 1159.9322424449788*logT**3 
            -129.32459151064742*logT**4 + 5.809518563834687*logT**5)
    return 10**logq

def q31a_approx_func(T):
    logT = np.where(T<5.e4,np.log10(T),4.69897)
    logq = (-207.46953026927196 + 176.49185283145317*logT 
            -58.97245658259958*logT**2 + 8.864833017158752*logT**3 
            -0.5072061695014867*logT**4)
    return 10**logq

def q31b_approx_func(T):
    logT = np.where(T<5.e4,np.log10(T),4.69897)
    logq = (-305.8645512077057 + 250.89047828967537*logT 
            -79.55573380402501*logT**2 + 11.252375193197931*logT**3 
            -0.598559329753177*logT**4)
    return 10**logq


# metastable state radiative decay rate
A31 = 1.272e-4

# 1083 nm line natural broadening
nu1=2.76807989669e+14
nu2=2.76810281474e+14
nu3=2.76839906622e+14

natural_gamma = 1.0216e7/(4.0*np.pi)
#doppler_alpha1 = np.sqrt(2.0*np.log(2.0))*nu1*np.sqrt(0.25*c.kB*temperature/c.mp)/c.c
#doppler_alpha2 = np.sqrt(2.0*np.log(2.0))*nu2*np.sqrt(0.25*c.kB*temperature/c.mp)/c.c
#doppler_alpha3 = np.sqrt(2.0*np.log(2.0))*nu3*np.sqrt(0.25*c.kB*temperature/c.mp)/c.c

# 1083 nm line absorption cross sections
cs1=0.29958*np.pi*4.8032e-10*4.8032e-10/(9.1e-28*c.c)
cs2=0.17974*np.pi*4.8032e-10*4.8032e-10/(9.1e-28*c.c)
cs3=0.059902*np.pi*4.8032e-10*4.8032e-10/(9.1e-28*c.c)
#print cs1, cs2, cs3

# limb darkening coeffs
ld1 = 0.034943203                
ld2 = 0.37691500
###############################################################
# wavelength (angstroms) and frequency (Hz) grids 
lamb = np.linspace(10827.0, 10833.0, 400)
nu = c.c / (lamb * 1e-8)

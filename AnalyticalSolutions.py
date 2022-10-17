from scipy.special import hankel1
import numpy as np
# definition of the analytical solution: scipy does not support the spherical Hankel function, so we define it in terms of the Hankel function
def spherical_hankel(n,z):
    return np.sqrt(np.pi/(2*z))*hankel1(n+0.5,z)

def d_spherical_hankel(n,z):
    return -spherical_hankel(n+1,z) + (n/z) * spherical_hankel(n,z)
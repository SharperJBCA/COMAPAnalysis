import numpy as np

# FUNCTION FOR FITTING ROUTINES

def Plane(P, x, y):
    
    return P[0]*(x) + P[1]*(y) 

def Gauss2d(P, x, y, ra_c, dec_c, plane=False):
    
    X = (x - ra_c - P[2])
    Y = (y - dec_c - P[3])

    a = (X/P[1])**2
    b = (Y/P[1])**2

    model = P[0] * np.exp( - 0.5 * (a + b)) + P[4]
    if plane:
        model += Plane([P[5], P[6]], x, y)
    return model


def Gauss2dElliptical(P, x, y, ra_c, dec_c, plane=False):
    
    X = (x - ra_c - P[4])
    Y = (y - dec_c - P[5])
    Xr =  np.cos(P[6]) * X + np.sin(P[6]) * Y
    Yr = -np.sin(P[6]) * X + np.cos(P[6]) * Y

    a = (Xr/P[1])**2
    b = (Yr/P[2])**2

    model = P[0] * np.exp( - 0.5 * (a + b)) + P[4]
    if plane:
        model += Plane([P[6], P[7]], x, y)
    return model

# Likelihood for fmin function

def ErrorFmin(P, x, y, z, ra_c, dec_c, plane=False):

    if (P[1] > 6./60./2.355) | (P[1] < 0) | (P[0] < 0) | (np.sqrt(P[2]**2 + P[3]**2) > 60./60.): 
        return  1e32
    else:
        return np.sum((z - Gauss2d(P, x, y, ra_c, dec_c, plane))**2)

# MCMC Fitting 
def lnlike(P, x, y,z, yerr, plane=False):
    model = Gauss2d2FWHM(P, x, y, 0,0)
    inv_sigma2 = 1.0/(yerr**2)#  + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((z-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprior(P):
    A, siga, sigb, x0, y0, bkgd = P
    r = np.sqrt(x0**2 + y0**2)
    if 0 < A < 1e8 and 0.0 < np.sqrt(siga**2 + sigb**2) < 10./60./2.355 and 0 < r < 1.0:
        return 0.0
    return -np.inf
    
def lnprob(P, x, y, z, yerr, plane=False):
    lp = lnprior(P)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(P, x, y, z, yerr, plane)

# Error Lstsq
def ErrorLstSq(P, x, y, z, ra_c, dec_c, plane=False):

    if (P[1] > 10./60./2.355) | (P[1] < 0) | (P[0] < 0) | (np.sqrt(P[2]**2 + P[3]**2) > 60./60.): 
        return 0.*z + 1e32
    else:
        return z - Gauss2d(P, x, y, ra_c, dec_c, plane)

def Gauss2d2FWHM(P, x, y, ra_c, dec_c, plane=False):
    
    X = (x - ra_c - P[3])
    Y = (y - dec_c - P[4])

    a = (X/P[1])**2
    b = (Y/P[2])**2

    model = P[0] * np.exp( - 0.5 * (a + b)) + P[5]

    return model

def Gauss2d2FWHMFixed(P, x, y, ra_c, dec_c, plane=False):
    
    X = (x - ra_c)
    Y = (y - dec_c)

    a = (X/P[1])**2
    b = (Y/P[2])**2

    model = P[0] * np.exp( - 0.5 * (a + b)) + P[3]

    return model



def ErrorLstSq2FWHM(P, x, y, z, ra_c, dec_c, plane=False):

    if (P[1] > 10./60./2.355) | (P[1] < 0) | (P[2] > 10./60./2.355) | (P[2] < 0) | (P[0] < 0): 
        return 0.*z + 1e32
    else:
        return z - Gauss2d2FWHMFixed(P, x, y, ra_c, dec_c, plane)

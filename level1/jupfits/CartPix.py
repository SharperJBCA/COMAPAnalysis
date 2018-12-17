import numpy as np
from matplotlib import pyplot

from astropy import wcs
from astropy.io import fits

def Info2WCS(naxis, cdelt, crval, ctype=['RA---TAN', 'DEC--TAN']):
    """
    """
    # Setup 2D wcs object
    w = wcs.WCS(naxis=2)


    w.wcs.crpix = [naxis[0]/2.+1, naxis[1]/2.+1]
    w.wcs.cdelt = np.array([-cdelt[0], cdelt[1]])
    w.wcs.crval = [crval[0], crval[1]]
    w.wcs.ctype = ctype

    return w

def ang2pix(naxis, cdelt, crval, theta, phi, ctype=['RA---TAN', 'DEC--TAN']):
    """
    """
    
    # Setup 2D wcs object
    w = Info2WCS(naxis, cdelt, crval, ctype=ctype)
    
    # Generate pixel coordinates
    pixcrd = np.floor(np.array(w.wcs_world2pix(phi, theta, 0))).astype('int64')

    bd = ((pixcrd[0,:] < 0) | (pixcrd[1,:] < 0)) | ((pixcrd[0,:] >= naxis[0]) | (pixcrd[1,:] >= naxis[1])) 

    pmax, pmin = (crval[0] + cdelt[0]*naxis[0]), (crval[0] - cdelt[0]*naxis[0])
    tmax, tmin = (crval[1] + cdelt[1]*naxis[1]), (crval[1] - cdelt[1]*naxis[1])
    cbd = (phi > pmax) | (phi <= pmin+1) | (theta <= tmin+1) | (theta > tmax)
    pix = pixcrd[0,:]*int(naxis[1]) + pixcrd[1,:]
    pix = pix.astype('int')

    pix[bd] = -1


    npix = int(naxis[0]*naxis[1])

    return pix

def pix2ang(naxis, cdelt, crval,  xpix, ypix, ctype=['RA---TAN', 'DEC--TAN']):

    # Setup 2D wcs object
    w = Info2WCS(naxis, cdelt, crval, ctype=ctype)

    # Generate pixel coordinates
    pixcrd = np.array(w.wcs_pix2world(xpix, ypix, 0))


    return pixcrd[1,:], pixcrd[0,:]

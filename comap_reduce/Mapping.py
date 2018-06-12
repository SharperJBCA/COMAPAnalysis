import numpy as np
import CartPix
from matplotlib import pyplot

def DefineWCS(naxis=[100,100], cdelt=[1./60., 1./60.],
              crval=[0,0]):

    wcs = CartPix.Info2WCS(naxis, cdelt, crval)
 # Setup WCS
    #naxis = [100, 100]
    #cdelt = [2./naxis[0], 2./naxis[1]]
    #crval = [rac, decc]
    #wcs = CartPix.Info2WCS(naxis, cdelt, crval)
    #npix = naxis[0]*naxis[1]

    return wcs

def ang2pixWCS(wcs, ra, dec):
    naxis = [int((wcs.wcs.crpix[0]-1.)*2.), int((wcs.wcs.crpix[1]-1.)*2.)]

    pix = CartPix.ang2pix(naxis, wcs.wcs.cdelt, wcs.wcs.crval, ra, dec).astype('int')
    
    return pix

def MakeMaps(tod, ra, dec, wcs):

    naxis = [int((wcs.wcs.crpix[0]-1.)*2.), int((wcs.wcs.crpix[1]-1.)*2.)]
    npix = naxis[0]*naxis[1]

    pixbins = np.arange(0, npix+1)

    nhorns = tod.shape[0]
    nsidebands = tod.shape[1]
    nchans  = tod.shape[2]

    maps = np.zeros((nhorns, nsidebands, nchans, naxis[0], naxis[1]))
    hits = np.zeros((nhorns, naxis[0], naxis[1]))

    for i in range(nhorns):
        pix = ang2pixWCS(wcs, dec[i,:], ra[i,:]).astype('int')

        h, b = np.histogram(pix, pixbins)
        hits[i,:,:] = np.reshape(h, (naxis[0], naxis[1]))

        for j in range(nsidebands):
            for k in range(nchans):
                w, b = np.histogram(pix, pixbins, weights=tod[i,j,k,:])
                maps[i,j,k,:,:] = np.reshape(w, (naxis[0], naxis[1]))
    return maps, hits

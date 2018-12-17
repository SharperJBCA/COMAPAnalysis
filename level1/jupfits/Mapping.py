import numpy as np
import CartPix
from matplotlib import pyplot

def DefineWCS(naxis=[100,100], cdelt=[1./60., 1./60.],
              crval=[0,0], ctype=['RA---TAN', 'DEC--TAN']):

    wcs = CartPix.Info2WCS(naxis, cdelt, crval, ctype=ctype)
 # Setup WCS
    #naxis = [100, 100]
    #cdelt = [2./naxis[0], 2./naxis[1]]
    #crval = [rac, decc]
    #wcs = CartPix.Info2WCS(naxis, cdelt, crval)
    #npix = naxis[0]*naxis[1]
    xpix, ypix= np.meshgrid(np.arange(naxis[0]), np.arange(naxis[1]),indexing='ij')
    #ypix = ypix[:,::-1]
    yr, xr = CartPix.pix2ang(naxis, cdelt, crval,  ypix, xpix)

    xr[xr > 180] -= 360
    return wcs, xr, yr

def ang2pixWCS(wcs, ra, dec, ctype=['RA---TAN', 'DEC--TAN']):
    naxis = [int((wcs.wcs.crpix[0]-1.)*2.), int((wcs.wcs.crpix[1]-1.)*2.)]

    pix = CartPix.ang2pix(naxis, wcs.wcs.cdelt, wcs.wcs.crval, ra, dec, ctype=ctype).astype('int')
    
    return pix

def MakeMapSimple(tod, ra, dec, wcs):
    naxis = [int((wcs.wcs.crpix[0]-1.)*2.), int((wcs.wcs.crpix[1]-1.)*2.)]
    npix = naxis[0]*naxis[1]

    pixbins = np.arange(0, npix+1)

    pix = ang2pixWCS(wcs, ra, dec).astype('int')
    h, b = np.histogram(pix, pixbins)
    hits = np.reshape(h, (naxis[0], naxis[1]))
    w, b = np.histogram(pix, pixbins, weights=tod)
    maps = np.reshape(w, (naxis[0], naxis[1]))
    return maps[:,::-1], hits[:,::-1]


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

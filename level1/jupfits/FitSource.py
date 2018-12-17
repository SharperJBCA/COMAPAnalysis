import numpy as np
from scipy.optimize import leastsq, fmin
from matplotlib import pyplot
from scipy.interpolate import interp1d
import Pointing
from scipy.ndimage.filters import median_filter
from scipy.ndimage.filters import gaussian_filter,maximum_filter, gaussian_filter1d
from scipy.signal import medfilt
from skimage.feature import peak_local_max
import scipy

import emcee
from scipy.stats import kurtosis

import Mapping
from Fitting import *
import CartPix
import corner

naxis = [100, 100]
cdelt = [1.5/60., 1.5/60.]
crval =[0., 0.]

def RemoveBackground(_tod, rms,x,y, sampleRate=50, cutoff=1.):
    """
    Takes the TOD and set of indices describing the location of the source.
    Fits polynomials beneath the source and then applies a low-pass filter to the full
    data. It returns this low pass filtered data
    """
    time = np.arange(_tod.size)
    tod = _tod*1.

    # initial pass
    background = medfilt(tod[:], 35)
    peakarg = np.argmax(tod-background)
    x0, y0 = x[peakarg], y[peakarg]
    r = np.sqrt((x-x0)**2 + (y-y0)**2)
    close = (r < 15./60.)

    
    # First we will find the beginning and end samples of the source crossings
    timeFit = time[close]
    timeZones = (timeFit[1:] - timeFit[:-1])
    timeSelect= np.where((timeZones > 5))[0]
    closeIndex = np.where(close)[0]
    indices = (closeIndex[:-1])[timeSelect]
    indices = np.concatenate((closeIndex[0:1], indices, (np.where(close)[0][:-1])[timeSelect+1], [closeIndex[-1]]))
    indices = np.sort(indices)
                
    # For each source crossing fit a polynomial using the data just before and after
    
    for m in range(indices.size//2):
        lo, hi = indices[2*m], indices[2*m+1]
        lo = max([lo, 0])
        hi = min([hi, tod.size])
        fitRange = np.concatenate((np.arange(lo-sampleRate,lo), np.arange(hi, hi+sampleRate))).astype(int)
        dmdl = np.poly1d(np.polyfit(time[fitRange], tod[fitRange],3))

        tod[lo:hi] = np.random.normal(scale=rms, loc=dmdl(time[lo:hi]))
                
    # apply the low-pass filter
    Wn = cutoff/(sampleRate/2.)
    b, a = scipy.signal.butter(4, Wn, 'low')
    background = gaussian_filter1d(tod[:], 55)

    #scipy.signal.filtfilt(b, a, tod[:])
    return background

def ImagePeaks(image, xgrid, ygrid, threshold):
    msmooth = [gaussian_filter(image, fsmooth) for fsmooth in [1,3]]
    dsmooth = msmooth[0] - msmooth[1]
    dsmooth = median_filter(dsmooth, 3)

    maximage = maximum_filter(dsmooth, 3)
    maxPixels = np.array(np.where(maximage==np.max(maximage)))
    maxPix = np.mean(maxPixels,axis=1).astype(int)

    if np.max(maximage) < threshold*5:
        return None, None, None, None
    else:
        #return xgrid[maxPix[0], maxPix[1]], ygrid[maxPix[0], maxPix[1]], maxPix[0], maxPix[1]
        return xgrid[maxPix[0], maxPix[1]], ygrid[maxPix[0], maxPix[1]], maxPix[0], maxPix[1]

def CalcRMS(tod):
    nSamps = tod.shape[-1]
    # Calculate RMS from adjacent pairs
    splitTOD = (tod[...,:(nSamps//2) * 2:2] - tod[...,1:(nSamps//2)*2:2])
    rms = np.std(splitTOD,axis=-1)/np.sqrt(2)
    return rms

badval = (None, None, None, None, None, None)
def FitTOD(tod, ra, dec, clon, clat, cpang, 
           prefix='', 
           normalize=True, 
           plotDir=None):
    """
    args:
    tod   - 
    ra    - 
    dec   - 
    clon  -
    clat  -
    cpang -

    kwargs:
    prefix      -
    destripe    -
    normalize   -
    justOffsets -
    """

    # Define the pixel grid
    # Pixel coordinates on sky
    wcs, xr, yr = Mapping.DefineWCS(naxis, cdelt, crval)
    r = np.sqrt((xr)**2 + (yr)**2)   

    # Pixel coordinates in image
    xpix, ypix = np.meshgrid(np.arange(xr.shape[0]), np.arange(yr.shape[1]), indexing='ij')

    # Calculate RMS from adjacent pairs
    rms = CalcRMS(tod)


    # Rotate the RA/DEC to the source centre
    x, y = Pointing.Rotate(ra, dec, clon, clat, -cpang)

    r = np.sqrt((x)**2 + (y)**2)
    close = (r < 3.) # Check if source is even with 3 degrees of field centre
    if np.sum((r < 6./60.)) < 10:
        print('Source not observed')
        return badval

    # Filter background or at least subtract a mean level
    try:
        todBackground = RemoveBackground(tod, rms, x, y, sampleRate=50, cutoff=1.)
        tod -= todBackground
    except (ValueError, IndexError):
        tod -= np.nanmedian(tod)

    

    # Create map of data centred on 0,0
    ms, hits = Mapping.MakeMapSimple(tod, x, y, wcs)
    m = ms/hits

    # Calculate the pair subtracted TOD to creat a residual map
    residTod = tod[:tod.size//2 * 2:2] - tod[1:tod.size//2 * 2:2]
    residmap, rh = Mapping.MakeMapSimple(residTod, x[:(tod.size//2) * 2:2], y[:(tod.size//2) * 2:2], wcs)
    residmap = residmap/rh
    mapNoise = np.nanstd(residmap)/np.sqrt(2)

    m -= np.nanmedian(m)
    m[np.isnan(m)] = 0.

    # Get an estimate of the peak location
    x0, y0, xpix0, ypix0 = ImagePeaks(m, xr, yr, mapNoise)
    
    if isinstance(x0, type(None)):
        print('No peak found')
        return badval

    # Just select the near data and updated peak location
    # Probably should add some way of not having these be hardcoded...
    r = np.sqrt((x-x0)**2 + (y-y0)**2)
    close = (r < 12.5/60.)
    near  = (r < 25./60.) & (r > 15/60.)
    far   = (r > 30./60.)
    fitselect = (r < 10./60.) & (np.isnan(tod) == False)
    plotselect = (r < 45./60.)

    if np.sum(fitselect) < 20:
        return badval
        
    fitdata = tod[fitselect]
    fitra   = x[fitselect]
    fitdec  = y[fitselect]

    # Initial guesses for fit
    P0 = [np.max(fitdata) -np.median(fitdata) ,
          4./60./2.355,
          4./60./2.355,
          x0,
          y0,
          np.median(fitdata)]


    # Run mcmc fit:
    ndim, nwalkers = len(P0), 100
    pos = [np.array(P0) + 1e-4*np.random.randn(ndim) for iwalker in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(fitra, fitdec, fitdata, rms))
    sampler.run_mcmc(pos, 1200)
    samples = sampler.chain[:, 500:sampler.chain.shape[1]:3, :].reshape((-1, ndim))
    Pest = np.mean(samples, axis=0)
    Pstd = np.std(samples, axis=0)

    chi2 = np.sum((fitdata-Gauss2d2FWHM(Pest, fitra, fitdec, 0,0))**2/rms**2)/(fitdata.size-len(Pest))
    if not isinstance(plotDir, type(None)):
        pyplot.plot(fitdata, label='data')
        pyplot.plot(Gauss2d2FWHM(Pest, fitra, fitdec, 0,0), label='fit')
        pyplot.legend(loc='upper right')
        pyplot.ylabel('T (K)')
        pyplot.xlabel('Sample')
        pyplot.text(0.05,0.9, r'$\chi^2$='+'{:.2f}'.format(chi2), transform=pyplot.gca().transAxes)
        pyplot.title(' {}'.format(prefix))
        pyplot.savefig('{}/PeakFits_{}.png'.format(plotDir, prefix), bbox_inches='tight')
        pyplot.clf()
        #fig = corner.corner(samples)
        #pyplot.title('{}'.format(prefix))
        #pyplot.savefig('{}/Corner_{}.png'.format(plotDir, prefix), bbox_inches='tight')
        #pyplot.clf()
        #del fig

    # Normalise by rms
    if normalize:
        ms /= Pest[0]
    # Output fits + sample of peak crossing
    cross = np.argmax(Gauss2d2FWHM(Pest, x, y, 0,0))

    return Pest, Pstd, cross, ms, hits,Gauss2d2FWHM([1., Pest[1],Pest[2], Pest[3], Pest[4], 0] , xr, yr, 0,0)*outweight

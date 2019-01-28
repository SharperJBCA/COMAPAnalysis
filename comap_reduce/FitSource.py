import numpy as np
from scipy.optimize import leastsq, fmin
from matplotlib import pyplot
from scipy.interpolate import interp1d
import Pointing
from scipy.ndimage.filters import median_filter
from scipy.ndimage.filters import gaussian_filter,maximum_filter
from skimage.feature import peak_local_max
import scipy

import emcee
from scipy.stats import kurtosis

import Mapping
from Fitting import *
import CartPix
import corner

naxis = [100, 100]
cdelt = [1./60., 1./60.]
crval =[0., 0.]

def RemoveBackground(_tod, rms, close, sampleRate=50, cutoff=1.):
    """
    Takes the TOD and set of indices describing the location of the source.
    Fits polynomials beneath the source and then applies a low-pass filter to the full
    data. It returns this low pass filtered data
    """
    time = np.arange(_tod.size)
    tod = _tod*1.
    
    # First we will find the beginning and end samples the source crossings
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
    background = scipy.signal.filtfilt(b, a, tod[:])
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

def removeNaN(d):
    dnan = np.where(np.isnan(d))[0]
    dgd  = np.where(~np.isnan(d))[0]
    if len(dnan) > 0:
        for nanid in dnan:
            d[nanid] = (d[dgd])[np.argmin((dgd-nanid))]
            d[dnan] = d[dnan-1]


def CalcRMS(tod):
    nSamps = tod.shape[-1]
    # Calculate RMS from adjacent pairs
    splitTOD = (tod[...,:(nSamps//2) * 2:2] - tod[...,1:(nSamps//2)*2:2])
    print(splitTOD.shape)
    rms = np.nanstd(splitTOD,axis=-1)/np.sqrt(2)
    return rms

def Bootstrap(P0, fitdata, fitra, fitdec, error):
    niter = 100
    P1 = np.zeros([niter, len(P0)])
    for i in range(niter):
        ind = np.random.uniform(low=0, high=fitdata.size, size=fitdata.size).astype(int)
        x = fitra[ind]
        y = fitdec[ind]
        z = fitdata[ind]
        P1[i,...], s = leastsq(error, P0, args=(x,y,z, 0,0))

    return np.nanmean(P1, axis=0), np.nanstd(P1, axis=0)

def FitTOD(tod, ra, dec, obs, clon, clat, cpang, prefix='', destripe=False, mode='mode1', justOffsets=True, doPlots=True):
    # Beam solid angle aken from James' report on wiki Optics
    nubeam = np.array([26., 33., 40.])
    srbeam = np.array([2.1842e-6, 1.6771e-6, 1.4828e-6])
    pmdl = interp1d(nubeam, srbeam) # straight interpolation

    nHorns = tod.shape[0]
    nSidebands = tod.shape[1]
    nChans = tod.shape[2]
    nSamps = tod.shape[3]

    if mode == 'mode1':
        nParams = 5
    elif mode == 'mode2':
        nParams = 6
    else:
        nParams = 0
        print('WARNING: No fitting method selected')


    # Define the pixel grid
    # Pixel coordinates on sky
    wcs, xr, yr = Mapping.DefineWCS(naxis, cdelt, crval)
    r = np.sqrt((xr)**2 + (yr)**2)   

    # Pixel coordinates in image
    xpix, ypix = np.meshgrid(np.arange(xr.shape[0]), np.arange(yr.shape[1]), indexing='ij')
    backgroundPixels = np.sqrt((xpix - xr.shape[0]/2.)**2 + (ypix - xr.shape[1]/2.)**2) > xr.shape[0]/3

    # Calculate RMS from adjacent pairs
    rms = CalcRMS(tod)


    # Set up data containers
    crossings = np.zeros(nHorns) # Crossing points
    time = np.arange(nSamps) # Useful for interpolation

    if justOffsets:
        #P1 = np.zeros((nHorns, nParams))
        #P1e = np.zeros((nHorns,  nParams))
        #chis = np.zeros((nHorns))
        P1 = np.zeros((nHorns, nSidebands, nChans, nParams-2))
        errors = np.zeros((nHorns, nSidebands, nChans, nParams-2))
        Pestout = np.zeros((nHorns, nParams))
        Pstdout = np.zeros((nHorns,  nParams))

    else:
        P1 = np.zeros((nHorns, nSidebands, nChans, nParams-2))
        errors = np.zeros((nHorns, nSidebands, nChans, nParams-2))
        Pestout = np.zeros((nHorns, nParams))
        Pstdout = np.zeros((nHorns,  nParams))

    #fig = pyplot.figure(figsize=(16,16))
    for i in  range( nHorns):

        # Rotate the RA/DEC to the source centre
        x, y = Pointing.Rotate(ra[i,:], dec[i,:], clon, clat, -cpang[i,:])

        r = np.sqrt((x)**2 + (y)**2)
        close = (r < 12.5/60.)
        if np.sum((r < 6./60.)) < 10:
            print('Source not observed')
            continue

        # Average the data into a single timestream
        print(tod.shape)
        if tod.shape[2] == 1:
            todTemp = tod[0,0,0,:]
        else:
            todTemp = np.mean(np.mean(tod[i,:,:],axis=0),axis=0) # average all data to get location of peak in data:

        removeNaN(todTemp)

        rmsTemp = CalcRMS(todTemp)

        try:
            todBackground = RemoveBackground(todTemp, rmsTemp, close, sampleRate=50, cutoff=1.)
            todTemp -= todBackground
        except (ValueError, IndexError):
            todTemp -= np.median(todTemp)

        offsets  =0
        print(crval)
        m, hits = Mapping.MakeMapSimple(todTemp, x, y, wcs)
        resid = todTemp[:todTemp.size//2 * 2:2] - todTemp[1:todTemp.size//2 * 2:2]
        residmap, rh = Mapping.MakeMapSimple(resid, x[:(todTemp.size//2) * 2:2], y[:(todTemp.size//2) * 2:2], wcs)
        m = m/hits
        residmap = residmap/rh
        mapNoise = np.nanstd(residmap)/np.sqrt(2)

        m -= np.nanmedian(m)
        m[np.isnan(m)] = 0.

        ipix = Mapping.ang2pixWCS(wcs, x, y).astype('int')
        gd = (np.isnan(ipix) == False) & (ipix >= 0) & (ipix < m.size)
        # Get an estimate of the peak location
        x0, y0, xpix0, ypix0 = ImagePeaks(m, xr, yr, mapNoise)

        if isinstance(x0, type(None)):
            print('No peak found')
            continue

        # Just select the near data and updated peak location
        r = np.sqrt((x-x0)**2 + (y-y0)**2)
        close = (r < 12.5/60.)
        near  = (r < 25./60.) & (r > 15/60.)
        far   = (r > 30./60.)
        fitselect = (r < 10./60.) & (np.isnan(todTemp) == False)
        plotselect = (r < 45./60.)

        if np.sum(fitselect) < 20:
            continue

        fitdata = todTemp[fitselect]
        fitra = x[fitselect]
        fitdec= y[fitselect]

        if mode == 'mode2':
            P0 = [np.max(fitdata) -np.median(fitdata) ,
                  4./60./2.355,
                  4./60./2.355,
                  x0,
                  y0,
                  np.median(fitdata)]

        print(P0,lnprior(P0))
        fout = leastsq(ErrorLstSq, P0, args=(fitra, fitdec, fitdata, 0,0), full_output=True)
        #P0 = fout[0]
        ndim, nwalkers = len(P0), 100
        
        #pos = np.zeros((nwalkers, ndim))
        #pos[:,0] = np.abs(P0[0])*1e-4*np.random.randn(nwalkers)
        #pos[:,1:3] = np.abs(P0[1:3])[np.newaxis,:]*1e-4*np.random.randn((nwalkers,2))
        ###pos[:,3:5] = np.abs(P0[3:5])[np.newaxis,:]+0.1*np.random.randn((nwalkers,2))
        #pos[:,5] =  np.abs(P0[5])*1e-4*np.random.randn(nwalkers)
        #pos = pos.T

        pos = [np.array(P0) + 1e-4*np.random.randn(ndim) for iwalker in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(fitra, fitdec, fitdata, rmsTemp))
        sampler.run_mcmc(pos, 1200)
        samples = sampler.chain[:, 500:sampler.chain.shape[1]:3, :].reshape((-1, ndim))
        print(samples.shape)
        Pest = np.mean(samples, axis=0)
        Pstd = np.std(samples, axis=0)
        #Pest = fout[0]
        chi2 = np.sum((fitdata-Gauss2d2FWHM(Pest, fitra, fitdec, 0,0))**2/rmsTemp**2)/(fitdata.size-len(Pest))
        print( np.std(samples, axis=0))
        if doPlots:
            pyplot.plot(fitdata, label='data')
            pyplot.plot(Gauss2d2FWHM(Pest, fitra, fitdec, 0,0), label='fit')
            pyplot.legend(loc='upper right')
            pyplot.ylabel('T (K)')
            pyplot.xlabel('Sample')
            pyplot.text(0.05,0.9, r'$\chi^2$='+'{:.2f}'.format(chi2), transform=pyplot.gca().transAxes)
            pyplot.title('Horn {:d} obs {}'.format(i+1, prefix))
            pyplot.savefig('PeakFitPlots/PeakFits_Horn{:d}_{}.png'.format(i+1, prefix), bbox_inches='tight')
            pyplot.clf()
            fig = corner.corner(samples)
            pyplot.title('Horn {:d} obs {}'.format(i+1, prefix))
            pyplot.savefig('PeakFitPlots/Corner_Horn{:d}_{}.png'.format(i+1, prefix), bbox_inches='tight')
            pyplot.clf()
            del fig
       # pyplot.figure()
        #pyplot.plot(samples[:,0])
        #pyplot.show()
        if justOffsets:
            #P1[i,:] = Pest
            #P1e[i,:] = np.std(samples, axis=0)
            Pestout[i,:] = Pest
            Pstdout[i,:] = Pstd
            #chis[i] = chi2
            continue
        else:
            Pestout[i,:] = Pest
            Pstdout[i,:] = Pstd
        print(x0, y0)
        siga, sigb = Pest[1:3]
        x0, y0 = Pest[3:5]
        print(x0, y0)





        for j in range(nSidebands):
            
            for k in range(nChans):
                
                try:
                    todBackground = RemoveBackground(tod[i,j,k,:], rms[i,j,k], close, sampleRate=50, cutoff=0.1)
                except (IndexError, ValueError):
                    todBackground = np.median(tod[i,j,k,:])

                tod[i,j,k,:] -= todBackground
                fitdata = tod[i,j,k,fitselect]
                fitra = x[fitselect]
                fitdec= y[fitselect]

                amax = np.argmax(fitdata)


                if mode == 'mode1':
                    P0 = [np.max(fitdata) -np.median(fitdata) ,
                          4./60./2.355,
                          x0,
                          y0,
                          np.median(fitdata)]
                    fout = leastsq(ErrorLstSq, P0, args=(fitra, fitdec, fitdata, 0,0), full_output=True)
                    fbootmean, fbootstd = Bootstrap(P0, fitdata, fitra, fitdec, ErrorLstSq)
                    fitModel = Gauss2d
                elif mode == 'mode2':
                    P0 = [np.max(fitdata) -np.median(fitdata) ,
                          Pest[1],
                          Pest[2],
                          np.median(fitdata)]
                    fout = leastsq(ErrorLstSq2FWHM, P0, args=(fitra, fitdec, fitdata, x0,y0), full_output=True)
                    fbootmean, fbootstd = Bootstrap(P0, fitdata, fitra, fitdec, ErrorLstSq2FWHM)
                    fitModel = Gauss2d2FWHMFixed
                else:
                    print ('Warning: No fitting method selected')

                
                if isinstance(fout[1], type(None)):
                    continue
                
                #fout = np.mean(samples,axis=0), 
                P1[i,j,k,:] = fout[0]
                errors[i,j,k,:] = fbootstd
                #pyplot.plot(fitdata-fitModel(fout[0], fitra, fitdec, x0,y0), label='data')
                #pyplot.legend(loc='upper right')
                #pyplot.ylabel('T (K)')
                #pyplot.xlabel('Sample')
                #pyplot.text(0.05,0.9, r'$\chi^2$='+'{:.2f}'.format(chi2), transform=pyplot.gca().transAxes)
                #pyplot.title('Horn {:d} obs {}'.format(i+1, prefix))
                #pyplot.show()

            #pyplot.errorbar(np.arange(P1.shape[2]), P1[i,j,:,1]*60, yerr=errors[i,j,:,1]*60)
            #pyplot.show()

        cross = np.argmax(fitModel(np.median(P1[i,0,:,:],axis=0), x, y, x0,y0))
        print( cross, nHorns)
        crossings[i] = cross
    if justOffsets:
        return P1, errors, Pestout, Pstdout, crossings#  P1, P1e, chis
    else:
        return P1, errors, Pestout, Pstdout, crossings


def FitTODazel(tod, az, el, maz, mel, ra, dec, clon, clat, prefix='', doPlots=True):
    # Beam solid angle aken from James' report on wiki Optics
    nubeam = np.array([26., 33., 40.])
    srbeam = np.array([2.1842e-6, 1.6771e-6, 1.4828e-6])
    pmdl = interp1d(nubeam, srbeam) # straight interpolation

    nHorns = tod.shape[0]
    nSidebands = tod.shape[1]
    nChans = tod.shape[2]
    nSamps = tod.shape[3]
    nParams = 6

    # Crossing indices
    crossings = np.zeros(nHorns)

    splitTOD = (tod[:,:,:,:(nSamps//2) * 2:2] - tod[:,:,:,1:(nSamps//2)*2:2])
    rms = np.std(splitTOD,axis=3)/np.sqrt(2)

    t = np.arange(nSamps)
    P1 = np.zeros((nHorns, nSidebands, nChans, nParams+1))
    if doPlots:
        fig = pyplot.figure()
    # Rotate the ra/dec
    for i in range(nHorns):
        skyx, skyy = Pointing.Rotate(ra[i,:], dec[i,:], clon, clat, pang[i,:])
        x, y = Pointing.Rotate(az[i,:], el[i,:], maz, mel, 0)

        # +/- 180
        x[x > 180] -= 360
        skyx[skyx > 180] -= 360

        #pyplot.plot(x,y)
        #pyplot.figure()
        #pyplot.plot(az[i,:], el[i,:])
        #pyplot.show()

        #xbins = np.linspace(-4, 4, 60*8+1)
        #ybins = np.linspace(-4, 4, 60*8+1)

        # Just select the near data
        r = np.sqrt(x**2 + y**2)

        close = (r < 10./60.)
        near  = (r < 25./60.) & (r > 15/60.)
        far   = (r > 15./60.)
        fitselect = (r < 25./60.)

        for j in range(nSidebands):
            
            for k in range(nChans):

                fitdata = tod[i,j,k,fitselect]
                fitra = x[fitselect]
                fitdec= y[fitselect]

                amax = np.argmax(fitdata)

                P0 = [np.max(fitdata) -np.median(fitdata) ,
                      4./60./2.355,
                      4./60./2.355,
                      np.median(fitdata),
                      fitra[amax],
                      fitdec[amax]]

                P1[i,j,k,:nParams], s = leastsq(Error, P0, args=(fitra, fitdec, fitdata, 0,0))

                # Capture the RMS
                P1[i,j,k,nParams] = rms[i,j,k]
                #np.std(tod[i,j,k,fitselect]-Gauss2d(P1[i,j,k,:nParams], x[fitselect], y[fitselect], 0,0))

                if doPlots:
                    if (j == 0) & (k == 0):
                        ax = fig.add_subplot(2,1,1)
                        pyplot.plot(tod[i,j,k,fitselect]-Gauss2dNoGrad(P1[i,j,k,:nParams], x[fitselect], y[fitselect], 0,0))
                        pyplot.title('Horn {}, Sideband {}, Avg. Channel {} \n {}'.format(i,j,k, prefix))
                        pyplot.xlabel('Sample')
                        pyplot.ylabel('Detector Units')
                        ax = fig.add_subplot(2,1,2)
                        pyplot.plot(tod[i,j,k,fitselect])
                        pyplot.plot(Gauss2dNoGrad(P1[i,j,k,:nParams], x[fitselect], y[fitselect], 0,0))
                        pyplot.xlabel('Sample')
                        pyplot.ylabel('Detector Units')
                        pyplot.savefig('TODResidPlots/TODResidual_{}_H{}_S{}_C{}.png'.format(prefix, i,j,k), bbox_inches='tight')
                        pyplot.clf()
        crossings[i] = np.argmax(Gauss2dNoGrad(np.mean(P1[i,0,:,:nParams],axis=0), x, y, 0,0))

    return P1, crossings

import numpy as np
from scipy.optimize import leastsq, fmin
from matplotlib import pyplot
from scipy.interpolate import interp1d
import Pointing
from scipy.ndimage.filters import median_filter
from scipy.ndimage.filters import gaussian_filter,maximum_filter
from skimage.feature import peak_local_max


import emcee

import Mapping

naxis = [100, 100]
cdelt = [1./60., 1./60.]
crval =[0., 0.]

def ImagePeaks(image, rgrid, threshold):
    msmooth = [gaussian_filter(image, fsmooth) for fsmooth in [1,3]]
    dsmooth = msmooth[0] - msmooth[1]
    #dsmooth[rgrid > 4] = 0.
    #coords = peak_local_max(dsmooth, threshold_abs=threshold)
    return dsmooth


def Plane(P, x, y):
    
    return P[0]*(x) + P[1]*(y)  #+ P[2]*(x-P[4])**2 + P[3]*(y-P[5])**2

def Gauss2d(P, x, y, ra_c, dec_c):
    
    X = (x - ra_c - P[2])
    Y = (y - dec_c - P[3])
    #Xr =  np.cos(P[6]) * X + np.sin(P[6]) * Y
    #Yr = -np.sin(P[6]) * X + np.cos(P[6]) * Y

    #a = (Xr/P[1])**2
    #b = (Yr/P[2])**2
    a = (X/P[1])**2
    b = (Y/P[1])**2

    return P[0] * np.exp( - 0.5 * (a + b)) + P[4] + Plane([P[5], P[6]], x, y)


def stopGauss2d(P, x, y, ra_c, dec_c):
    
    X = (x - ra_c - P[4])
    Y = (y - dec_c - P[5])
    Xr =  np.cos(P[6]) * X + np.sin(P[6]) * Y
    Yr = -np.sin(P[6]) * X + np.cos(P[6]) * Y

    a = (Xr/P[1])**2
    b = (Yr/P[2])**2
    #a = (X/P[1])**2
    #b = (Y/P[1])**2

    return P[0] * np.exp( - 0.5 * (a + b)) + P[3] + Plane([P[7], P[8]], x, y)

def ErrorLstSq(P, x, y, z, ra_c, dec_c):

    if (P[1] > 6./60./2.355) | (P[1] < 0) | (P[0] < 0) | (np.sqrt(P[2]**2 + P[3]**2) > 60./60.): 
        return 0.*z + 1e32
    else:
        return z - Gauss2d(P, x, y, ra_c, dec_c)


def ErrorFmin(P, x, y, z, ra_c, dec_c):

    P[6] = np.mod(P[6], np.pi*2.)
    #if (P[1] > 6./60./2.355) | (P[1] < 0) | (P[2] > 1) | (P[2] < 0) | (P[0] < 0) | (np.sqrt(P[4]**2 + P[5]**2) > 8./60.): # | (np.abs(P[4]-ra_c) > 5./60.) | (np.abs(P[5]-dec_c) > 5./60.):
    if (P[1] > 6./60./2.355) | (P[1] < 0) | (P[0] < 0) | (np.sqrt(P[2]**2 + P[3]**2) > 60./60.): 
        return  1e32
    else:
        return np.sum((z - Gauss2d(P, x, y, ra_c, dec_c))**2)
        
        
# Now MCMC fitting?
def lnlike(P, x, y,z, yerr):
    model = Gauss2d(P, x, y, 0,0)
    inv_sigma2 = 1.0/(yerr**2)#  + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((z-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprior(P):
    A, sig, x0, y0, bkgd, dx, dy = P
    r = np.sqrt(x0**2 + y0**2)
    if 0 < A < 1e8 and 0.0 < sig < 10./60./2.355 and 0 < r < 1.0:
        return 0.0
    return -np.inf
    
def lnprob(P, x, y, z, yerr):
    lp = lnprior(P)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(P, x, y, z, yerr)


def fitJupiter(specTOD, ra, dec, ra_c = 0., dec_c = 0.):

    # FIT JUST THE MEAN SIDEBAND SIGNAL
    P0 = [np.max(fitdata) -np.median(fitdata) ,
          4./60.,
          4./60.,
          np.median(fitdata),
          ra_c,
          dec_c,
          0.]

    P1sb, s = leastsq(Error, P0, args=(fitra, fitdec, fitdata,ra_c, dec_c))
    
    ampsSB = P1sb[0]
    rmsSB  = np.std(fitdata-Gauss2d(P1sb, fitra, fitdec))
    

    fitselect = (r < 25./60.)
    t = np.arange(specTOD.shape[1])
    amps = np.zeros((len(P1sb), specTOD.shape[0]))
    errors = np.zeros((len(P1sb), specTOD.shape[0]))

    for i in range(specTOD.shape[0]):
        mSpec = specTOD[i,:]
        #mSpec = np.mean(specTOD[5:-5,:], axis=0)
        bkgd = np.poly1d(np.polyfit(t[near], mSpec[near], 5))

        fitdata = mSpec[fitselect] - bkgd(t[fitselect])
        fitra = ra[fitselect]
        fitdec= dec[fitselect]


        ra_c, dec_c = 0.03, 0
        P0 = P1sb*1.
        P0[0] = np.max(fitdata) -np.median(fitdata) 
        P0[3] = np.median(fitdata)

        P1, cov_x, info, mesg, s = leastsq(Error, P0, args=(fitra, fitdec, fitdata,ra_c, dec_c), full_output=True)

        amps[:,i] = P1
        if not isinstance(cov_x, type(None)):
            resid = np.std(fitdata -  Gauss2d(P1, fitra, fitdec, ra_c, dec_c))**2
            errors[:,i] = np.sqrt(np.diag(cov_x)*resid)
        else:
            errors[:,i] = 0.

    nHalf = (specTOD.shape[1]//2)*2
    rms = np.std(specTOD[:,1:nHalf:2] - specTOD[:,0:nHalf:2],axis=1)/np.sqrt(2)
    return amps, errors, rms, ampsSB, np.mean(rdist)


def FitTOD(tod, ra, dec, obs, clon, clat, prefix='', destripe=False):
    # Beam solid angle aken from James' report on wiki Optics
    nubeam = np.array([26., 33., 40.])
    srbeam = np.array([2.1842e-6, 1.6771e-6, 1.4828e-6])
    pmdl = interp1d(nubeam, srbeam) # straight interpolation

    nHorns = tod.shape[0]
    nSidebands = tod.shape[1]
    nChans = tod.shape[2]
    nSamps = tod.shape[3]
    nParams = 5 + 2


    wcs, xr, yr = Mapping.DefineWCS(naxis, cdelt, crval)

    # Crossing indices
    crossings = np.zeros(nHorns)


    # Calculate RMS from adjacent pairs
    splitTOD = (tod[:,:,:,:(nSamps//2) * 2:2] - tod[:,:,:,1:(nSamps//2)*2:2])
    rms = np.std(splitTOD,axis=3)/np.sqrt(2)

    t = np.arange(nSamps)
    P1 = np.zeros((nHorns, nSidebands, nChans, nParams+1))
    errors = np.zeros((nHorns, nSidebands, nChans, nParams))

    fig = pyplot.figure(figsize=(16,16))
    # Rotate the ra/dec
    width = 4.
    pixwidth = 2./60.
    nbins = int(width/pixwidth)
    xygrid = [np.linspace(-width/2, width/2,nbins+1), np.linspace(-width/2, width/2.,nbins+1)]
    xgrid, ygrid = np.meshgrid(np.linspace(-width/2, width/2,nbins)+pixwidth/2., np.linspace(-width/2, width/2.,nbins) + pixwidth/2.)
    for i in  range( nHorns):
        x, y = Pointing.Rotate(ra[i,:], dec[i,:], clon, clat, 0)
        
        # Bin the map up, apply a filter to remove spikes, to give first guess at pixel centre.
        hmap = np.histogram2d(y,x, xygrid)[0]
        todTemp = np.median(np.median(tod[i,:,:,:],axis=0),axis=0)
        rmsTemp = np.std(todTemp[1:todTemp.size//2 *2:2] - todTemp[:todTemp.size//2 * 2:2])/np.sqrt(2)
        smap = np.histogram2d(y,x, xygrid, weights = todTemp)[0]
        xpix = (x+width/2)//pixwidth 
        ypix = (y+width/2)//pixwidth
        #pyplot.plot(xpix, ypix, 'o')
        #pyplot.figure()
        #pyplot.plot(x, y, 'o')
        #pyplot.show()

        if destripe:
            pixels = (ypix + nbins*xpix).astype(int)
            offset = 300
            gd = (pixels > 0) & (pixels < nbins*nbins-1)
            m, offsets = Mapping.Destripe(todTemp[gd], pixels[gd], obs[gd], offset, nbins*nbins)
            m = np.reshape(m, (nbins, nbins)).T 
            #m = smap/hmap
        else:
            #h, wx, wy = np.histogram2d(x,y, (xygrid[0], xygrid[1]))
            #s, wx, wy = np.histogram2d(x,y, (xygrid[0], xygrid[1]), weights=todTemp)
            #m = s/h
            offsets  =0
            m, hits = Mapping.MakeMapSimple(todTemp, x, y, wcs)
            m = m/hits

        m -= np.nanmedian(m)
        #m /= rmsTemp
        m[np.isnan(m)] = 0.

        #pyplot.plot(ra[i,:], todTemp-offsets,'.')
        #pyplot.figure()
        #pyplot.plot(dec[i,:], todTemp-offsets,'.')
        #pyplot.show()


        r = np.sqrt((xgrid)**2 + (ygrid)**2)   
        
        d1 = ImagePeaks(m, r, 4)
        # if len(coords1) > 0:
        #     ximax, yimax = coords1[0]
        # else:
        #     m = median_filter(m, size=(2,2))
        # Remove background?
        xgrid2,ygrid2 = np.meshgrid(np.linspace(-d1.shape[0]/2, d1.shape[0]/2, d1.shape[0]), np.linspace(-d1.shape[1]/2, d1.shape[1]/2, d1.shape[1]))
        background = np.sqrt((xgrid2 - 0)**2 + (ygrid2 - 0)**2) > 25
        print(d1.shape, xgrid.shape)
        d1[background] = 0
        
        ximax, yimax = np.unravel_index(np.argmax(d1), m.shape)
        print(ximax, yimax, np.max(d1))
        #pyplot.imshow(m)
        #pyplot.plot(yimax, ximax,'ow')
        #pyplot.figure()
        #pyplot.imshow(d1)
        #pyplot.plot(yimax, ximax,'ow')
        #pyplot.show()

        # +/- 180
        x[x > 180] -= 360

        #pyplot.plot(x,y)
        #pyplot.figure()
        #pyplot.plot(az[i,:], el[i,:])
        #pyplot.show()

        #xbins = np.linspace(-4, 4, 60*8+1)
        #ybins = np.linspace(-4, 4, 60*8+1)

        # Just select the near data
        y0,x0 = xr[ximax,yimax], yr[ximax,yimax]
        
        print(x0, y0)
        
        #print(x0, y0)
        #pyplot.figure(figsize=(6,6))
        #pyplot.plot(yr.flatten(),m.flatten(),'-')
        #pyplot.axvline(y0,color='r')
        #pyplot.show()


        background = np.sqrt((x - x0)**2 + (y - y0)**2) > 30./60.
        
        

        #x0, y0 = 0, 0
        #pyplot.plot(x,tod[0,0,0,:])
        #pyplot.axvline(x0)
        #pyplot.axvline(x[np.argmax(tod[0,0,0,:])],color='r')
        #pyplot.figure()
        #pyplot.plot(tod[0,0,0,:])
        #pyplot.show()
        r = np.sqrt((x-x0)**2 + (y-y0)**2)
        close = (r < 10./60.)
        near  = (r < 25./60.) & (r > 15/60.)
        far   = (r > 15./60.)
        fitselect = (r < 20./60.)
        time = np.arange(tod.shape[-1])
        #pyplot.plot(time[:], tod[0,0,0,:])
        #pyplot.plot(time[fitselect], tod[0,0,0,fitselect])
        #pyplot.show()
        #extent = [-naxis[0]/2. * cdelt[0], naxis[0]/2. * cdelt[0], 
        #          -naxis[1]/2. * cdelt[1], naxis[1]/2. * cdelt[1]]

        #pyplot.figure(figsize=(6,6))
        
        #m2 = m*0
        #r2 = np.sqrt((xr-x0)**2 + (yr-y0)**2)
        
        #m2 = (r2 < 10./60.)
        #pyplot.imshow(m.T,extent=extent,origin='lower')
        #pyplot.scatter(x0,y0, marker='o',color='r')
        #pyplot.figure()
        #pyplot.imshow(m2.T,extent=extent,origin='lower')
        #pyplot.scatter(x0,y0, marker='o',color='r')
        #pyplot.show()
        

        #pyplot.plot(time[:],todTemp[:],'.')
        #pyplot.plot(time[fitselect],todTemp[fitselect],'.')
        #pyplot.show()

        plotselect = (r < 120./60.)
        for j in range(nSidebands):
            
            for k in range(nChans):
                
                rmdl = np.poly1d(np.polyfit(time[background], tod[i,j,k,background]-offsets,11))
                tod[i,j,k,:] -= (rmdl(time[:]))

                #dmdl = np.poly1d(np.polyfit(dec[i,background], tod[i,j,k,background]-offsets,2))
                #tod[i,j,k,:] -= ( dmdl(dec[i,:]))

                if destripe:
                    m, offsets = Mapping.Destripe(tod[i,j,k,gd], pixels[gd], obs[gd], offset, nbins*nbins)
                    m = np.reshape(m, (nbins,nbins)).T
                else:
                    #h, wx, wy = np.histogram2d(x[:],y[:], (xygrid[0], xygrid[1]))
                    #s, wx, wy = np.histogram2d(x[:],y[:], (xygrid[0], xygrid[1]), weights=tod[i,j,k,:])
                    #m = s/h
                    m, hits = Mapping.MakeMapSimple(tod[i,j,k,:], x, y, wcs)
                    m = m/hits


                #pyplot.plot(ra[i,:], tod[i,j,k,:]-offsets,'.')
                #pyplot.figure()
                #pyplot.plot(dec[i,:], tod[i,j,k,:]-offsets,'.')
                #pyplot.show()

                #pyplot.plot(tod[i,j,k,:])
                #pyplot.plot(offsets)
                #pyplot.figure()
                #pyplot.plot(tod[i,j,k,:]-offsets)
                #pyplot.plot()
                #pyplot.show()


                tod[i,j,k,:] = tod[i,j,k,:] -offsets #-(rmdl(ra[i,:])+dmdl(dec[i,:]))
                tod[i,j,k,:] -= np.median(tod[i,j,k,:])
                fitdata = tod[i,j,k,fitselect]
                fitra = x[fitselect]
                fitdec= y[fitselect]

                #pyplot.plot(y, tod[i,j,k,:],'-')
                #pyplot.plot(fitdec, fitdata,'-')
                #pyplot.axvline(y0,color='k')
                #pyplot.figure()                
                #pyplot.plot(x, tod[i,j,k,:],'-')
                #pyplot.plot(fitra, fitdata,'-')
                #pyplot.axvline(x0,color='k')
                #pyplot.show()                

                amax = np.argmax(fitdata)

                P0 = [np.max(fitdata) -np.median(fitdata) ,
                      4./60./2.355,
                      4./60./2.355,
                      np.median(fitdata),
                      x0,
                      y0,
                      0.,
                      0., 0.]#,0.,0., 0., 0.]
                P0 = [np.max(fitdata) -np.median(fitdata) ,
                      4./60./2.355,
                      x0,
                      y0,
                      np.median(fitdata),
                      0., 0.]#,0.,0., 0., 0.]


                #P1[i,j,k,:nParams], cov_x, info, mesg, s = leastsq(Error, P0, args=(fitra, fitdec, fitdata, 0,0), full_output=True)
                #fout = fmin(ErrorFmin, P0, maxfun=3000, args=(fitra, fitdec, fitdata, 0,0), full_output=True)
                #fout = leastsq(ErrorLstSq, P0, args=(fitra, fitdec, fitdata, 0,0), full_output=True)
                
                ndim, nwalkers = len(P0), 100
                pos = [np.array(P0) + 1e-4*np.random.randn(ndim) for iwalker in range(nwalkers)]
                sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(fitra, fitdec, fitdata, rms[0,0,i]))
                sampler.run_mcmc(pos, 1200)
                samples = sampler.chain[:, 500:sampler.chain.shape[1]:3, :].reshape((-1, ndim))
                
                fout = np.mean(samples,axis=0), 
                #import corner
                #pyplot.clf()
                #pyplot.hist(samples[:,0], 50)
                #pyplot.show()
                #pyplot.plot(sampler.chain[0,:,0])
                #pyplot.show()
                #fig = corner.corner(samples)
                #pyplot.show()
                P1[i,j,k,:nParams] = fout[0]
                print(fout[0])
                print (P0)
                print(x0, y0)
                import corner
                pyplot.clf()
                fig = corner.corner(samples)
                pyplot.show()
                
                #pyplot.plot(time[fitselect], fitdata)
                #pyplot.show()

                #pyplot.plot(fitdata)
                #pyplot.plot(Gauss2d(P0, x[fitselect], y[fitselect], 0,0))
                #pyplot.show()
                #pyplot.plot(fitdata)
                #pyplot.plot(Gauss2d(P1[i,j,k,:nParams], fitra, fitdec, 0,0))
                #pyplot.show()


                # Capture the RMS
                #r2 = np.sqrt((x-P1[i,j,k,4])**2 + (y-P1[i,j,k,5])**2)
                #plotselect = (r2 < 4./60.)

                P2 = P1[i,j,k,:nParams]*1.
                P2[0] = 0.

                ntod = Gauss2d(P1[i,j,k,:nParams], x, y, 0,0)
                nulltod = Gauss2d(P2, x, y, 0.,0.)
                otod = tod[i,j,k,:]
                omaps, hits = Mapping.MakeMapSimple(otod, x, y, wcs)
                nmaps, hits = Mapping.MakeMapSimple(ntod, x, y, wcs)
                nulls, hits = Mapping.MakeMapSimple(nulltod, x, y, wcs)


                xgrid, ygrid = np.meshgrid(np.arange(naxis[0]), np.arange(naxis[1]))
                xgrid = (xgrid-naxis[0]/2.)
                ygrid = (ygrid-naxis[1]/2.)
                rgrid = np.sqrt(xgrid**2 + ygrid**2)
                getchi = (rgrid < 5)
                nearChi = (rgrid > 7.5) & (rgrid < 15)

                maprms = np.nanstd((nmaps[nearChi]-omaps[nearChi])/hits[nearChi])
                rmap = (omaps[getchi]-nmaps[getchi])/hits[getchi]
                mapChi = np.nansum((rmap-np.mean(rmap))**2/maprms**2)/(nParams-1.)
                mapOrig = np.nansum(((omaps[getchi]-nulls[getchi])/hits[getchi])**2/maprms**2)/(nParams-1.)

                
                origChi = np.sum((tod[i,j,k,plotselect] -  Gauss2d(P2, x[plotselect], y[plotselect], 0.,0.))**2/rms[i,j,k]**2)/(nParams-1.)
                reducedChi = np.sum((tod[i,j,k,plotselect] -  Gauss2d(P1[i,j,k,:nParams], x[plotselect], y[plotselect], 0.,0.))**2/rms[i,j,k]**2)/(nParams-1.)
                #print('CHI2', origChi, reducedChi, mapChi, mapOrig, reducedChi/origChi)
                P1[i,j,k,nParams] = rms[i,j,k]


                if (k >= 0):
                    #r2 = np.sqrt((x-P1[i,j,k,2])**2 + (y-P1[i,j,k,3])**2)
                    #plotselect = (r2 < 15./60.)

                    ax = fig.add_subplot(3,1,1)
                    todPlot = tod[i,j,k,plotselect]
                    todPlot -= np.median(todPlot)
                    pyplot.plot(todPlot-Gauss2d(P1[i,j,k,:nParams], x[plotselect], y[plotselect], 0,0))
                    pyplot.title('Horn {}, Sideband {}, Avg. Channel {} \n {}'.format(i,j,k, prefix))
                    pyplot.xlabel('Sample')
                    pyplot.ylabel('Detector Units')
                    pyplot.text(0.9,0.9,r'$rms$ = '+'{:.3f}'.format(rms[i,j,k]), ha='right', transform=ax.transAxes) 
                    ax = fig.add_subplot(3,1,2)
                    pyplot.plot(todPlot)

                    P3 = P1[i,j,k,:nParams]*1.
                    P3[3] = 0
                    P3[7:8] = 0
                    p3Model = Gauss2d(P3, x[plotselect], y[plotselect], 0,0)
                    pyplot.plot(Gauss2d(P1[i,j,k,:nParams], x[plotselect], y[plotselect], 0,0))#p3Model - np.nanmedian(p3Model))
                    pyplot.plot(Gauss2d(P2, x[plotselect], y[plotselect], 0,0))

                    pyplot.xlabel('Sample')
                    pyplot.ylabel('Detector Units')
                    pyplot.text(0.9,0.9,r'$\chi^2$ = '+'{:.3f}'.format(mapChi), ha='right', transform=ax.transAxes) 
                    
                    
                    extent = [-naxis[0]/2. * cdelt[0], naxis[0]/2. * cdelt[0], 
                              -naxis[1]/2. * cdelt[1], naxis[1]/2. * cdelt[1]]
                    ax = fig.add_subplot(3,3,7)
                    ax.imshow(m, aspect='auto', extent=extent)
                    ax.scatter(P1[i,j,k,2], P1[i,j,k,3], marker='o',color='r')
                    ax = fig.add_subplot(3,3,8)
                    ax.imshow(nmaps/hits, extent=extent)
                    ax.scatter(P1[i,j,k,2], P1[i,j,k,3], marker='o',color='r')

                    ax = fig.add_subplot(3,3,9)
                    ax.imshow(nmaps/hits-m, extent=extent)
                    ax.scatter(P1[i,j,k,2], P1[i,j,k,3], marker='o',color='r')

                    pyplot.tight_layout(True)
                    pyplot.savefig('TODResidPlots/TODResidual_{}_H{}_S{}_C{}.png'.format(prefix, i,j,k), bbox_inches='tight')
                    pyplot.clf()
        cross = np.argmax(Gauss2d(np.median(P1[i,0,:,:nParams],axis=0), x, y, 0,0))
        print( cross, nHorns)
        crossings[i] = cross
    return P1, errors, crossings
                #xhist = np.histogram(x[far], xbins, weights=tod[i,j,k,far])[0]/np.histogram(x[far], xbins)[0]
                #yhist = np.histogram(y[far], ybins, weights=tod[i,j,k,far])[0]/np.histogram(y[far], ybins)[0]

                #bkgd = np.poly1d(np.polyfit(y[far], tod[i,j,k,far], 9))
                #pyplot.plot(y, tod[i,j,k,:])
                #pyplot.plot((ybins[1:]+ybins[:-1])/2., yhist)
                #pyplot.plot(y, bkgd(y))

                #pyplot.show()
                #tod[i,j,k,:] -= bkgd(t)


    #     # FIT JUST THE MEAN SIDEBAND SIGNAL
    # fitselect = (r < 25./60.)
    # goodChans = range(60,440) + range(560,940)
    # mSpec =  np.mean(specTOD[goodChans,:], axis=0)
    # bkgd = np.poly1d(np.polyfit(t[far], mSpec[far], 9))
    # mSpec -= bkgd(t)
    
    # bkgd = np.poly1d(np.polyfit(t[near], mSpec[near], 9))
    
    # fitdata = mSpec[fitselect]- bkgd(t[fitselect])
    # fitra = ra[fitselect]
    # fitdec= dec[fitselect]

                
    #             amps, rms, ampsSB, rdist = fitJupiter(tod[i,j,k,:], x, y)

def FitTODazel(tod, az, el, maz, mel, ra, dec, clon, clat, prefix=''):
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
    fig = pyplot.figure()
    # Rotate the ra/dec
    for i in range(nHorns):
        skyx, skyy = Pointing.Rotate(ra[i,:], dec[i,:], clon, clat, 0)
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

                if (j == 0) & (k == 0):
                    ax = fig.add_subplot(2,1,1)
                    pyplot.plot(tod[i,j,k,fitselect]-Gauss2d(P1[i,j,k,:nParams], x[fitselect], y[fitselect], 0,0))
                    pyplot.title('Horn {}, Sideband {}, Avg. Channel {} \n {}'.format(i,j,k, prefix))
                    pyplot.xlabel('Sample')
                    pyplot.ylabel('Detector Units')
                    ax = fig.add_subplot(2,1,2)
                    pyplot.plot(tod[i,j,k,fitselect])
                    pyplot.plot(Gauss2d(P1[i,j,k,:nParams], x[fitselect], y[fitselect], 0,0))
                    pyplot.xlabel('Sample')
                    pyplot.ylabel('Detector Units')
                    pyplot.savefig('TODResidPlots/TODResidual_{}_H{}_S{}_C{}.png'.format(prefix, i,j,k), bbox_inches='tight')
                    pyplot.clf()
        crossings[i] = np.argmax(Gauss2d(np.mean(P1[i,0,:,:nParams],axis=0), x, y, 0,0))

    return P1, crossings

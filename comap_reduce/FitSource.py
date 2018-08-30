import numpy as np
from scipy.optimize import leastsq
from matplotlib import pyplot
from scipy.interpolate import interp1d
import Pointing
from scipy.ndimage.filters import median_filter
from scipy.ndimage.filters import gaussian_filter,maximum_filter
from skimage.feature import peak_local_max

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
    
    X = (x - ra_c - P[4])
    Y = (y - dec_c - P[5])
    #Xr =  np.cos(P[6]) * X + np.sin(P[6]) * Y
    #Yr = -np.sin(P[6]) * X + np.cos(P[6]) * Y

    #a = (Xr/P[1])**2
    #b = (Yr/P[2])**2
    a = (X/P[1])**2
    b = (Y/P[1])**2

    return P[0] * np.exp( - 0.5 * (a + b)) + P[3] + Plane([P[7], P[8]], x, y)



def Error(P, x, y, z, ra_c, dec_c):

    P[6] = np.mod(P[6], np.pi*2.)
    if (P[1] > 6./60./2.355) | (P[1] < 0) | (P[2] > 1) | (P[2] < 0) | (P[0] < 0) | (np.sqrt(P[4]**2 + P[5]**2) > 3./60.): # | (np.abs(P[4]-ra_c) > 5./60.) | (np.abs(P[5]-dec_c) > 5./60.):
        return 0.*z + 1e32
    else:
        return z - Gauss2d(P, x, y, ra_c, dec_c)



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


def FitTOD(tod, ra, dec, obs, clon, clat, prefix=''):
    # Beam solid angle aken from James' report on wiki Optics
    nubeam = np.array([26., 33., 40.])
    srbeam = np.array([2.1842e-6, 1.6771e-6, 1.4828e-6])
    pmdl = interp1d(nubeam, srbeam) # straight interpolation

    nHorns = tod.shape[0]
    nSidebands = tod.shape[1]
    nChans = tod.shape[2]
    nSamps = tod.shape[3]
    nParams = 7 + 2

    # Crossing indices
    crossings = np.zeros(nHorns)

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

        pixels = (ypix + nbins*xpix).astype(int)
        offset = 300
        gd = (pixels > 0) & (pixels < nbins*nbins-1)
        m, offsets = Mapping.Destripe(todTemp[gd], pixels[gd], obs[gd], offset, nbins*nbins)
        m = np.reshape(m, (nbins, nbins)).T 
        #m = smap/hmap
        m -= np.nanmedian(m)
        m /= rmsTemp
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
        ximax, yimax = np.unravel_index(np.argmax(d1), m.shape)

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
        x0, y0 = xygrid[0][yimax], xygrid[1][ximax]
        x0, y0 = 0, 0
        r = np.sqrt((x-x0)**2 + (y-y0)**2)

        close = (r < 10./60.)
        near  = (r < 25./60.) & (r > 15/60.)
        far   = (r > 15./60.)
        fitselect = (r < 25./60.)

        r = np.sqrt((x-x0)**2 + (y-y0)**2)
        plotselect = (r < 45./60.)

        for j in range(nSidebands):
            
            for k in range(nChans):

                m, offsets = Mapping.Destripe(tod[i,j,k,gd], pixels[gd], obs[gd], offset, nbins*nbins)
                m = np.reshape(m, (nbins,nbins)).T

                rmdl = np.poly1d(np.polyfit(ra[i,:], tod[i,j,k,:]-offsets,1))
                dmdl = np.poly1d(np.polyfit(dec[i,:], tod[i,j,k,:]-offsets,1))

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


                tod[i,j,k,:] = tod[i,j,k,:] -offsets -(rmdl(ra[i,:])+dmdl(dec[i,:]))
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

                P1[i,j,k,:nParams], cov_x, info, mesg, s = leastsq(Error, P0, args=(fitra, fitdec, fitdata, 0,0), full_output=True)


                #pyplot.plot(fitdata)
                #pyplot.plot(Gauss2d(P1[i,j,k,:nParams], fitra, fitdec, 0,0))
                #pyplot.show()


                # Capture the RMS
                r2 = np.sqrt((x-P1[i,j,k,4])**2 + (y-P1[i,j,k,5])**2)
                plotselect = (r2 < 4./60.)

                P2 = P1[i,j,k,:nParams]*1.
                P2[0] = 0.

                ntod = Gauss2d(P1[i,j,k,:nParams], x, y, 0,0)
                nulltod = Gauss2d(P2, x, y, 0.,0.)
                otod = tod[i,j,k,:]
                wcs = Mapping.DefineWCS(naxis, cdelt, crval)
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
                #print(maprms)

                #print(mapChi, np.nansum((rmap-np.mean(rmap))**2) ,maprms**2)
                #rmap = (omaps-nmaps)/hits
                #omaps /= hits
                #omaps /= np.nanstd(omaps)
                #pyplot.imshow(omaps)
                #omaps[getchi] = 116
                #omaps[nearChi] = 115
                #pyplot.figure()
                #pyplot.imshow(omaps)
                #pyplot.figure()
                #pyplot.imshow(rmap)
                #pyplot.show()

                
                origChi = np.sum((tod[i,j,k,plotselect] -  Gauss2d(P2, x[plotselect], y[plotselect], 0.,0.))**2/rms[i,j,k]**2)/(nParams-1.)
                reducedChi = np.sum((tod[i,j,k,plotselect] -  Gauss2d(P1[i,j,k,:nParams], x[plotselect], y[plotselect], 0.,0.))**2/rms[i,j,k]**2)/(nParams-1.)
                #print('CHI2', origChi, reducedChi, mapChi, mapOrig, reducedChi/origChi)
                P1[i,j,k,nParams] = rms[i,j,k]
                if not isinstance(cov_x, type(None)):
                    resid = np.std(fitdata -  Gauss2d(P1[i,j,k,:nParams], fitra, fitdec, 0.,0.))**2
                    errors[i,j,k,:] = np.sqrt(np.diag(cov_x)*resid)
                else:
                    errors[i,j,k,:] = 0.


                #np.std(tod[i,j,k,fitselect]-Gauss2d(P1[i,j,k,:nParams], x[fitselect], y[fitselect], 0,0))

                
                #pyplot.imshow(m, extent=[-2,2,-2,2], origin='lower')
                #residual = tod[i,j,k,:]-Gauss2d(P1[i,j,k,:nParams], x, y, 0,0)
                #ps = np.abs(np.fft.fft(residual))**2
                #psnu = np.fft.fftfreq(residual.size, d=1./20.)
                #pyplot.plot(psnu[1:ps.size//2], ps[1:ps.size//2])
                #pyplot.yscale('log')
                #pyplot.xscale('log')
                #pyplot.show()


                if (k >= 0):
                    r2 = np.sqrt((x-P1[i,j,k,4])**2 + (y-P1[i,j,k,5])**2)
                    plotselect = (r2 < 4./60.)

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
                    pyplot.plot(Gauss2d(P1[i,j,k,:nParams], x[plotselect], y[plotselect], 0,0))
                    pyplot.plot(Gauss2d(P2, x[plotselect], y[plotselect], 0,0))

                    pyplot.xlabel('Sample')
                    pyplot.ylabel('Detector Units')
                    pyplot.text(0.9,0.9,r'$\chi^2$ = '+'{:.3f}'.format(mapChi), ha='right', transform=ax.transAxes) 
                    
                    
                    extent = [-naxis[0]/2. * cdelt[0], naxis[0]/2. * cdelt[0], 
                              -naxis[1]/2. * cdelt[1], naxis[1]/2. * cdelt[1]]
                    ax = fig.add_subplot(3,3,7)
                    ax.imshow(m, aspect='auto', extent=extent)
                    ax.scatter(P1[i,j,k,4], P1[i,j,k,5], marker='o',color='r')
                    ax = fig.add_subplot(3,3,8)
                    ax.imshow(nmaps/hits, extent=extent)
                    ax.scatter(P1[i,j,k,4], P1[i,j,k,5], marker='o',color='r')

                    ax = fig.add_subplot(3,3,9)
                    #ax.imshow(nmaps/hits-m, extent=extent)
                    ax.scatter(P1[i,j,k,4], P1[i,j,k,5], marker='o',color='r')

                    pyplot.tight_layout(True)
                    pyplot.savefig('TODResidPlots/TODResidual_{}_H{}_S{}_C{}.png'.format(prefix, i,j,k), bbox_inches='tight')
                    pyplot.clf()
        cross = np.argmax(Gauss2d(np.median(P1[i,0,:,:nParams],axis=0), x, y, 0,0))
        print cross, nHorns
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

import numpy as np
from scipy.optimize import leastsq
from matplotlib import pyplot
from scipy.interpolate import interp1d
import Pointing
from scipy.ndimage.filters import median_filter

def Gauss2d(P, x, y, ra_c, dec_c):
    
    a = ((x - ra_c - P[4])/P[1])**2
    b = ((y - dec_c - P[5])/P[2])**2

    return P[0] * np.exp( - 0.5 * (a + b)) + P[3]



def Error(P, x, y, z, ra_c, dec_c):

    if (P[1] > 1) | (P[1] < 0) | (P[2] > 1) | (P[2] < 0) | (P[0] < 0): # | (np.abs(P[4]-ra_c) > 5./60.) | (np.abs(P[5]-dec_c) > 5./60.):
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
          dec_c]

    P1sb, s = leastsq(Error, P0, args=(fitra, fitdec, fitdata,ra_c, dec_c))
    
    ampsSB = P1sb[0]
    rmsSB  = np.std(fitdata-Gauss2d(P1sb, fitra, fitdec))
    

    fitselect = (r < 25./60.)
    t = np.arange(specTOD.shape[1])
    amps = np.zeros((len(P1sb), specTOD.shape[0]))
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

        P1, s = leastsq(Error, P0, args=(fitra, fitdec, fitdata,ra_c, dec_c))

        amps[:,i] = P1

    nHalf = (specTOD.shape[1]//2)*2
    rms = np.std(specTOD[:,1:nHalf:2] - specTOD[:,0:nHalf:2],axis=1)/np.sqrt(2)
    return amps, rms, ampsSB, np.mean(rdist)


def FitTOD(tod, ra, dec, clon, clat, prefix=''):
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
    xygrid = [np.linspace(-2,2,int(4./(2./60.))+1), np.linspace(-2,2,int(4./(2./60.))+1) ]
    for i in range( nHorns):
        x, y = Pointing.Rotate(ra[i,:], dec[i,:], clon, clat, 0)

        # Bin the map up, apply a filter to remove spikes, to give first guess at pixel centre.
        hmap = np.histogram2d(y,x, xygrid)[0]
        smap = np.histogram2d(y,x, xygrid, weights = np.median(np.median(tod[i,:,:,:],axis=0),axis=0) )[0]
        m = smap/hmap
        m[np.isnan(m)] = 0.
        m = median_filter(m, size=(2,2))
        ximax, yimax = np.unravel_index(np.argmax(m), m.shape)

        
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
        r = np.sqrt((x-x0)**2 + (y-y0)**2)

        close = (r < 10./60.)
        near  = (r < 25./60.) & (r > 15/60.)
        far   = (r > 15./60.)
        fitselect = (r < 25./60.)

        for j in range(nSidebands):
            
            for k in range(nChans):

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
                      y0]

                P1[i,j,k,:nParams], s = leastsq(Error, P0, args=(fitra, fitdec, fitdata, 0,0))

                # Capture the RMS
                P1[i,j,k,nParams] = rms[i,j,k]
                #np.std(tod[i,j,k,fitselect]-Gauss2d(P1[i,j,k,:nParams], x[fitselect], y[fitselect], 0,0))

                
                #pyplot.imshow(m, extent=[-2,2,-2,2], origin='lower')
                #pyplot.plot(P1[i,j,k,4], P1[i,j,k,5],'or')
                #pyplot.show()


                if (k == 0):
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
        cross = np.argmax(Gauss2d(np.median(P1[i,0,:,:nParams],axis=0), x, y, 0,0))
        print cross, nHorns
        crossings[i] = cross
    return P1, crossings
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

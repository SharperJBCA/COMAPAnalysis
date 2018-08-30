import numpy as np
from pipeline.Tools import FileTools
from matplotlib import pyplot
import sys
import CartPix

from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.filters import maximum_filter, gaussian_filter, median_filter




def DefineWCS(naxis=[100,100], cdelt=[1./60., 1./60.],
              crval=[0,0]):

    wcs = CartPix.Info2WCS(naxis, cdelt, crval)

    return wcs

def RaDecGrid(naxis, cdelt, crval):
    
    rmin = crval[0] - naxis[0]*cdelt[0]/2.
    rmax = crval[0] + naxis[0]*cdelt[0]/2.
    dmin = crval[1] - naxis[1]*cdelt[1]/2.
    dmax = crval[1] + naxis[1]*cdelt[1]/2.
    ra = np.linspace(rmin, rmax, naxis[0])
    dec= np.linspace(dmin, dmax, naxis[1])

    xpix, ypix = np.meshgrid(np.arange(naxis[0]), np.arange(naxis[1]))
    dec, ra = CartPix.pix2ang(naxis, cdelt, crval,  xpix, ypix)
    return ra, dec

def AperPhot(m, ra, dec, r0, d0, rinner, rinner2, router):

    d2 = (dec - d0)

    r2 = (ra - r0)/np.cos(d2*np.pi/180.)
    rInside = (r2**2 + d2**2 < rinner**2)
    rOutside= ((r2**2 + d2**2) < router**2) & ((r2**2 + d2**2) > rinner2**2)

    #pyplot.plot(ra.flatten(), dec.flatten(),'.')
    #pyplot.plot((ra[rInside].flatten()),(dec[rInside].flatten()),'.')
    #pyplot.plot([r0], [d0], 'o')
    #pyplot.show()

    #m[rInside] = np.nan
    #pyplot.imshow(m)
    #pyplot.show()

    nPix = m[rInside].size
    fluxIn = np.nanmean(m[rInside])
    fluxOut = np.nanmedian(m[rOutside])

    #print(m[rInside])
    #print(fluxIn)
    #stop

    return fluxIn - fluxOut, np.nanstd(m[rOutside])

from scipy.optimize import leastsq
def Gauss2D(P,x,y,r0,d0):
    B = ((y - d0 - P[2])/P[3])**2

    xr = (x- r0 - P[0])/np.cos(y*np.pi/180.)
    A = (xr/P[1])**2

    return P[4]*np.exp(-0.5 * (A + B)) + P[5]

def error(P, x, y, z, r0, d0):
    return z - Gauss2D(P, x, y, r0, d0)

def AperFit(m, ra, dec, r0, d0, rnear):

    d2 = (dec - d0)

    r2 = (ra - r0)/np.cos(d2*np.pi/180.)
    rNear = (r2**2 + d2**2 < rnear**2) & (np.isnan(m) == False)

    P0 = [0., 3./60., 0., 3./60., np.nanmax(m), np.nanmedian(m)]

    P1, s = leastsq( error, P0,args=(ra[rNear], dec[rNear], m[rNear], r0, d0 ))

    # print(P1)
    # fig = pyplot.figure()
    # ax = fig.add_subplot(2,2,1)
    # pyplot.imshow(m)
    # ax = fig.add_subplot(2,2,2)
    # pyplot.imshow(Gauss2D(P1, ra, dec,r0,d0))
    # ax = fig.add_subplot(2,2,3)
    # pyplot.imshow(m-Gauss2D(P1, ra, dec,r0,d0))

    # pyplot.show()

    P1 = np.concatenate((P1, [np.nanstd(m-Gauss2D(P1,ra,dec,r0,d0))]))
    return P1

def FindPeaks(img):

    ngbhd = generate_binary_structure(2,2)
    #ngbhd
    print(np.max(img))
    img[np.isnan(img)] = 0.
    img_s = median_filter(img, footprint=np.ones((2,2))) # filter any single pixel spikes
    img_s = gaussian_filter(img, sigma=4) # smooth out the noise

    xpix, ypix = np.unravel_index(np.argmax(img_s), img_s.shape) # find peak pixel
    #pyplot.imshow(local_max)
    #pyplot.figure()
    #pyplot.imshow(img_s)
    #pyplot.show()
    
    return xpix, ypix

def MeasureAmps(filename):
    d = FileTools.ReadH5Py(filename)

    naxis = d['naxis']
    cdelt = d['cdelt']
    crval = d['crval']
    mAzEl = d['mAzEl']

    print(crval)
    wcs = DefineWCS(naxis, cdelt, crval)
    ra, dec = RaDecGrid(naxis, cdelt, crval)
    if crval[0] == 0:
        ra[ra > 180] -= 360.

    w = d['maps']
    h = d['hits']
    A = d['atmos']
    nhorns = w.shape[0]
    nsbs =  w.shape[1]
    nfreqs = w.shape[2]

    amps = np.zeros((nhorns, nsbs*nfreqs))
    stds = np.zeros((nhorns, nsbs*nfreqs))
    fits = np.zeros((nhorns, nsbs*nfreqs, 7))
    tfits = np.zeros((nhorns, nsbs, 7))

    amaxs = np.zeros((nhorns, nsbs*nfreqs,2))

    for i in range(nhorns):
        l = 0
        for j in range(nsbs):
            m = w[i,j,:,...]/h[i,...]
            xpix, ypix =  FindPeaks(np.nanmean(m,axis=0)) # first perform quick estimate of peak location per sideband

            print ra[xpix,ypix], dec[xpix,ypix]
            
            for k in range(nfreqs):
                try:
                    fits[i,l,:] = AperFit(m[k], ra, dec,ra[xpix,ypix], dec[xpix,ypix], 10./60.)
                except TypeError:
                    fits[i,l,:] = np.nan

                
                amps[i,l], stds[i,l] = AperPhot(m[k], ra, dec, 
                                     crval[0], crval[1], 
                                     6./60., 8./60., 15./60.)
                l += 1
            if (j == 0) | (j == 2):
                amps[i,j*nfreqs:(j+1)*nfreqs] = (amps[i,j*nfreqs:(j+1)*nfreqs:1])[::-1]
                stds[i,j*nfreqs:(j+1)*nfreqs] = (stds[i,j*nfreqs:(j+1)*nfreqs:1])[::-1]
                fits[i,j*nfreqs:(j+1)*nfreqs,:] = (fits[i,j*nfreqs:(j+1)*nfreqs:1, :])[::-1,:]

    
    # Then plot the images
    for i in range(nhorns):
        fig = pyplot.figure(figsize=(15,10))
        for j in range(nsbs):
            m = w[i,j,:,...]/h[i,...]
            xpix, ypix =  FindPeaks(np.nanmean(m,axis=0)) # first perform quick estimate of peak location per sideband

            # weighted mean by amplitude of jupiter e.g., the gain
            if (j == 0) | (j == 2):
                signal = amps[0,(j+1)*nfreqs:j*nfreqs:-1]
                noise = stds[0,(j+1)*nfreqs:j*nfreqs:-1]
                weight = (signal/noise)**2
                amax =  np.nansum(signal*weight)/np.nansum( weight)
                nmax = np.nansum(noise*weight)/np.nansum( weight)

            else:
                signal = amps[0,j*nfreqs:(j+1)*nfreqs:1]
                noise = stds[0,j*nfreqs:(j+1)*nfreqs:1]
                weight = (signal/noise)**2
                weight[np.isnan(weight)] = 0.
                amax = np.nansum(signal*weight)/np.nansum( weight)
                nmax = np.nansum(noise*weight)/np.nansum( weight)


            #print(np.sum(weight))
            #pyplot.plot(weight)
            #pyplot.show()
            m = np.nanmean(m,axis=0)
            #m = np.nansum(m*weight[:,np.newaxis,np.newaxis], axis=0)/np.nansum(np.ones(m.shape) * weight[:,np.newaxis,np.newaxis],axis=0)
            if np.nansum(m) == 0:
                print('Horn {} SB {} has no data'.format(i, j))
                continue
            tfits[i,j,:] = AperFit(m, ra, dec,ra[xpix,ypix], dec[xpix,ypix], 10./60.)


            ax = fig.add_subplot(2,4,2*j+1, projection=wcs)
            pyplot.imshow(m,origin='lower', vmax=amax+nmax*3, vmin=-nmax*3)
            pyplot.plot([crval[0]], [crval[1]],'kx', transform=ax.get_transform('world'), alpha=0.7)
            pyplot.title('SB {}'.format(j))
            if crval[0] == 0:
                pyplot.xlabel(r'$\Delta \alpha$')
                pyplot.ylabel(r'$\Delta \delta$')
            else:
                pyplot.xlabel(r'$\alpha$')
                pyplot.ylabel(r'$\delta$')

            if j < 2:
                ax.coords[0].set_ticklabel_visible(False)
                ax.coords[0].set_axislabel('')

            if (j == 1) | (j == 3):
                ax.coords[1].set_ticklabel_visible(False)
                ax.coords[1].set_axislabel('')


            ax.coords[0].set_ticks(size=10)
            ax.coords[1].set_ticks(size=10)

            ax = fig.add_subplot(2,4,2*j+2, projection=wcs)
            mdl =  Gauss2D(tfits[i,j,:], ra, dec,ra[xpix,ypix], dec[xpix,ypix])
            pyplot.imshow(m-mdl,origin='lower', vmax=amax+nmax*3, vmin=-nmax*3)
            pyplot.plot([crval[0]], [crval[1]],'kx', transform=ax.get_transform('world'), alpha=0.7)
            pyplot.title('SB {}'.format(j))
            if crval[0] == 0:
                pyplot.xlabel(r'$\Delta \alpha$')
                pyplot.ylabel(r'$\Delta \delta$')
            else:
                pyplot.xlabel(r'$\alpha$')
                pyplot.ylabel(r'$\delta$')
            ax.coords[1].set_ticklabel_visible(False)
            ax.coords[1].set_axislabel('')

            if j < 2:
                ax.coords[0].set_ticklabel_visible(False)
                ax.coords[0].set_axislabel('')

            if (j == 1) | (j == 3):
                ax.coords[1].set_ticklabel_visible(False)
                ax.coords[1].set_axislabel('')


            ax.coords[0].set_ticks(size=10)
            ax.coords[1].set_ticks(size=10)




        pyplot.subplots_adjust(wspace=0.2,hspace=0.05)
        outname = filename.split('.')[0]+'horn{}_images.png'.format(i)
        pyplot.savefig('Plots/{}'.format(outname), bbox_inches='tight')
        pyplot.clf()


    return amps, fits, mAzEl, tfits

listname = sys.argv[1]
filelist = np.loadtxt(listname, dtype='string', ndmin=1)

for i, filename in enumerate(filelist):
    amps, fits, mAzEl, tfits = MeasureAmps(filename)
    outname = filename.split('.')[0]+'_amps.h5'
    FileTools.WriteH5Py('amps/{}'.format(outname), {'amps':amps, 'fits':fits, 'mAzEl':mAzEl, 'meanFits':tfits})




#     

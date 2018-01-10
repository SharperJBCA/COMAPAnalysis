import ConfigParser
import sys
import os
from astropy.io import fits
import numpy as np
import healpy as hp

from matplotlib import pyplot
import FileTools
from MapMaker import Control
import Coordinates
import CartPix

import h5py

import jdcal

from scipy.interpolate import interp1d


def AtmosFits(jd, el, tod, stepSize, PlotDir=None, DataFile='Anon'):
    '''
    AtmosFits - Linear fit for atmosphere in data and subtract model

    el - encoder elevation measurements
    tod - receiver measurements
    stepSize - size of step (in samples)

    '''

    nSteps = int(tod.size//stepSize)

    offsets    = np.zeros(nSteps)
    gradients  = np.zeros(nSteps)
    mjds       = np.zeros(nSteps)

    ze = 1./np.sin(el*np.pi/180.)

    for i in range(nSteps):
        if i == nSteps - 1:
            high = tod.size
        else:
            high = (i+1)*stepSize

        pfit = np.poly1d(np.polyfit(ze[i*stepSize:high],
                                    tod[i*stepSize:high], 1))

        offsets[i] = pfit[0]
        gradients[i] = pfit[1]
        mjds[i] = np.mean(jd[i*stepSize:high])
        
        pyplot.clf()
        if not isinstance(PlotDir, type(None)):        
            pyplot.plot(ze[i*stepSize:high], tod[i*stepSize:high])
            pyplot.plot(np.arange(np.min(ze[i*stepSize:high]), np.max(ze[i*stepSize:high])),
                        pfit(np.arange(np.min(ze[i*stepSize:high]), np.max(ze[i*stepSize:high]))))
            pyplot.xlabel('Model')
            pyplot.ylabel('Observations')
            pyplot.legend(loc='best')
            pyplot.title('{} Samples: {} - {}'.format(DataFile, i*stepSize, high))
            pyplot.savefig('{}/{}_AtmosFit_{:04d}.png'.format(PlotDir, DataFile, int(i)))
            pyplot.clf()

        tod[i*stepSize:high] -= pfit(ze[i*stepSize:high])

    if not isinstance(PlotDir, type(None)):        
        pyplot.plot(mjds - int(mjds[0]), gradients,'o')
        pyplot.title('{} Fitted Amplitudes'.format(DataFile))
        pyplot.savefig('{}/{}_FittedAmps.png'.format(PlotDir, DataFile))
        pyplot.clf()

    return offsets, gradients

def TimeString2JD(time):
    ts = time.split(':')
    jd = np.sum(jdcal.gcal2jd(ts[0], ts[1],ts[2]))
    return jd + float(ts[3])/24. + float(ts[4])/24./60. + float(ts[5])/24./3600.


params = sys.argv[1]
Config = ConfigParser.ConfigParser()
Config.read(params)

PlotDir = Config.get('Inputs','PlotDir')
DataFile = Config.get('Inputs','DataFile')
if 'none'.lower() in PlotDir.lower():
    PlotDir = None

todjd0 = TimeString2JD(Config.get('Observation','todstart')) + float(Config.get('TimeCorrections', 'additiveFactor') )

nside = Config.getint('Inputs', 'nside')
bl = Config.getint('Inputs', 'baseline')
blong = Config.getint('Inputs', 'blong')
npix = 12*nside**2

# Read in the TOD, integrate over all SB0
if os.path.isfile('{}/{}'.format(Config.get('Inputs', 'datadir'),Config.get('Inputs', 'compressedtod'))):
    tod = FileTools.ReadH5Py('{}/{}'.format(Config.get('Inputs', 'datadir'),Config.get('Inputs', 'compressedtod')))['auto_py']
    todjd = np.arange(tod.size)/Config.getfloat('Observation', 'todsr') / 3600./ 24. * (1. - float(Config.get('TimeCorrections', 'multiFactor')) ) + todjd0
else:
    todfile = h5py.File('{}/{}'.format(Config.get('Inputs', 'datadir'), Config.get('Inputs', 'todfile')))
    tod = np.mean(todfile['auto_py'][:,0,:],axis=1)
    todfile.close()
    FileTools.WriteH5Py('{}/{}'.format(Config.get('Inputs', 'datadir'),Config.get('Inputs', 'compressedtod')),
                        {'auto_py': tod})

# Check for nans in TOD and normalise
tod[np.isnan(tod)] = 0.
tod = (tod - np.mean(tod))/np.std(tod)



# Read in encoder data
encfile = h5py.File('{}/{}'.format(Config.get('Inputs', 'datadir'), Config.get('Inputs', 'encfile')))
az = encfile['comap']['pointing']['azEncoder'][...]
el = encfile['comap']['pointing']['elEncoder'][...]
jd = encfile['comap']['pointing']['MJD'][...] + 2400000.5
encfile.close()

# Interpolate encoder data to receiver sample rate
coords = [el, az]
rCoords = []
for c in coords:
    pmdl = interp1d(jd, c, bounds_error=False, fill_value=0)
    rCoords += [pmdl(todjd)]

lat = Config.getfloat('Telescope', 'Latitude')
lon = Config.getfloat('Telescope', 'Longitude')
ra, dec =Coordinates._hor2equ(rCoords[1], rCoords[0], todjd, lat, lon)

# Plot RA DEC Positions
if not isinstance(PlotDir, type(None)):   
    pyplot.clf()
    pyplot.plot(ra, dec,'.')
    pyplot.xlabel('Right Ascension')
    pyplot.ylabel('Declination')
    pyplot.savefig('{}/{}_RADec.png'.format(PlotDir, DataFile))
            
AtmosFits(todjd, rCoords[0], tod, 
          Config.getint('Atmosphere', 'stepSize'),
          PlotDir=PlotDir,
          DataFile=Config.get('Inputs', 'DataFile'))

# Setup WCS
cdelt = [Config.getfloat('Observation', 'cdelt1'), Config.getfloat('Observation', 'cdelt2')]
naxis = [Config.getint('Observation', 'naxis1'), Config.getint('Observation', 'naxis2')]
crval = [Config.getfloat('Observation', 'crval1'), Config.getfloat('Observation', 'crval2')]
pix = CartPix.ang2pix(naxis, cdelt, crval, dec, ra).astype('int')
wcs = CartPix.Info2WCS(naxis, cdelt, crval)
npix = naxis[0]*naxis[1]

# Running MapMaking
bl  = Config.getint('Inputs', 'baseline')
baselines  = np.arange(tod.size).astype('int')//bl
mask = np.ones(tod.size)
mask[:Config.getint('Inputs', 'maskStart')] = 0
hitmap, recmap, a0, wroot, vmap = Control.Run(mask, mask.astype('int'), pix, baselines.astype('int') ,npix, threads=6, submean=False, noavg=True)
outputmap, recmap, a0, wroot, vmap = Control.Run(tod, mask.astype('int'), pix, baselines.astype('int') ,npix, threads=6)

gd = (mask[::bl] != 0)
afit = np.poly1d(np.polyfit(np.arange(a0.size)[gd], a0[gd], 2))

a0 -= afit(np.arange(a0.size))
tod -= np.repeat(a0, bl)

recmap, v2, v3, v4, vmap = Control.Run(tod, mask.astype('int'), pix, baselines.astype('int') ,npix, threads=6, destripe=False)
wroot, v2, v3, v4, vmap = Control.Run( np.repeat(a0, bl), mask.astype('int'), pix, baselines.astype('int') ,npix, threads=6, destripe=False)

# Regrid data into 2D images:
recmap[recmap == 0] = np.nan
outputmap[outputmap == 0] = np.nan
wroot[wroot == 0] = np.nan
hitmap[hitmap == 0] = np.nan
m = np.reshape(recmap, (naxis[1], naxis[0]))
m0 = np.reshape(outputmap, (naxis[1], naxis[0]))
a0 = np.reshape(wroot, (naxis[1], naxis[0]))
hit = np.reshape(hitmap, (naxis[1], naxis[0]))

# Save to fits
header = wcs.to_header()
hdu = fits.PrimaryHDU(m, header=header)
hdu2 = fits.ImageHDU(m0)
hdu3 = fits.ImageHDU(a0)
hdu4 = fits.ImageHDU(hit)
hdulist = fits.HDUList()
hdulist.append(hdu)
hdulist.append(hdu2)
hdulist.append(hdu3)
hdulist.append(hdu4)
hdulist.writeto( Config.get('Inputs', 'mapfile'), overwrite=True)

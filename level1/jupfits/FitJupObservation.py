import numpy as np
from matplotlib import pyplot
try:
    import ConfigParser
except ModuleNotFoundError:
    import configparser as ConfigParser
import argparse
import h5py
from scipy.signal import savgol_filter
import Pointing
from os import listdir, getcwd
from os.path import isfile, join
import Mapping
import Pointing

import mpi4py

import FitSource
import EphemNew

import healpy as hp
def cel2gal(ra,dec, inverse=False):
    _r, _d = ra*np.pi/180., (np.pi/2. - dec*np.pi/180.)
    if inverse:
        r = hp.Rotator(coord=['G','C'])
    else:
        r = hp.Rotator(coord=['C','G'])
    _d, _r = r(_d, _r)
    
    return _r*180./np.pi, (np.pi/2. - _d)*180./np.pi

def SlewDistance(az):
    daz = np.abs(az[:az.size-1] - az[1:az.size])
    
    # loop over spikes
    start = np.argmax(daz)
    peaks = [start]
    searchRange = 1000
    indices = np.arange(daz.size).astype(int)
    find = np.zeros(daz.size).astype(bool)
    thres = 0.01
    while True:
        find = find | (indices > start-searchRange) & (indices < start + searchRange)
        if (np.sum(find) == daz.size):
            break
        start = (indices[~find])[np.argmax(daz[~find])]
        peaks += [start]
        if np.max(daz[find]) < thres:
            break
    peaks = np.sort(np.array(peaks))

    peakAz = az[peaks]
    slewDist = np.abs(peakAz[:peakAz.size//2 *2:2] - peakAz[1:peakAz.size//2 *2:2])

    return np.median(slewDist)

    
            
def main(filename, plotDir='Plots/'):
    """
    """

    # Which pixels and sidebands?
    pixelOffsets = Pointing.GetPixelOffsets('COMAP_FEEDS.dat')

    # READ IN THE DATA
    d = h5py.File(filename)

    tod = d['spectrometer/tod']
    mjd = d['spectrometer/MJD'][:] 
    ra  = d['pointing/ra'][...]
    dec = d['pointing/dec'][...]
    az = d['pointing/az'][0,:]
    el = d['pointing/el'][0,:]
    slewDist = SlewDistance(az)

    # Calculate data sizes:
    nHorns = tod.shape[0]
    nSBs   = tod.shape[1]
    nFreqs = tod.shape[2]
    nSamps = tod.shape[3]

    # Calculate the position of Jupiter
    clon, clat, diam = EphemNew.rdplan(mjd[0:1], 5, 
                                       Pointing.comap_lon*np.pi/180., 
                                       Pointing.comap_lat*np.pi/180.)

    EphemNew.precess(clon, clat, mjd[0:1])
    pa = EphemNew.pa(ra[0,:]*np.pi/180., dec[0,:]*np.pi/180.,mjd, 
                     Pointing.comap_lon*np.pi/180.,Pointing.comap_lat*np.pi/180.)
    pa *= 180./np.pi
    # Loop over horns/SBs 
    P1out = None
    prefix = filename.split('/')[-1].split('.')[0]
    for iHorn in range(nHorns):
        print(iHorn)
        _tod = np.nanmean(np.nanmean(tod[iHorn,:,5:-5,:],axis=0),axis=0)

        #Tim: Pass this function whatever chunk of time-ordered data you have in memory
        P1, P1e, cross, mweight, weight, model = FitSource.FitTOD(_tod,
                                                                  ra[0,:],  # horn 0 because we want the relative offset from Focal Plane
                                                                  dec[0,:], 
                                                                  clon*180./np.pi, 
                                                                  clat*180./np.pi, 
                                                                  pa, 
                                                                  prefix='{}_Horn{}'.format(prefix, iHorn+1),
                                                                  plotDir=plotDir)
        
        if isinstance(P1out, type(None)):
            P1out = np.zeros((nHorns, len(P1)))
            Peout = np.zeros((nHorns, len(P1e)))
            mout = np.zeros(mweight.shape)
            hout = np.zeros(weight.shape)

        P1out[iHorn, :] = P1
        Peout[iHorn, :] = P1e
        mout += mweight*(model+1)**2
        hout += weight*(model+1)**2

    pyplot.imshow(mout/hout, extent=[-100/2. * 1.5, 100/2.*1.5,-100/2. * 1.5, 100/2.*1.5] )
    pyplot.xlabel('Az offset (arcmin)')
    pyplot.ylabel('EL offset (arcmin)')
    pyplot.title('{}'.format(prefix))
    pyplot.grid(True)
    pyplot.savefig('{}/FeedPositions_{}.png'.format(plotDir, prefix), bbox_inches='tight')
    pyplot.clf()

    
    meanMJD = np.mean(mjd)
    meanEl  = np.median(el)
    meanAz  = np.median(az)

    d.close()
    print('SLEW DISTANCE', slewDist)
    return P1out, Peout, mout/hout, meanMJD, meanEl, meanAz

from mpi4py import MPI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str)
    parser.add_argument('--filelist', default=None, type=str)
    parser.add_argument('--fitoutputdir', default='.', type=str)
    args = parser.parse_args()

    P1 = None
    if isinstance(args.filelist, type(None)):
        main(args.filename)
    else:
        filelist = np.loadtxt(args.filelist, dtype=str)

        for i, f in enumerate(filelist):
            print('Opening',f)
            _P1, _P1e, m, meanMJD, meanEl, meanAz = main(f)
            prefix = f.split('/')[-1].split('.h')[0]
            output = h5py.File('{}/{}_JupiterFits.h5'.format(fitoutputDir, prefix))
            output['P1'] = _P1
            output['P1e'] = _P1e
            coords = np.zeros(3)
            coords[:] = meanAz, meanEl, meanMJD,

            output['coords'] = coords
            output['map'] = m
            output.close()

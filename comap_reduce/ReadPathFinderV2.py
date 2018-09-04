import numpy as np
from matplotlib import pyplot
import h5py
import argparse
from scipy.interpolate import interp1d
from pipeline.Observatory.Telescope import Coordinates

import CartPix
import EphemNew
from scipy.optimize import leastsq

try:
    import ConfigParser
except ModuleNotFoundError:
    import configparser as ConfigParser
    
from matplotlib import pyplot

# COMAP PIPELINE MODULES
import Pointing
import Atmosphere
import Mapping
import Filters
import FitSource

import json
import sys

from pipeline.Tools import FileTools



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('paramFile')
    parser.add_argument('--badchannels', nargs='+', type=int)
    parser.add_argument('--horn', default=0, type=int)
    parser.add_argument('--sb', default=0, type=int)
    parser.add_argument('--nstart', default=0, type=int)
    parser.add_argument('--nend', default=None, type=int)
    parser.add_argument('--mapname', default='outputmaps')
    parser.add_argument('--isfilelist', default=True, type=bool)
    parser.add_argument('--circular', default=False, type=bool)
    parser.add_argument('--lat', default=37.2314, type=float)
    parser.add_argument('--lon', default=-118.2941, type=float)
    args = parser.parse_args()

    Parameters = ConfigParser.ConfigParser()

    Parameters.read(sys.argv[1])#args['paramFile'])

    # First get the list of files
    if Parameters.getboolean('Inputs', 'isfilelist'):
        filelist = np.loadtxt(Parameters.get('Inputs', 'filename'), ndmin=1, dtype=str)
    else:
        filelist = [Parameters.get('Inputs', 'filename')]

    dataDir = Parameters.get('Inputs', 'dataDir')

    # Which pixels and sidebands?
    pixMap = {1:0, 12:1}
    pixels = json.loads(Parameters.get('Inputs', 'pixels'))
    pixels = np.array([pixMap[p] for p in pixels]) # put the pixels into data order
    sidebands = json.loads(Parameters.get('Inputs', 'sidebands'))
    print(sidebands)
    # What is the source being observed?
    source = Parameters.get('Inputs', 'source')


    # What is the lon and lat of the telescope?
    lon = Parameters.getfloat('Telescope', 'lon')
    lat = Parameters.getfloat('Telescope', 'lat')
    # alt = ...
    
    # Describe the format of the data:

    dataout = dict()
    
    if Parameters.getboolean('Inputs', 'mergeDatafiles'):
        count = 0
        countPointing = 0
        for i, filename in enumerate(filelist):
            prefix = filename.split('.')[0]
            if len(prefix.split('/')) > 1:
                prefix = prefix.split('/')[1]

            print ('READING: {}'.format(filename))
            dfile = h5py.File('{}/{}'.format(dataDir,filename),'r')
            count += dfile['spectrometer/tod'].shape[-1]
            countPointing += dfile['pointing/azActual'].shape[0]
            print (dfile['pointing/azActual'].shape[0])
            if i == 0:
                nchans = dfile['spectrometer/tod'].shape[2]
            dfile.close()

        az, el, mjd = np.zeros(countPointing), np.zeros(countPointing), np.zeros(countPointing)
        tod = np.zeros((len(pixels), len(sidebands), nchans, count))
        mjdTOD = np.zeros(count)
        obs = np.zeros(count)
        last = {'tod':0, 'point':0}
        this = {'tod':0, 'point':0}
        for i, filename in enumerate(filelist):
            dfile = h5py.File('{}/{}'.format(dataDir,filename),'r')
            this['tod'] = dfile['spectrometer/tod'].shape[-1]
            this['point'] = dfile['pointing/azActual'].shape[0]
            print ('READING: {}'.format(filename))
            print (dfile['pointing/azActual'].shape[0])

            print (this, dfile['pointing/MJD'].shape, dfile['pointing/azActual'].shape, dfile['pointing/elActual'].shape, count, countPointing)
            # Read in encoder info
            az[last['point']:last['point']+this['point']] = dfile['pointing/azActual'][:]
            el[last['point']:last['point']+this['point']] = dfile['pointing/elActual'][:]
            mjd[last['point']:last['point']+this['point']]= dfile['pointing/MJD'][:]
        
            # Read in TOD info
            mjdTOD[last['tod']:last['tod']+this['tod']] = dfile['spectrometer/MJD'][:]
            obs[last['tod']:last['tod']+this['tod']] = i
            for isideband, sideband in enumerate(sidebands):
                print (isideband, sidebands)
                tod[:,sideband,:,last['tod']:last['tod']+this['tod']] = dfile['spectrometer/tod'][pixels, sideband, :, :]

            last['tod']   += this['tod'] 
            last['point'] += this['point']
            dfile.close()
        
        nLoop = 1
    else:
        nLoop = len(filelist)

    #pyplot.plot(np.mean(tod[0,0,:,:],axis=0))
    #pyplot.show()

    # Now loop over each file
    for i in range(nLoop):
        if not Parameters.getboolean('Inputs', 'mergeDatafiles'):
            filename = filelist[i]
            prefix = filename.split('.')[0]
            if len(prefix.split('/')) > 1:
                prefix = prefix.split('/')[1]

            print ('READING: {}'.format(filename))
            dfile = h5py.File('{}/{}'.format(dataDir,filename),'r')
            az = dfile['pointing/azActual'][:]
            el = dfile['pointing/elActual'][:]
            mjd= dfile['pointing/MJD'][:]
        
            # Read in TOD info
            mjdTOD = dfile['spectrometer/MJD'][:]
            tod = dfile['spectrometer/tod'][pixels, :, :, :]
            tod = tod[:, sidebands,:,:]
            obs = np.zeros(1)
            dfile.close()

        # Split off from here

        # First get the telescope pointings
        if 'JUPITER' in source.upper():
            doPrecess = False
        elif 'JUPITERSIM' in source.upper():
            doPrecess = False
        else:
            doPrecess = True

        print('GENERATING POINTING')
        ra, dec, pang, az, el, mjd = Pointing.GetPointing(az, el, mjd, mjdTOD, pixels, sidebands, precess=doPrecess, lon=lon, lat=lat)
        nhorns = ra.shape[0]


        if  source.upper() == 'JUPITER':
            print('HELLO')
            r0, d0, dist = Pointing.GetSource(source, lon, lat, mjd)
            dataout['Jupiter'] = np.array([r0,d0,dist])
        elif 'JUPITERSIM' in source.upper():
            r0, d0, dist =  225.34744833333335, -16.262374277777777, 5.708
            dataout['Jupiter'] = np.array([r0,d0,dist])
        else:
            dist = 0
            r0, d0 = json.loads(Parameters.get('Inputs', 'sourceRADEC'))

        # Calculate the mean az and el of the observation
        #meanAz, meanEl = Pointing.MeanAzEl(r0, d0, mjd, precess=doPrecess)
        #dataout['mAzEl'] = np.array([meanAz, meanEl])

        # Next check if we want to rotate source frame?
        if Parameters.getboolean('Inputs', 'rotate'):
            for i in range(nhorns):
                ra[i,:], dec[i,:] = Pointing.Rotate(ra[i,:], dec[i,:], r0, d0, pang[i,:])

        # Get the spectral data
        # Level 1 v2 format: feed, sb, chan, samp

        # Averaging Channels to improve signal to noise:
        if Parameters.getboolean('Averaging', 'average'):
            stride = Parameters.getint('Averaging','stride')
            badChans = json.loads(Parameters.get('Averaging','badChannels'))

            dnu = 2./tod.shape[2]
            nu = np.arange(tod.shape[2])*dnu + dnu/2.
            print(nu, dnu)
            tod, anu = Filters.AverageFreqs(tod, nu, stride, badChans)
            dataout['nu'] = anu


        # Next make atmospheric corrections
        if Parameters.getboolean('Atmosphere', 'remove'):
            stride = Parameters.getint('Atmosphere','stride')
            A = Atmosphere.SimpleRemoval(tod, el, stride)
            dataout['atmos']= A

        if Parameters.getboolean('Filters','median'):
            stride = Parameters.getint('Filters','stride')
            Filters.MedianFilter(tod, stride)
            


        # Fit for the Source in TOD
        if Parameters.getboolean('Fitting', 'fit'):
            print('FITTING JUPITER IN TOD')
            if Parameters.getboolean('Inputs', 'mergeDatafiles'):
                Pout, errors, crossings = FitSource.FitTOD(tod, ra, dec, obs, r0, d0, prefix, destripe=True)
            else:
                Pout, errors, crossings = FitSource.FitTOD(tod, ra, dec, obs, r0, d0, prefix, destripe=False)

            # Pout, crossings = FitSource.FitTOD(tod, az, el, meanAz, meanEl, prefix)
            print (mjd.shape)
            crossings = crossings.astype('int')
            peakaz, peakel = Pointing.MeanAzEl(r0, d0, mjd[crossings], precess=doPrecess, lon=lon, lat=lat)

            print(peakaz, peakel)
            print(peakaz.shape)
            dataout['TODFits'] = Pout
            dataout['TODFitsErrors'] = errors
            dataout['mAzEl'] = np.array([[az[ci,crossings[ci]], 
                                         el[ci,crossings[ci]],
                                         peakaz[ci],
                                         peakel[ci],
                                         pang[ci,crossings[ci]],
                                         mjd[crossings[ci]]] for ci in range(crossings.size) ])
            print(crossings)
            print(dataout['mAzEl'].shape)
        # Make maps
        if Parameters.getboolean('Mapping', 'map'):
            print('Making Maps')
            naxis = json.loads(Parameters.get('Mapping','naxis'))

            if Parameters.getboolean('Inputs', 'rotate'):
                crval = [0, 0]
            else:
                crval = [r0, d0]
            cdelt = np.array(json.loads(Parameters.get('Mapping','cdelt')))/60.

            minRa  = np.min(ra)
            maxRa  = np.max(ra)
            minDec = np.min(dec)
            maxDec = np.max(dec)

            if (crval[0] < minRa) | (crval[0] > maxRa) | (crval[1] < minDec) | (crval[1] > maxDec):
                print('WARNING: MAP CENTRE DOES NOT MATCH TELESCOPE POINTING CENTRE. CHECK COORDINATES')
                print('MEAN RA: {:.2f}, MEAN DEC: {:.2f}'.format(np.mean(ra), np.mean(dec)))
            

            wcs,_,_ = Mapping.DefineWCS(naxis, cdelt, crval)
            maps, hits = Mapping.MakeMaps(tod, ra, dec, wcs)
            dataout['hits']  = hits
            dataout['maps']  = maps
            dataout['naxis'] = np.array(naxis)
            dataout['cdelt'] = np.array(cdelt)
            dataout['crval'] = np.array(crval)


        sbStr = ''.join(str(e) for e in sidebands)
        hoStr = ''.join(str(e) for e in pixels)
        FileTools.WriteH5Py('{}/{}_{}_Horns{}_Sidebands{}.h5'.format(Parameters.get('Inputs', 'outputDir'), 
                                                                     Parameters.get('Inputs', 'outputname'),
                                                                     prefix,
                                                                     hoStr,
                                                                     sbStr), dataout)



        dataout = {} # clear data

from pipeline.Observatory.Telescope import Coordinates
import numpy as np
from scipy.interpolate import interp1d
import CartPix
import EphemNew

import healpy as hp

# PIXEL INFORMATION TAKEN FROM JAMES' MEMO ON WIKI
p = 0.1853 # arcmin mm^-1, inverse of effective focal length

theta = np.pi/2.
Rot = np.array([[np.cos(theta), -np.sin(theta)],
                [-np.sin(theta),-np.cos(theta)]])

pixelOffsets = {0: [0, 0], # pixel 1
                1: Rot.dot(np.array([-65.00, 112.58])).flatten()} # pixel 12
pixelOffsets = {0: [0, 0], # pixel 1
                1: [0, 0]} # pixel 12

pixelOffsets = {0: [0,[0, 0]], # pixel 1
                2:  [2, Rot.dot(np.array([97.50, 56.29])).flatten()], # Pixel 9
                3: [3, Rot.dot(np.array([-97.50, 56.29])).flatten()], # Pixel 13
                5: [5, Rot.dot(np.array([-97.50, -56.29])).flatten()], # Pixel 15
                12: Rot.dot(np.array([-65.00, 112.58])).flatten()} # pixel 12



#pixelOffsets = {0: [0,[0, 0]], # pixel 1
#                1: [1,
#                2: [2, Rot.dot(np.array([97.50, 56.29])).flatten()], # Pixel 9
#                3: [3, Rot.dot(np.array([-97.50, 56.29])).flatten()], # Pixel 13
#                5: [5, Rot.dot(np.array([-97.50, -56.29])).flatten()], # Pixel 15
#                12: Rot.dot(np.array([-65.00, 112.58])).flatten()} # pixel 12


feedpositions = np.loadtxt('COMAP_Feed_Positions.dat')
pixelOffsets = {k+1: [k, (Rot.dot(f[1:,np.newaxis])).flatten()] for k, f in enumerate(feedpositions)} #k, f enumerate(feedpositions)}

def GetSource(source, lon, lat, mjdtod):

    if 'JUPITER' in source.upper():
        r0, d0, jdia = EphemNew.rdplan(mjdtod, 5,  lon*np.pi/180., lat*np.pi/180.)
        jdist = EphemNew.planet(mjdtod, 5)
        edist = EphemNew.planet(mjdtod, 3)
        rdist = np.sqrt(np.sum((jdist[:3,:] - edist[:3,:])**2,axis=0))

        r0 = np.mean(r0)*180./np.pi
        d0 = np.mean(d0)*180./np.pi
        dist = np.mean(rdist)
    else:
        r0, d0, dist = 0, 0, 0

    return r0, d0, dist


def GetPointing(_az, _el, mjdp, mjdtod, pixels, sidebands, lon= -118.2941, lat=37.2314, precess=True):
    """
    Expects a level 1 COMAP data file and returns ra, dec, az, el and mjd for each pixel

    Default lon/lat set to COMAP pathfinder telescope
    """

    #_az, _el, mjdp = dfile['pointing/azActual'][:], dfile['pointing/elActual'][:], dfile['pointing/MJD'][:]
    #mjdtod =  dfile['spectrometer/MJD'][:]
    
    # Need to map pointing MJD to Spectrometer MJD
    amdl = interp1d(mjdp, _az, bounds_error=False, fill_value=0)
    emdl = interp1d(mjdp, _el, bounds_error=False, fill_value=0)
    print( np.sum(_az), np.sum(_el))

    _az, _el = amdl(mjdtod), emdl(mjdtod)
    #findNan = np.where(np.isnan(_az))[0]
    #print(findNan)
    #_az[np.isnan(_az
    # What pixels are in this data file?
    #pixels =  np.unique([s[:-1] for s in dfile['spectrometer/pixels']])
    ra, dec = np.zeros((pixels.size, _az.size)), np.zeros((pixels.size, _az.size))
    az, el = np.zeros((pixels.size, _az.size)), np.zeros((pixels.size, _az.size))
    pang = np.zeros((pixels.size, _az.size))
    # Calculate RA/DEC for each pixel
    for i, pix in enumerate(pixels):
        print(pix, pixelOffsets[pix][1])
        el[i,:] = _el+pixelOffsets[pix][1][1]/60.*p
        az[i,:] = _az+pixelOffsets[pix][1][0]/60.*p/np.cos(el[i,:]*np.pi/180.)
        ra[i,:], dec[i,:] = Coordinates._hor2equ(az[i,:],
                                                 el[i,:],
                                                 mjdtod[:]+2400000.5,
                                                 lat,
                                                 lon,precess=precess)
        pang[i,:] = Coordinates._pang(el[i,:], dec[i,:], lat)

    return ra, dec, pang, az, el, mjdtod

def MeanAzEl(r0, d0, mjd, lon= -118.2941, lat=37.2314, precess=True):
    
    meanmjd = np.mean(np.array([mjd]))
    maz, mel  = Coordinates._equ2hor(np.repeat([r0], mjd.size),
                                     np.repeat([d0], mjd.size),
                                     mjd + 2400000.5,
                                     #np.array([meanmjd])+2400000.5,
                                     lat,
                                     lon, precess=precess)

    return maz, mel


def RotatePhi(skyVec, objRa):
    outVec = skyVec*0.
    # Rotate first RA
    outVec[:,0] =  skyVec[:,0]*np.cos(objRa*np.pi/180.) + skyVec[:,1]*np.sin(objRa*np.pi/180.) 
    outVec[:,1] = -skyVec[:,0]*np.sin(objRa*np.pi/180.) + skyVec[:,1]*np.cos(objRa*np.pi/180.) 
    outVec[:,2] =  skyVec[:,2]
    return outVec

def RotateTheta(skyVec, objDec):
    outVec = skyVec*0.
    # Rotate first Dec
    outVec[:,0] =  skyVec[:,0]*np.cos(objDec*np.pi/180.) + skyVec[:,2]*np.sin(objDec*np.pi/180.) 
    outVec[:,1] =  skyVec[:,1]
    outVec[:,2] = -skyVec[:,0]*np.sin(objDec*np.pi/180.) + skyVec[:,2]*np.cos(objDec*np.pi/180.) 
    return outVec


def RotateR(skyVec, objPang):
    outVec = skyVec*0.
    # Rotate first pang
    outVec[:,0] =  skyVec[:,0]
    outVec[:,1] =  skyVec[:,1]*np.cos(objPang*np.pi/180.) + skyVec[:,2]*np.sin(objPang*np.pi/180.) 
    outVec[:,2] = -skyVec[:,1]*np.sin(objPang*np.pi/180.) + skyVec[:,2]*np.cos(objPang*np.pi/180.) 
    return outVec

def Rotate(ra, dec, r0, d0, p0):
    """
    Rotate coordinates to be relative to some ra/dec and sky rotation pang
    
    All inputs in degrees

    """
    skyVec = hp.ang2vec((90.-dec)*np.pi/180., ra*np.pi/180.)

    outVec = RotatePhi(skyVec, r0)
    outVec = RotateTheta(outVec, d0)
    outVec = RotateR(outVec, p0)

    _dec, _ra = hp.vec2ang(outVec)
    _dec = (np.pi/2. - _dec)*180./np.pi
    _ra = _ra * 180./np.pi

    _ra[_ra > 180] -= 360.
    return _ra, _dec

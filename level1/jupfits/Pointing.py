#from pipeline.Observatory.Telescope import Coordinates
import numpy as np
from scipy.interpolate import interp1d
import CartPix
import EphemNew

import healpy as hp

# PIXEL INFORMATION TAKEN FROM JAMES' MEMO ON WIKI
p = 0.1853 # arcmin mm^-1, inverse of effective focal length

theta = np.pi/2.*1.
Rot = np.array([[np.cos(theta), -np.sin(theta)],
                [-np.sin(theta),-np.cos(theta)]])
comap_lon = -118.2941
comap_lat = 37.2314


def GetPixelOffsets(pixelFile):
    feedpositions = np.loadtxt(pixelFile, ndmin=1)
    pixelOffsets, c = {},0
    pixels = np.arange(19).astype(int) + 1
    for k, f in enumerate(feedpositions):
        if f[0] in pixels:
            pixelOffsets[f[0]] = [c, (Rot.dot(f[1:,np.newaxis])).flatten()/60.*p, k]
            c += 1

    return pixelOffsets

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

import EphemNew
def GetPointing(_az, _el, mjdp, mjdtod, pixelOffsets, lon= -118.2941, lat=37.2314, precess=True):
    """
    Expects a level 1 COMAP data file and returns ra, dec, az, el and mjd for each pixel

    Default lon/lat set to COMAP pathfinder telescope
    """

    nPix = len(pixelOffsets)

    # Need to map pointing MJD to Spectrometer MJD
    amdl = interp1d(mjdp, _az, bounds_error=False, fill_value=0)
    emdl = interp1d(mjdp, _el, bounds_error=False, fill_value=0)

    _az, _el = amdl(mjdtod), emdl(mjdtod)

    # What pixels are in this data file?
    ra, dec = np.zeros((nPix, _az.size)), np.zeros((nPix, _az.size))
    az, el = np.zeros((nPix, _az.size)), np.zeros((nPix, _az.size))
    pang = np.zeros((nPix, _az.size))
    # Calculate RA/DEC for each pixel
    for k ,pix in pixelOffsets.iteritems():
        i = int(pix[0])
        #print(k,i,pix)
        el[i,:] = _el+pix[1][1]
        az[i,:] = _az+pix[1][0]/np.cos(el[i,:]*np.pi/180.) + 4.25/60. # azimuth correction of 4.25 arcmin
        ra[i,:], dec[i,:] = EphemNew.h2e(az[i,:]*np.pi/180., el[i,:]*np.pi/180., mjdtod, lon*np.pi/180., lat*np.pi/180.)
        if precess:
            EphemNew.precess(ra[i,:], dec[i,:], mjdtod)
        #pang[i,:] = Coordinates._pang(el[i,:], dec[i,:]*180./np.pi, lat)
    
    ra *= 180./np.pi
    dec *= 180./np.pi
    return ra, dec, pang, az, el, mjdtod

def MeanAzEl(r0, d0, mjd, lon= -118.2941, lat=37.2314, precess=True):
    
    meanmjd = np.mean(np.array([mjd]))
    maz, mel  = Coordinates._equ2hor(np.repeat([r0], mjd.size),
                                     np.repeat([d0], mjd.size),
                                     mjd + 2400000.5,
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

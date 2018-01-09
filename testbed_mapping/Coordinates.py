"""
Name: Coordinates.py

Description: Marie-Annes Python code for converting between Horizon and Sky coordinate systems.

"""

from __future__ import division

import numpy as np
from astropy.time import TimeDelta
from numpy.random import random_sample as randomu

def _premat(equinox1, equinox2, FK4=True):
    """
    Return precession matrix needed to go from equinox1 to equinox2

    equinox1 - original equinox of coordinates
    equinox2 - equinox to precess to

    returns 3x3 precession matrix

    Shameless stolen from astrolib premat.pro
    """
    d2r = np.pi/180.
    s2r = d2r/3600. # convert seconds to radians

    T = 1e-3*(equinox2 - equinox1)

    if FK4:
        ST = 1e-3*(equinox1 - 2000.)

        # Compute 3 rotation angles
        A = s2r * T * \
            (23062.181 + ST * (139.656 + 0.0139*ST) \
                 + T * (30.188 - 0.344 * ST + 17.998 * T))

        B = s2r * T * T * (79.280 + 0.410*ST + 0.205*T) + A

        C = s2r * T * (20043.109 - ST*(85.33 + 0.217*ST) \
                           + T*(-42.665 - 0.217*ST - 41.833*T))
    else:
        A = 0.
        B = 0.
        C = 0.

    sina = np.sin(A)
    sinb = np.sin(B)
    sinc = np.sin(C)
    cosa = np.cos(A)
    cosb = np.cos(B)
    cosc = np.cos(C)

    R = np.zeros((3, 3))
    
    R[:,0] = np.array([ cosa*cosb*cosc - sina*sinb, 
                        sina*cosb+cosa*sinb*cosc,
                        cosa*sinc]).flatten()
    R[:,1] = np.array([-cosa*sinb - sina*cosb*cosc,
                        cosa*cosb - sina*sinb*cosc,
                       -sina*sinc]).flatten()
    R[:,2] = np.array([-cosb*sinc,
                       -sinb*sinc,
                        cosc]).flatten()
    return R

def _precess(_ra, _dec, equinox1, equinox2):
    """
    Precess coordinate system from equinox1 to equinox2.

    ra and dec are inputs in radians.

    Shameless stolen from astrolib routine precess.pro
    (Based on procedure from Computational Spherical Astronomy by Taff (1983).
    p. 24.)
    """
    
    ra = _ra*np.pi/180.
    dec= _dec*np.pi/180.
    
    a = np.cos(dec)
    vec1 = np.array([a*np.cos(ra), 
                     a*np.sin(ra), 
                     np.sin(dec)]) # cartesian vector on sphere

    R0 = _premat(equinox1, equinox2) # get rotation matrix
    
    vec2 = R0.dot(vec1)

    ra_out = np.arctan2(vec2[1], vec2[0])
    dec_out= np.arcsin(vec2[2])

    if ra_out < 0:
        ra_out += 2.*np.pi
    #ra_out[ra_out < 0] += 2.*np.pi

    return ra_out*180./np.pi, dec_out*180./np.pi
    
def _jd2gst(jd):
    """
    Convert julian dates into Greenwich Sidereal Time.

    From Practical Astronomy With Your Calculator.
    """
    jd0 = np.floor(jd - 0.5) + 0.5
    T = (jd0 - 2451545.0) / 36525
    T0 = 6.697374558 + 2400.051336 * T + 0.000025862 * T**2
    T0 %= 24
    ut = (jd - jd0) * 24
    T0 += ut * 1.002737909
    T0 %= 24
    return T0


def _gst2lst(gst, geolon):
    """
    Convert Greenwich Sidereal Time into Local Sidereal Time.

    """
    # geolon: Geographic longitude EAST in degrees.
    return (gst + geolon / 15.) % 24

def _pang(el, dec, geolat):
    """
    Generate parallactic angle from elevation and declination

    """
    d2r = np.pi/180.0
    r2d = 180.0/np.pi

    top = np.sin(geolat*d2r) - np.sin(el*d2r)*np.sin(dec*d2r)
    bot = np.cos(el*d2r)*np.cos(dec*d2r)

    p = np.arccos(top/bot)
    if isinstance(p, type(np.array([]))):
        p[np.isnan(p)] = 0
        p[p > np.pi/2.] -= np.pi
    else:
        if np.isnan(p):
            p = 0
        else:
            if p > np.pi/2.:
                p -= np.pi

    

    return p*r2d

def _equ2hor(_ra, _dec, _jd, geolat, geolon, precess=False):
    """
    Convert from ra/dec to az/el (by Ken Genga).
    
    All inputs as degrees

    """
    # Imports
    from numpy import arccos, arcsin, cos, pi, sin, where

    ra = np.array([_ra]).flatten()
    dec= np.array([_dec]).flatten()
    jd = np.array([_jd]).flatten()
    
    lst = _gst2lst(_jd2gst(jd), geolon)
    
    
    if precess:
        J_now = (jd - 2451545.)/365.25 + 2000.0
        for i in range(len(J_now)):
            ra[i], dec[i] = _precess(ra[i], dec[i], 2000., J_now[i])
        
    az, el = _equ2hor_lst(ra, dec, lst, geolat)
    
    # Later
    return az, el
    
def _equ2hor_lst(_ra, _dec, _lst, geolat):
    from numpy import arccos, arcsin, cos, pi, sin, where

    ra = np.array([_ra]).flatten()
    dec= np.array([_dec]).flatten()
    lst = np.array([_lst]).flatten()
    
    d2r = pi/180.0
    r2d = 180.0/pi
    sin_dec = sin(dec*d2r)
    cos_dec = cos(dec*d2r)

    phi_rad = geolat*d2r
    sin_phi = sin(phi_rad)
    cos_phi = cos(phi_rad)
    ha = 15.0*_ra2ha(ra, lst)
        
    sin_ha = sin(ha*d2r)
    cos_ha = cos(ha*d2r)
    
    x = - cos_ha * cos_dec * sin_phi + sin_dec * cos_phi
    y = - sin_ha * cos_dec
    z =   cos_ha * cos_dec * cos_phi + sin_dec * sin_phi
    r =   np.sqrt(x**2 + y**2)
    
    az = np.arctan2(y, x)*180./np.pi
    el = np.arctan2(z, r)*180./np.pi
    w = (az < 0)
    az[w] = az[w] + 360.
    
    return az, el


def _hor2equ(_az, _el, _jd, geolat, geolon, precess=False):
    """
    Convert from az/el to ra/dec (by Ken Genga).

    All inputs in degrees
    """
    # Imports
    from numpy import arccos, arcsin, cos, pi, sin, where

    az = np.array([_az]).flatten()
    el = np.array([_el]).flatten()
    jd = np.array([_jd]).flatten()
    
    lst = _gst2lst(_jd2gst(jd), geolon)
    
    ra, dec = _hor2equ_lst(az, el, lst, geolat)

    
    if precess:
        J_now = (jd - 2451545.)/365.25 + 2000.0
        for i in range(len(J_now)):
            ra[i], dec[i] = _precess(ra[i], dec[i], J_now[i], 2000.)

    # Later
    return ra, dec

def _hor2equ_lst(_az, _el, _lst, geolat):
    az = np.array([_az]).flatten()
    el = np.array([_el]).flatten()
    lst = np.array([_lst]).flatten()

    from numpy import arccos, arcsin, cos, pi, sin, where

    d2r = pi/180.0
    r2d = 180.0/pi
    az_r = az*np.pi/180.
    el_r = el*np.pi/180.
    geolat_r = geolat*np.pi/180.

    # Convert to equatorial coordinates
    cos_el = cos(el_r)
    sin_el = sin(el_r)
    cos_phi = cos(geolat_r)
    sin_phi = sin(geolat_r)
    cos_az = cos(az_r)
    sin_az = sin(az_r)
    
    sin_dec = sin_el*sin_phi + cos_el*cos_phi*cos_az
    dec = arcsin(sin_dec)

    ha = [-sin_az*cos_el, -cos_az*sin_phi*cos_el + sin_el*cos_phi]    
    ha = np.arctan2(ha[0], ha[1])
    ha = np.mod(ha, np.pi*2.)
    
    
    ra = lst*15.0*np.pi/180.-ha
    ra = where(ra >= 2.*np.pi, ra - 2.*np.pi, ra)
    ra = where(ra < 0.0, ra + 2.*np.pi, ra)

    ra *= 180./np.pi
    dec *= 180./np.pi
    return ra, dec


def _ra2ha(ra, lst):
    """
    Converts a right ascension to an hour angle.

    """
    return (lst - ra / 15.0) % 24

def _equ2gal(ra, dec):
    """
    Converts right ascension and declination to Galactic lon and lat.

    Uses rotation matrix Rg from 'Spherical Astronomy by Robert Green, Chapter 14, page 355'
    """
    
    ra = ra*np.pi/180.
    dec= dec*np.pi/180.

    equVec = np.array([[np.cos(ra)*np.cos(dec)],
                        [np.sin(ra)*np.cos(dec)],
                        [np.sin(dec)]])

    tg = 0. 
    ag = (17. + 45.6/60.)*15.*np.pi/180.
    dg = -28.94*np.pi/180.

    Rg = np.array([[-0.054876, -0.873437, -0.483835],
                   [ 0.494109, -0.444830,  0.746982],
                   [-0.867666, -0.198076,  0.455984]])


    Rg = np.reshape(Rg, (3,  3, 1))
    Rg = np.transpose(Rg, [1,0,2])
    test = Rg*equVec
    galVec = np.sum(Rg*equVec, axis=0)#Rg.dot(equVec)
    lon = np.arctan2(galVec[1],galVec[0])
    lat = np.pi/2. - np.arctan2(np.sqrt(galVec[0]**2 + galVec[1]**2), galVec[2])
    
    lon = lon *180./np.pi
    lat = lat*180./np.pi
    return lon, lat

def _gal2equ(gl, gb):
    """
    Shamelessly copied from the IDL glactc routine.
    """

    rapol = (12. + 49./60.)*np.pi/180.*15.
    decpol= (27.4)*np.pi/180.
    dlon  = (123.0)*np.pi/180.

    sdp = np.sin(decpol)
    cdp = np.sqrt(1.0 - sdp*sdp)


    sgb = np.sin(gb)
    cgb = np.sqrt(1. - sgb**2)
    
    sdec = sgb*sdp + cgb*cdp*np.cos(dlon - gl)
    dec  = np.arcsin(sdec)

    cdec = np.sqrt(1.-sdec**2)
    sinf = cgb * np.sin(dlon-gl)/cdec
    cosf = (sgb-sdp*sdec)/(cdp*cdec)
    ra   = rapol + np.arctan2(sinf, cosf)

    return np.mod(ra, 2*np.pi), dec

def _nutate(jd):
    
    dtor = np.pi/180.
    
    T = (jd[:] - 2451545.0)/36525.0
    
    # Mean elongation of the Moon

    coeff1 = np.array([297.85036,  445267.111480, -0.0019142, 1./189474.])
    d = np.mod(np.polyval(coeff1[::-1], T)*dtor, 2*np.pi)
    d = np.reshape(d, (d.size, 1))
    
    # Sun's mean anomaly
    
    coeff2 = np.array([357.5277, 35999.050340, -0.0001603, -1./3e5 ])
    m = np.mod(np.polyval(coeff2[::-1], T)*dtor, 2.*np.pi)
    m = np.reshape(m, (m.size, 1))
    
    # Moon's mean anomaly
    
    coeff3 = np.array([134.96298, 477198.867398, 0.0086972, 1.0/5.625e4 ])
    mprime = np.mod(np.polyval(coeff3[::-1], T)*dtor, 2.*np.pi)
    mprime = np.reshape(mprime, (mprime.size, 1))
    
    # Moon's argument of latitude
    
    coeff4 = np.array([93.27191, 483202.017538, -0.0036825, -1.0/3.27270e5 ])
    f = np.mod(np.polyval(coeff4[::-1], T)*dtor, 2.*np.pi)
    f = np.reshape(f, (f.size, 1))
    
    # Longitude of the ascending node of the Moon's mean orbit on the ecliptic,
    #  measured from the mean equinox of the date
    coeff5 = np.array([125.04452, -1934.136261, 0.0020708, 1./4.5e5])
    omega = np.mod(np.polyval(coeff5[::-1], T)*dtor, 2.*np.pi)
    omega = np.reshape(omega, (omega.size, 1))
    
    
    d_lng = np.array([0,-2,0,0,0,0,-2,0,0,-2,-2,-2,0,2,0,2,0,0,-2,0,2,0,0,-2,0,-2,0,0,2,
   -2,0,-2,0,0,2,2,0,-2,0,2,2,-2,-2,2,2,0,-2,-2,0,-2,-2,0,-1,-2,1,0,0,-1,0,0,
     2,0,2])
    d_lng = np.reshape(d_lng, (d_lng.size, 1))

    m_lng = np.concatenate(( np.array([0,0,0,0,1,0,1,0,0,-1]),np.zeros(17),np.array([2,0,2,1,0,-1,0,0,0,1,1,-1,0,
    0,0,0,0,0,-1,-1,0,0,0,1,0,0,1,0,0,0,-1,1,-1,-1,0,-1]) ))
    m_lng = np.reshape(m_lng, (m_lng.size, 1))

    mp_lng = np.array([0,0,0,0,0,1,0,0,1,0,1,0,-1,0,1,-1,-1,1,2,-2,0,2,2,1,0,0,-1,0,-1, 
   0,0,1,0,2,-1,1,0,1,0,0,1,2,1,-2,0,1,0,0,2,2,0,1,1,0,0,1,-2,1,1,1,-1,3,0])
    mp_lng = np.reshape(mp_lng, (mp_lng.size, 1))

    f_lng = np.array([0,2,2,0,0,0,2,2,2,2,0,2,2,0,0,2,0,2,0,2,2,2,0,2,2,2,2,0,0,2,0,0,
   0,-2,2,2,2,0,2,2,0,2,2,0,0,0,2,0,2,0,2,-2,0,0,0,2,2,0,0,2,2,2,2])
    f_lng = np.reshape(f_lng, (f_lng.size, 1))

    om_lng = np.array([1,2,2,2,0,0,2,1,2,2,0,1,2,0,1,2,1,1,0,1,2,2,0,2,0,0,1,0,1,2,1,
   1,1,0,1,2,2,0,2,1,0,2,1,1,1,0,1,1,1,1,1,0,0,0,0,0,2,0,0,2,2,2,2])
    om_lng = np.reshape(om_lng, (om_lng.size, 1))

    sin_lng = np.array([-171996, -13187, -2274, 2062, 1426, 712, -517, -386, -301, 217,
    -158, 129, 123, 63, 63, -59, -58, -51, 48, 46, -38, -31, 29, 29, 26, -22, 
     21, 17, 16, -16, -15, -13, -12, 11, -10, -8, 7, -7, -7, -7, 
     6,6,6,-6,-6,5,-5,-5,-5,4,4,4,-4,-4,-4,3,-3,-3,-3,-3,-3,-3,-3 ])
    sin_lng = np.reshape(sin_lng, (sin_lng.size, 1))
 
    sdelt = np.concatenate(( np.array([-174.2, -1.6, -0.2, 0.2, -3.4, 0.1, 1.2, -0.4, 0, -0.5, 0, 0.1,
     0,0,0.1, 0,-0.1]), np.zeros(10), np.array([-0.1, 0, 0.1]), np.zeros(33) ))
    sdelt = np.reshape(sdelt, (sdelt.size, 1))


    cos_lng = np.concatenate(( np.array([ 92025, 5736, 977, -895, 54, -7, 224, 200, 129, -95,0,-70,-53,0,
    -33, 26, 32, 27, 0, -24, 16,13,0,-12,0,0,-10,0,-8,7,9,7,6,0,5,3,-3,0,3,3,
     0,-3,-3,3,3,0,3,3,3]), np.zeros(14) ))
    cos_lng = np.reshape(cos_lng, (cos_lng.size, 1))

    cdelt = np.concatenate(( np.array([8.9, -3.1, -0.5, 0.5, -0.1, 0.0, -0.6, 0.0, -0.1, 0.3]),
     np.zeros(53) ))
    cdelt = np.reshape(cdelt, (cdelt.size, 1))
     
    n = len(jd)
    
    nut_long = np.zeros(n)
    nut_obliq = np.zeros(n)
    arg = d_lng.dot(d.T) + m_lng.dot(m.T) + mp_lng.dot(mprime.T) + f_lng.dot(f.T) + om_lng.dot(omega.T)

    sarg = np.sin(arg)
    carg = np.cos(arg)
    for i in range(n):
        nut_long[i]  = 1e-4*np.sum( (sdelt.flatten()*T[i] + sin_lng.flatten())*sarg[:,i].flatten())
        nut_obliq[i] = 1e-4*np.sum( (cdelt.flatten()*T[i] + cos_lng.flatten())*carg[:,i].flatten())
    
    return nut_long, nut_obliq
    
def _co_nutate(jd, ra, dec):
    d2r = np.pi/180.
    d2as = np.pi/(180.*3600.)
    T = (jd - 2451545.0)/36525.0 # Julian centures from J2000 of JD
    
    # must calculate obliquity of ecliptic 
    d_psi, d_eps = _nutate(jd)
    eps0 = 23.4392911*3600. - 46.8150*T - 0.00059*T**2 + 0.001813*T**3
    eps = (eps0 + d_eps)/3600.*d2r # true obliquity of the ecliptic in radians
    
    
    #useful numbers
    ce = np.cos(eps)
    se = np.sin(eps)

    # convert ra-dec to equatorial rectangular coordinates
    x = np.cos(ra*d2r) * np.cos(dec*d2r)
    y = np.sin(ra*d2r) * np.cos(dec*d2r)
    z = np.sin(dec*d2r)

    # apply corrections to each rectangular coordinate
    x2 = x - (y*ce + z*se)*d_psi * d2as
    y2 = y + (x*ce*d_psi - z*d_eps) * d2as
    z2 = z + (x*se*d_psi + y*d_eps) * d2as
    
    # convert back to equatorial spherical coordinates
    r = np.sqrt(x2**2 + y2**2 + z2**2)
    xyproj = np.sqrt(x2**2 + y2**2)

    ra2 = x2 * 0.
    dec2= x2 * 0.

    w1 = np.where( ((xyproj == 0) and (z != 0)) )[0]
    w2 = np.where(xyproj != 0)[0]
    
    if len(w1) > 0: 
	    #; places where xyproj=0 (point at NCP or SCP)
	    dec2[w1] = np.arcsin(z2[w1]/r[w1])
	    ra2[w1] = 0.
    if len(w2) > 0:
	    # places other than NCP or SCP
	    ra2[w2] = np.arctan2(y2[w2],x2[w2])
	    dec2[w2] = np.arcsin(z2[w2]/r[w2])

    ra2 = ra2 /d2r
    dec2 = dec2 /d2r

    ra2[ra2 < 0] += 360.
        
    d_ra = (ra2 - ra)*3600.
    d_dec = (dec2 - dec)*3600.
    
    return d_ra, d_dec
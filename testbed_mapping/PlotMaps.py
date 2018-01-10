import numpy as np
from matplotlib import pyplot
from astropy.io import fits
import sys
from astropy.wcs import WCS
import ConfigParser

def CalcLimits(m):
    gd = ~np.isnan(m)
    nCounts = float(len(m[gd]))

    ratio = 1.
    nbins = 10
    while ratio > 0.075:
        h, binedges = np.histogram(m[gd], bins=nbins)
        nbins += 1
        ratio = np.max(h/nCounts)

    peak = np.argmax(h)
    binmids = (binedges[1:] + binedges[:-1])/2.

    integral = 0.
    low = peak
    high = peak
    while integral < 0.95:
        if low == 0:
            low = 0
        else:
            low -= 1
        if high == h.size-1:
            high = h.size-1
        else:
            high += 1
        integral = np.sum(h[low:high])/nCounts

    vlow = binmids[low]
    vhigh= binmids[high]
    
    return vhigh, vlow


filename = sys.argv[1]

hdu = fits.open( filename)
wcs = WCS(hdu[0].header)





fig = pyplot.figure()

titles = ['Destriped Map', 'Average Map', 'Residual Map', 'Hit Map']
labels = ['V', 'V', 'V', 'Samples']
for i in range(4):

    vhigh, vlow = CalcLimits(hdu[i].data.flatten())

    pyplot.subplot(2,2,i+1, projection=wcs)
    pyplot.imshow(hdu[i].data, origin='lower', cmap=pyplot.cm.viridis, vmin=vlow, vmax=vhigh)

    if i >= 2:
        pyplot.xlabel('RA')
    else:
        pyplot.gca().coords[0].set_ticklabel_visible(False)
    if (i == 0) | (i == 2):
        pyplot.ylabel('Dec')
    else:
        pyplot.gca().coords[1].set_ticklabel_visible(False)

    pyplot.colorbar(label=labels[i])
    pyplot.title(titles[i])


pyplot.savefig('{}.png'.format(filename.split('.')[0]))

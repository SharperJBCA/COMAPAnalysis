import numpy as np 
from matplotlib import pyplot
import sys
from pipeline.Tools import FileTools
from scipy.interpolate import interp1d


filelist = sys.argv[1]
filenames = np.loadtxt(filelist, dtype='string', ndmin=1)

for i, filename in enumerate(filenames):
    d = FileTools.ReadH5Py(filename)

    if i == 0:
        m = d['maps']
        h = d['hits']
    else:
        m+= d['maps']
        h+= d['hits']

pyplot.imshow(np.mean(m[0,:,:,:,:],axis=(0,1))/h[0,:,:], vmax=1500,vmin=-1500)
pyplot.figure()
m2 = np.mean(m[1,:,:,:,:],axis=(0,1))/h[1,:,:]
#m2 = m2 - np.nanmean(m2[:,60:],axis=1)[:,np.newaxis]
pyplot.imshow(m2, vmax=1500,vmin=-1500)
pyplot.show()

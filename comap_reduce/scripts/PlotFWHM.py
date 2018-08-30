import numpy as np 
from matplotlib import pyplot
import sys
from pipeline.Tools import FileTools
from scipy.interpolate import interp1d


theta = np.pi/2.
Rot = np.array([[np.cos(theta), -np.sin(theta)],
                [-np.sin(theta),-np.cos(theta)]])

pixelOffsets = {0: [0, 0], # pixel 1
                1: Rot.dot(np.array([-65.00, 112.58])).flatten()} # pixel 12


def JupiterModel(nu):
    b = 2.14842271
    a = 0.1487
    
    z = a * np.log10(nu/22.8) + b

    return 10**z

k = 1.38e-23
c = 3e8
# Beam solid angle aken from James' report on wiki Optics
nubeam = np.array([26., 33., 40.])
srbeam = np.array([2.1842e-6, 1.6771e-6, 1.4828e-6])
beamModel = interp1d(nubeam, srbeam) # straight interpolation
beamModel = np.poly1d(np.polyfit(np.log(nubeam), srbeam, 2))

nujwmap = np.array([22.8, 33., 40.9, 61., 93.8])
Tjwmap  = np.array([135.2, 146.6, 154.7, 165., 172.3])
tjModel = np.poly1d(np.polyfit(np.log(nujwmap), Tjwmap, 2))
Oref = 2.481e-8 
rref = 5.2



filelist = sys.argv[1]
filenames = np.loadtxt(filelist, dtype='string', ndmin=1)

offsets = np.zeros((len(filenames), 2,  32*4))


labels = []
r0, d0 = 83.63308, 22.0145

amps = np.zeros((len(filenames), 32*4))
fwhm = np.zeros((len(filenames), 32*4))
date = np.zeros((len(filenames)))
rms = np.zeros((len(filenames), 32*4))
rms_Data = np.zeros((len(filenames), 32*4))

nu = np.zeros((32*4))
r = np.zeros((len(filenames)))


outputs = np.zeros((4,2,len(filenames)))
az, el = np.zeros(filenames.size), np.zeros(filenames.size)
horn = 0
hornID = [1,12]
markers = ['x','o']

axis = 0
fig = pyplot.figure()
for horn in [0,1]:
    for i, filename in enumerate(filenames):
        print(filename)
        d = FileTools.ReadH5Py(filename)
        P = d['TODFits']

        if i == 0:
            nu = np.tile(d['nu'], 4) + np.repeat(np.array([26.,28.,30.,32.]), 32)

        #for horn in [0,1]:
        print np.mean(np.mod(d['TODFits'][horn,:,:,6], np.pi*2.)*180./np.pi), np.std(d['TODFits'][horn,:,:,6])*180./np.pi
        amps[i,:] = d['TODFits'][horn,:,:,0].flatten()
        fwhm[i,:] = np.mean(d['TODFits'][horn,:,:,1+axis:2+axis],axis=-1).flatten()* 2.355* 60.
        rms[i,:] = np.mean(d['TODFitsErrors'][horn,:,:,1+axis:2+axis],axis=-1).flatten()* 2.355* 60
        rms_Data[i,:] = d['TODFits'][horn,:,:,-1].flatten()
         
        r[i] = d['Jupiter'][2]

        date[i] = d['mAzEl'][0,-1]
        el[i] = d['mAzEl'][horn,1]
        
    rms = np.sqrt(rms**2 + (fwhm * rms_Data/amps)**2)
    errFull = 1./np.sum(1./rms**2,axis=0)
    dataFull= np.sum(fwhm/rms**2,axis=0)*errFull

    for i in [0,2]:
        dataFull[i*32:(i+1)*32] = (dataFull[i*32:(i+1)*32])[::-1]
        errFull[i*32:(i+1)*32] = (errFull[i*32:(i+1)*32])[::-1]

    #pyplot.errorbar(nu,dataFull,yerr=np.sqrt(errFull),fmt='o', linewidth=3)

    fig.add_subplot(2,1,horn+1)
    pyplot.fill_between(nu, dataFull-np.sqrt(errFull), dataFull+np.sqrt(errFull), color='r', alpha=0.3)
    pyplot.plot(nu, dataFull)
    pyplot.ylabel('FWHM (arcmin)')
    pyplot.ylim(1.5,7.5)
    pyplot.text(0.1,0.1, 'Horn {}'.format(hornID[horn]), transform=pyplot.gca().transAxes, ha='left')
    pyplot.text(0.9,0.1, r'$\left< \mathrm{FWHM} \right>$:'+' {:.2f} +/- {:.2f} arcmin'.format(np.mean(dataFull),np.std(dataFull)), transform=pyplot.gca().transAxes, ha='right')

pyplot.xlabel('Frequency (GHz)')
pyplot.savefig('COMAP_FWHM_Axis{}.png'.format(axis), bbox_inches='tight')
pyplot.show()


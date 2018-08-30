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
date = np.zeros((len(filenames)))
rms = np.zeros((len(filenames), 32*4))
rms_Data = np.zeros((len(filenames), 32*4))

nu = np.zeros((32*4))
r = np.zeros((len(filenames)))


fwhm = np.zeros((4,32,2, 2, len(filenames)))
fwhmErrs = np.zeros((4,32,2, 2, len(filenames)))
outputs = np.zeros((4,2,len(filenames)))
az, el = np.zeros(filenames.size), np.zeros(filenames.size)
horn = 1
hornID = [1,12]
markers = ['x','o']
Gain = np.zeros((32*4, 2))
for horn in [0,1]:
    for i, filename in enumerate(filenames):
        print(filename)
        d = FileTools.ReadH5Py(filename)
        P = d['TODFits']

        if i == 0:
            nu = np.tile(d['nu'], 4) + np.repeat(np.array([26.,28.,30.,32.]), 32)

        #for horn in [0,1]:
        amps[i,:] = d['TODFits'][horn,:,:,0].flatten()
        rms[i,:] = d['TODFitsErrors'][horn,:,:,0].flatten()
        rms_Data[i,:] = d['TODFits'][horn,:,:,-1].flatten()
         
        r[i] = d['Jupiter'][2]
        
        date[i] = d['mAzEl'][0,-1]
        el[i] = d['mAzEl'][horn,1]

    #pyplot.plot(rms_Data.flatten(),',')
    #pyplot.show()
    rms = np.sqrt(rms**2)# + rms_Data**2)
    errFull = 1./np.nansum(1./rms**2,axis=0)
    dataFull= np.nanmean(amps,axis=0)#/rms**2,axis=0)*errFull

    for i in [0,2]:
        dataFull[i*32:(i+1)*32] = (dataFull[i*32:(i+1)*32])[::-1]
        errFull[i*32:(i+1)*32] = (errFull[i*32:(i+1)*32])[::-1]

    #pyplot.errorbar(nu,dataFull,yerr=np.sqrt(errFull),fmt='o', linewidth=3)

    #pyplot.fill_between(nu, dataFull-np.sqrt(errFull), dataFull+np.sqrt(errFull), color='r', alpha=0.3)
    #pyplot.plot(nu, dataFull)
    #pyplot.figure()

    Oj = Oref * (rref/r)**2
    print(r)
    Sjup = 2. * k * (tjModel(np.log(nu)) * (nu*1e9/c)**2)[np.newaxis,:] * Oj[:, np.newaxis] 
    print(Sjup)
    Tjup = Sjup * ((c/ nu/1e9)**2 / beamModel(np.log(nu)))[np.newaxis,:] / 2. / k  
    print(Tjup)
    Gain[:,horn] = dataFull/np.mean(Tjup,axis=0)

    pyplot.plot(nu, 10*np.log10(Gain[:,horn]), label='Pixel {}'.format(hornID[horn]))
    pyplot.ylabel('Gain (dB)')
pyplot.xlabel('Frequency (GHz)')
pyplot.savefig('COMAP_Gain_H{}.png', bbox_inches='tight')
pyplot.legend(loc='best')
pyplot.show()


np.savetxt('COMAP_Gain_05_05_2018.dat'.format(horn), np.concatenate((nu[:,np.newaxis], Gain[:, :]),axis=1) )



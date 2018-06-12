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


filelist = sys.argv[1]
filenames = np.loadtxt(filelist, dtype='string', ndmin=1)

offsets = np.zeros((len(filenames), 2,  32*4))


labels = []
r0, d0 = 83.63308, 22.0145

amps = np.zeros((len(filenames), 32*4))
rms = np.zeros((len(filenames), 32*4))
nu = np.zeros((32*4))
r = np.zeros((len(filenames)))


fwhm = np.zeros((4,32,2, 2, len(filenames)))
outputs = np.zeros((4,2,len(filenames)))
az, el = np.zeros(filenames.size), np.zeros(filenames.size)
horn = 1
hornID = [1,12]
markers = ['x','o']
for i, filename in enumerate(filenames):
    d = FileTools.ReadH5Py(filename)
    P = d['TODFits']

    for horn in [0,1]:
        amps[i,:] = d['TODFits'][horn,:,:,0].flatten()
        rms[i,:] = d['TODFits'][horn,:,:,-1].flatten()
              
        fwhm[:,:,:,horn, i] = d['TODFits'][horn,:,:,1:3]

        #print(d.keys())
        r[i] = d['Jupiter'][2]
        
        if i == 0:
            #print(d['nu'])
            nu = np.tile(d['nu'], 4) + np.repeat(np.array([26.,28.,30.,32.]), 32)
            #print(nu)
        #print(d['mAzEl'])
        azelMeasured = d['mAzEl'][horn,:2]
        azelTrue = d['mAzEl'][horn,2:4]

        dx = np.array([azelTrue[0] - azelMeasured[0]])
        dy = np.array([azelTrue[1] - azelMeasured[1]])
        outputs[0,horn,i] = dx*60. *np.cos(azelTrue[1]*np.pi/180.)
        outputs[1,horn,i] = dy*60.
        outputs[2,horn,i] = azelTrue[0]
        outputs[3,horn,i] = azelTrue[1]
        pyplot.scatter([outputs[0,horn,i]], [outputs[1,horn,i]] ,marker=markers[horn], c=np.array([azelMeasured[1]]), vmin=20, vmax=40)
pyplot.xlim(-25,25)
pyplot.ylim(-25,25)
pyplot.ylabel(r'$\Delta$El (arcmin)')
pyplot.xlabel(r'$\Delta$Az cos(El) (arcmin)')
pyplot.colorbar(label='Elevation')
pyplot.savefig('Pointing_AzEl_PreFit_El_H{}.png'.format(hornID[horn]))
pyplot.show()

fwhm[:,0:32,:] = (fwhm[:,0:32,:])[:,::-1,:]
fwhm[:,32*2:32*3,:] = (fwhm[:,32*2:32*3,:])[:,::-1,:]

r1 = np.sqrt(np.mean(fwhm[:,:,0,0,:],axis=-1).flatten()**2 + np.mean(fwhm[:,:,1,0,:],axis=-1).flatten()**2)*2.355*60.
#r1 = (np.mean(fwhm[:,:,0,0,:],axis=-1).flatten() + np.mean(fwhm[:,:,1,0,:],axis=-1).flatten())*2.355*60./2.

r12 = np.sqrt(np.mean(fwhm[:,:,0,1,:],axis=-1).flatten()**2 + np.mean(fwhm[:,:,1,1,:],axis=-1).flatten()**2)*2.355*60.
#r12 = (np.mean(fwhm[:,:,0,1,:],axis=-1).flatten() + np.mean(fwhm[:,:,1,1,:],axis=-1).flatten())*2.355*60./2.

pyplot.plot(nu, r1)
pyplot.plot(nu, r12)
pyplot.show()
for horn in [0,1]:
    ftext = open('1705_AzElOffsets_horn{}_Jupiter.dat'.format(hornID[horn]),'w')
    ftext.write('Offset Az (arcmin) - Offset El (arcmin) - True Az (degrees) - True El (degrees)' + '\n')

    for i in range(filenames.size):
        for val in outputs[:,horn,i]:
            ftext.write('{:.2f} '.format(val))
        ftext.write('\n')
    ftext.close()



# FIT FOR THE OFFSETS
from scipy.optimize import leastsq

    
def error(P, x, y):

    m = P[0] * Rot.dot(np.array(x)).flatten()
    rmdl = np.sqrt(m[0]**2 + m[1]**2)
    rsky = np.sqrt(y[0,:]**2 + y[1,:]**2)
    
    resid = rmdl- rsky
    return resid.flatten()

p = 0.1853
P0 = [p]

P1, cov_x, info, mesg, s = leastsq(error, P0, args=([-65., 112.58], outputs[:2,1,:]), full_output=True)
pOff = P1[0] * Rot.dot(np.array([-65., 112.58])) 
resid = np.sum(error(P1,[-65., 112.58], outputs[:2,1,:])**2)/outputs.shape[2]


def MCErrors():

    niter = outputs.shape[2]
    Pout = np.zeros(niter)
    s = np.arange(outputs.shape[2]).astype('int')
    for i in range(niter):
        np.random.shuffle(s)
        data = outputs[:2,1,s[:s.size-1]]*1.

        Pout[i], cov_x, info, mesg, si = leastsq(error, P0, args=([-65., 112.58], data + np.random.normal(size=data.shape, scale=np.abs(data)*0.01)), full_output=True)

    return np.std(Pout)

np.std(pOff[:,np.newaxis]-outputs[:2,1,:])**2
print np.sqrt(cov_x*resid), MCErrors()

print P1

#theta = P1[3] # np.pi/2.
#Rot = np.array([[np.cos(theta), -np.sin(theta)],
#                [-np.sin(theta),-np.cos(theta)]])

#pixelOffsets[horn]
pyplot.plot((pOff[:,np.newaxis]-outputs[:2,1,:]).T,'o')
#pyplot.plot(outputs[:2,1,:].T,'o')
pyplot.show()

 
print (outputs[:2,1,:]-pOff[:,np.newaxis]).T
#print(P1)

print pOff


for i, filename in enumerate(filenames):
    d = FileTools.ReadH5Py(filename)
    P = d['TODFits']

    for horn in [0,1]:
        amps[i,:] = d['TODFits'][horn,:,:,0].flatten()
        rms[i,:] = d['TODFits'][horn,:,:,-1].flatten()
              
        #print(d.keys())
        r[i] = d['Jupiter'][2]
        
        if i == 0:
            #print(d['nu'])
            nu = np.tile(d['nu'], 4) + np.repeat(np.array([26.,28.,30.,32.]), 32)
            #print(nu)
        #print(d['mAzEl'])
        azelMeasured = d['mAzEl'][horn,:2]
        azelTrue = d['mAzEl'][horn,2:4]
        if horn == 1:
            azelMeasured[0] += pOff[0]/60./np.cos(azelTrue[1]*np.pi/180.)
            azelMeasured[1] += pOff[1]/60.
        if horn == 0:
            azelMeasured[0] += pixelOffsets[horn][0]/60.*P1[0]
            azelMeasured[1] += pixelOffsets[horn][1]/60.*P1[0]



        dx = np.array([azelTrue[0] - azelMeasured[0]])
        dy = np.array([azelTrue[1] - azelMeasured[1]])
        outputs[0,horn,i] = dx*60. *np.cos(azelTrue[1]*np.pi/180.)
        outputs[1,horn,i] = dy*60.
        if horn == 1:
            print outputs[:2,horn,i]
        outputs[2,horn,i] = azelTrue[0]
        outputs[3,horn,i] = azelTrue[1]
        pyplot.scatter([outputs[0,horn,i]], [outputs[1,horn,i]] ,marker=markers[horn], c=np.array([azelMeasured[1]]), vmin=20, vmax=40)
pyplot.xlim(-25,25)
pyplot.ylim(-25,25)
pyplot.ylabel(r'$\Delta$El (arcmin)')
pyplot.xlabel(r'$\Delta$Az cos(El) (arcmin)')
pyplot.colorbar(label='Elevation')
pyplot.savefig('Pointing_AzEl_PostFit_El_H{}.png'.format(hornID[horn]))
pyplot.show()

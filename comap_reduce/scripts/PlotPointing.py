import numpy as np 
from matplotlib import pyplot
import sys
from pipeline.Tools import FileTools
from scipy.interpolate import interp1d

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

outputs = np.zeros((4,len(filenames)))
az, el = np.zeros(filenames.size), np.zeros(filenames.size)
horn = 0
hornID = [1,12]
for i, filename in enumerate(filenames):
    print(filename)
    d = FileTools.ReadH5Py(filename)
    P = d['TODFits']

    amps[i,:] = d['TODFits'][horn,:,:,0].flatten()
    rms[i,:] = d['TODFits'][horn,:,:,-1].flatten()
              
    print(d.keys())
    r[i] = d['Jupiter'][2]

    if i == 0:
        print(d['nu'])
        nu = np.tile(d['nu'], 4) + np.repeat(np.array([26.,28.,30.,32.]), 32)
        print(nu)
    print(d['mAzEl'])
    azel = d['mAzEl'][horn,:2]
    print(azel)
    offsets[i,0, :] = d['TODFits'][horn,:,:,4].flatten()
    offsets[i,1, :] = d['TODFits'][horn,:,:,5].flatten()
    az[i] = azel[0]
    el[i] = azel[1]

pyplot.scatter(offsets[:,0,:].flatten()*60., offsets[:,1,:].flatten()*60,marker='x', c=np.repeat(el,32*4))
pyplot.xlim(-6,6)
pyplot.ylim(-6,6)
pyplot.ylabel(r'$\delta \Delta$ (arcmin)')
pyplot.xlabel(r'$\delta \alpha$ (arcmin)')
pyplot.colorbar(label='Elevation')
pyplot.savefig('Pointing_El_H{}.png'.format(horn))
pyplot.show()

ftext = open('1705_horn{}_JupiterOffsets.dat'.format(hornID[horn]),'w')
ftext.write('Offset RA (arcmin) - Offset Dec (arcmin) - Mean Az (degree) - Mean El (degree)' + '\n')

outputs = np.zeros((4, filenames.size))
outputs[:2,:] = np.mean(offsets,axis=2).T
outputs[2,:] = az
outputs[3,:] = el
for i in range(filenames.size):
    for val in outputs[:,i]:
        ftext.write('{:.2f} '.format(val))
    ftext.write('\n')
ftext.close()

amps[:,0:32] = (amps[:,0:32])[:,::-1]
amps[:,32*2:32*3] = (amps[:,32*2:32*3])[:,::-1]
rms[:,0:32] = (rms[:,0:32])[:,::-1]
rms[:,32*2:32*3] = (rms[:,32*2:32*3])[:,::-1]

snr = np.mean(amps,axis=0)/np.mean(rms,axis=0)

pyplot.imshow(amps,aspect='auto', extent=[np.min(nu), np.max(nu), 0, rms.shape[1]])
pyplot.show()
pyplot.imshow(rms,aspect='auto', extent=[np.min(nu), np.max(nu), 0, rms.shape[1]])
pyplot.show()

pyplot.plot(nu, np.mean(amps,axis=0)/np.mean(rms,axis=0))
pyplot.show()

pyplot.plot(el, np.average(amps, weights=snr**2, axis=1),'o')
pyplot.show()

# Beam calc

# Beam solid angle aken from James' report on wiki Optics
nubeam = np.array([26., 33., 40.])
srbeam = np.array([2.1842e-6, 1.6771e-6, 1.4828e-6])
beamModel = interp1d(nubeam, srbeam) # straight interpolation
beamModel = np.poly1d(np.polyfit(np.log(nubeam), srbeam, 2))

print(beamModel)

pyplot.plot(nubeam, srbeam,'o')
pyplot.plot(nu, beamModel(np.log(nu)))
pyplot.show()

nujwmap = np.array([22.8, 33., 40.9, 61., 93.8])
Tjwmap  = np.array([135.2, 146.6, 154.7, 165., 172.3])
tjModel = np.poly1d(np.polyfit(np.log(nujwmap), Tjwmap, 2))


pyplot.plot(nujwmap,  JupiterModel(nujwmap))
pyplot.plot(nujwmap,  tjModel(np.log10(nujwmap)))
pyplot.plot(nujwmap, Tjwmap,'x')
pyplot.show()
dnu = 2./32. # GHz
tau  = 0.02 # seconds

k = 1.38e-23
c = 3e8

Oref = 2.481e-8 
rref = 5.2
Oj = Oref * (rref/r)**2
print(r)
Sjup = 2. * k * (tjModel(np.log(nu)) * (nu*1e9/c)**2)[np.newaxis,:] * Oj[:, np.newaxis] 
print(Sjup)
Tjup = Sjup * ((c/ nu/1e9)**2 / beamModel(np.log(nu)))[np.newaxis,:] / 2. / k  
print(Tjup)
Tsys = np.mean(rms,axis=0)*np.sqrt(dnu * tau*1e9) / np.mean(amps,axis=0) * np.mean(Tjup,axis=0)

# pyplot.plot(nu, np.mean(Sjup,axis=0)*1e26)
# pyplot.ylabel(r'$S_j$ (Jy)')
# pyplot.xlabel('Frequency (GHz)')
# pyplot.title('Jupiter flux at {:.1f} AU'.format(np.mean(r)))
# pyplot.savefig('JupiterFlux.png', bbox_inches='tight')
# pyplot.show()

# pyplot.plot(nu, np.mean(Tjup,axis=0))
# pyplot.ylabel(r'$T_j$ (K)')
# pyplot.xlabel('Frequency (GHz)')
# pyplot.title('Jupiter Antenna Temperature at {:.1f} AU'.format(np.mean(r)))
# pyplot.savefig('JupiterTa.png', bbox_inches='tight')
# pyplot.show()

# pyplot.plot(nu, tjModel(np.log(nu)))
# pyplot.ylabel(r'$T_j$ (K)')
# pyplot.xlabel('Frequency (GHz)')
# pyplot.title('Jupiter Brightness Temperature Model')
# pyplot.savefig('JupiterTj.png', bbox_inches='tight')
# pyplot.show()

# pyplot.plot(nu, beamModel(np.log(nu))*1e6)
# pyplot.ylabel(r'$\Omega_\mathrm{beam}$ ($\mu$Sr)')
# pyplot.xlabel('Frequency (GHz)')
# pyplot.title('Beam Model Solid Angle')
# pyplot.savefig('BeamModel.png', bbox_inches='tight')
# pyplot.show()

#stop
pyplot.plot(nu, Tsys)
pyplot.ylabel(r'$T_\mathrm{sys}$ (K)')
pyplot.xlabel('Frequency (GHz)')
pyplot.title('Horn {}'.format(hornID[horn]))
pyplot.savefig('Tsys_H{}.png'.format(hornID[horn]))
pyplot.show()

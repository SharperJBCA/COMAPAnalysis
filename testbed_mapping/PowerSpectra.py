import scipy.fftpack as sfft
from scipy.signal import hamming
from astropy.io import fits
import numpy as np

def PowerSpectrum(d, stepLen, iStart, iTime, sr):
    '''
    d - input data
    stepLen - Size of step between each power spectrum
    iStart  - Initial offset in data (e.g., ignore initial zero values
    iTime   - Length of time to used to integrate spectrum
    sr      - Sample rate (Hz)
    '''

    nSteps = (d.size - iStart)//stepLen
    steps = [[int(iStart + stepLen*i), int(iStart + stepLen*i + iTime)] for i in range(nSteps)]

    ps = np.zeros(iTime)
    freqs = sfft.fftfreq(ps.size, d=1./sr)

    for s in steps:

        ps += np.abs(sfft.fft( (d[s[0]:s[1]]-np.mean(d[s[0]:s[1]]))*hamming(d[s[0]:s[1]].size)  ))**2/ nSteps

    return freqs, ps
    
# sr = 10
# elLen = 4*60*sr
# i0 = 200*sr
# intTime = 3*60*sr
# sel = 5*sr

# elSteps = [[int(i0 + elLen*i + sel*i), int(i0 + elLen*i + sel*i + intTime)] for i in range(120)]

# sr = 1./20e-3
# elLen = 4*60*sr
# i0 = 200*sr
# intTime = 3*60*sr
# sel = 5*sr

# todSteps = [[int(i0 + elLen*i + sel*i), int(i0 + elLen*i + sel*i + intTime)] for i in range(120)]

# tod_ps = np.zeros(todSteps[0][1]-todSteps[0][0])
# ze_ps = np.zeros(elSteps[0][1]-elSteps[0][0])
# ze = 1./np.sin(el*np.pi/180.)

# for s, t in zip(elSteps,todSteps) :

#     ze_ps += np.abs(sfft.fft( (ze[s[0]:s[1]]-np.mean(ze[s[0]:s[1]]))*hamming(ze[s[0]:s[1]].size)  ))**2/ len(elSteps)
#     ze_freqs = sfft.fftfreq(ze_ps.size, d=0.1)

#     tod_ps += np.abs(sfft.fft( (tod[t[0]:t[1]]-np.mean(tod[t[0]:t[1]]))*hamming(tod[t[0]:t[1]].size) ))**2 / len(elSteps)
#     tod_freqs= sfft.fftfreq(tod_ps.size, d=20e-3)

# zmax = np.argmax(ze_ps[:ze_ps.size//2])
# tmax = np.argmax(tod_ps[:tod_ps.size//2])

# print('Frequency:', ze_freqs[zmax], tod_freqs[tmax], ze_freqs[zmax]/ tod_freqs[tmax])

# pyplot.plot(ze_freqs[:ze_ps.size//2], (ze_ps/np.median(ze_ps))[:ze_ps.size//2], label='Elevation')
# pyplot.plot(tod_freqs[:tod_ps.size//2], (tod_ps/np.median(tod_ps))[:tod_ps.size//2], label='Receiver')
# pyplot.yscale('log')
# pyplot.xscale('log')
# pyplot.xlabel('Frequency (Hz)')
# pyplot.ylabel(r'Power')
# pyplot.title('comap_ncp_1264')
# pyplot.legend(loc='best')
# pyplot.show()

# pyplot.subplot(2,1,1)
# pyplot.plot(jd - todjd0, (ze-np.nanmean(ze))/np.nanstd(ze))
# pyplot.plot(todjd - todjd0, (tod-np.nanmean(tod))/np.nanstd(tod))
# pyplot.xlabel('Time (jd-jd0)')
# pyplot.ylabel(r'$\frac{T-\left< T \right> }{\sigma}$')
# pyplot.xlim(0.116,0.117)
# pyplot.ylim(1,1.4)
# pyplot.title('comap_ncp_1264')
# pyplot.subplot(2,1,2)
# pyplot.plot(jd - todjd0, (ze-np.nanmean(ze))/np.nanstd(ze))
# pyplot.plot(todjd - todjd0, (tod-np.nanmean(tod))/np.nanstd(tod))
# pyplot.xlabel('Time (jd-jd0)')
# pyplot.ylabel(r'$\frac{T-\left< T \right> }{\sigma}$')
# pyplot.xlim(0.2859, 0.2865)
# pyplot.ylim(0.1, 0.4)
# pyplot.tight_layout()
# pyplot.show()

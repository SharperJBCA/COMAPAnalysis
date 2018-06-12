import numpy as np

def SimpleRemoval(tod, el, stride):
    """
    Removes the atmospheric fluctuations in place
    
    returns the fitted amplitudes for the atmosphere
    """

    if stride < 0:
        stride = tod.shape[3]
    nHorns = tod.shape[0]
    nSidebands = tod.shape[1]
    nChans = tod.shape[2]


    nSteps = tod.shape[-1]//stride
    Aout = np.zeros((nHorns, nSidebands, nChans, nSteps))


    for i in range(nSteps):
        lo = i*stride
        hi = (i+1)*stride
        for j in range(nHorns):
            # Simple atmosphere removal:
            A = 1./np.sin(el[j,:]*np.pi/180.)

            for k in range(nSidebands):
                for l in range(nChans):
                    pmdl = np.poly1d( np.polyfit(A, tod[j,k,l,lo:hi], 1))
                    tod[j,k,l,lo:hi] -= pmdl(A)
                    Aout[j,k,l,i] = pmdl[1]

    return Aout

import numpy as np

from matplotlib import pyplot

def MedianFilter(tod, stride):
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


    for i in range(nSteps):
        lo = i*stride
        hi = (i+1)*stride
        for j in range(nHorns):
            # Simple atmosphere removal:
            pmdl = np.nanmedian(tod[j,:,:,lo:hi],axis=2)
            tod[j,:,:,lo:hi] -= pmdl[:,:, np.newaxis]

            #for k in range(nSidebands):
            #    pmdl = np.nanmedian(tod[j,k,:,lo:hi],axis=1)
            #    tod[j,k,:,lo:hi] -= pmdl[:, np.newaxis]

                #for l in range(nChans):
                #    tod[j,k,l,lo:hi] -= pmdl
                

def AverageFreqs(tod, nu, stride, badChans = []):

    nHorns     = tod.shape[0]
    nSidebands = tod.shape[1]
    nChans     = tod.shape[2]
    nSamps     = tod.shape[3]

    nSteps = int(nChans/stride)

    # We average the remaining channels into the last channel? Trim?
    # Trim for now so that the frequency widths are evenly spaced
    if len(badChans) > 0:
        tod[:,:,badChans,:] = np.nan
        nu[badChans] = np.nan

    outTod = np.reshape(tod[:,:,:nSteps*stride,:], (nHorns, nSidebands, nSteps, stride, nSamps))
    
    outnu = np.reshape(nu[:nSteps*stride], (nSteps, stride))

    #t = np.arange(stride)
    #pyplot.plot(t, outTod[0,0,10,:,0])
    #pyplot.plot(t+stride, outTod[0,0,11,:,0])
    #pyplot.show()

    return np.nanmean(outTod, axis=3), np.nanmean(outnu, axis=1)
        
        

        

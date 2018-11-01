import numpy as np
from matplotlib import pyplot
import configparser
import sys
from chistogram import histogram, histogram_weights, unwrap, dotcovar
from scipy import linalg as la

from CG import CG

def MakeLong(d):
    """
    Makes the to row of a toeplitz matrix into the top row of a larger circulant matrix.
    """
    N = d.size
    dlong = np.zeros(N*2)
    dlong[:N] = d
    dlong[N+1:] = (d[::-1])[:-1]
    return dlong


class DataClass(object):
    """
    This should contain the data relevant to each node.
    
    It should also contain short cuts to the relevant destriper functions
    """
    def __init__(self):
        pass
        
    # READ IN DATA
    def setTOD(self, tod):
        self.tod = tod.astype(float)
        self.weights = np.ones(tod.size, dtype=float)
        
    def setPixels(self, pixels):
        self.pixels = pixels.astype(int)
        
    def setOffsets(self, offsets):
        self.offsets = offsets.astype(int)
        
    def setData(self, tod, pixels, offsets, npix, noffsets, offsetTime, noiseParams=None):
        
        # Set the data
        self.npix = int(npix)
        self.noffsets = int(noffsets)
        
        self.setTOD(tod)
        self.setPixels(pixels)
        self.setOffsets(offsets)
        self.offsetTime = offsetTime
        
        # Need some product containers
        self.sumOffsets = np.zeros(self.noffsets, dtype=float)
        self.weiOffsets = np.zeros(self.noffsets, dtype=float)
        self.outOffsets = np.zeros(self.noffsets, dtype=float)
        
        self.sumMap = np.zeros(self.npix, dtype=float)
        self.weiMap = np.zeros(self.npix, dtype=float)
        self.outMap = np.zeros(self.npix, dtype=float)
        
        # Need some containers to store offsets solutions + b-vector
        self.bVec = np.zeros(self.noffsets, dtype=float)
        self.a0   = np.zeros(self.noffsets, dtype=float)
        self.AxVec = np.zeros(self.noffsets, dtype=float)
        
        # Also need some unwrapped data containers
        self.unwrapped = np.zeros(self.tod.size, dtype=float)
        self.unwrapped2 = np.zeros(self.tod.size, dtype=float)
        
        
        if isinstance(noiseParams, type(None)):
            self.useCovar = False
            
        else:
            sigma, fknee, alpha, offsetTime = noiseParams
            self.offsetTime = offsetTime
            self.sigma, self.fknee, self.alpha = sigma, fknee, alpha
            f = np.fft.fftfreq(self.noffsets, d=offsetTime)
            psNoise = sigma**2*( (fknee/np.abs(f))**alpha)
            psNoise[0] = psNoise[1]
            #psNoise *= 2./self.offsetTime
            psNoise = psNoise + 0j
            #acf = np.real(np.fft.fft(psNoise))
            
            self.noiseLong = psNoise*1e6
            
            #f = np.fft.fftfreq(100, d=offsetTime)
            #p#sNoise = sigma**2*(1. + (fknee/np.abs(f))**alpha)
            #psNoise[0] = psNoise[1]
            #self.ps2 = np.real(np.fft.fft(psNoise))
            
            #self.noiseLong /= self.noiseLong[0]
            
            #C = la.toeplitz(acf)
            #x = np.arange(10)
            #b = C.dot(x[:,np.newaxis]).flatten()
            #blong = np.zeros(b.size*2)
            
            
            #print(b.shape, self.noiseLong.shape, acf.shape, MakeLong(acf).shape)
            #blong = np.concatenate((b,b))*MakeLong(acf)
            #xout = np.fft.irfft(np.fft.rfft(blong)/self.noiseLong)
            #print(xout)
            #print(x)
            #pyplot.imshow(C)
            #pyplot.show()
            
            #print(self.noiseLong.size)
            #stop
            self.useCovar = True
            
            
    # DESTRIPING FUNCTIONS
    def SumTo(self, d, bins, output, weights=None):
        
        if isinstance(weights, type(None)):
            histogram(bins, d, output)
        else:
            histogram_weights(bins, d,  weights, output)
            
    def Unwrap(self, d, bins, output):
        unwrap(bins, d, output)
                    
    def b(self, d):
        """
        For solving Ax = b
        
        This creates the vector b
        """
        
        # This is the Z operation
        self.sumMap *= 0.
        self.weiMap *= 0.

        self.SumTo(d, self.pixels, self.sumMap, self.weights)
        self.SumTo(self.weights, self.pixels, self.weiMap)
        self.outMap *= 0.
        self.outMap += self.sumMap
        self.outMap /= self.weiMap
        self.outMap[np.isnan(self.outMap)] = 0.
        
        self.unwrapped *= 0.
        self.Unwrap(self.outMap, self.pixels, self.unwrapped)
        self.unwrapped *= -1
        self.unwrapped += d
        
        # Then weighted binning to Offsets
        self.bVec *= 0.
        self.SumTo(self.unwrapped, self.offsets, self.bVec, self.weights)
        
    def Ax(self, d):
        """
        For solving Ax = b
        
        This creates the vector Ax
        """
        
        self.unwrapped2 *= 0.
        self.Unwrap(d, self.offsets, self.unwrapped2)
        
        # This is the Z operation
        self.sumMap *= 0.
        self.weiMap *= 0.
        self.SumTo(self.unwrapped2, self.pixels, self.sumMap, self.weights)
        self.SumTo(self.weights, self.pixels, self.weiMap)
        self.outMap *= 0.
        self.outMap += self.sumMap
        self.outMap /= self.weiMap
        self.outMap[np.isnan(self.outMap)] = 0.

        
        self.unwrapped *= 0.
        self.Unwrap(self.outMap, self.pixels, self.unwrapped)
        self.unwrapped *= -1
        self.unwrapped += self.unwrapped2
    
        # Then weighted binning to Offsets
        self.AxVec *= 0.
        self.SumTo(self.unwrapped, self.offsets, self.AxVec, self.weights)
    
        # if using noise covariance
        if self.useCovar:
            ratio = np.fft.fft(d)/self.noiseLong
            ratio[np.isnan(ratio) | np.isinf(ratio)] = 0
            #print(ratio)
            #print(np.sum(ratio), np.sum(self.noiseLong))
            x = np.real(np.fft.ifft(ratio))
            #pyplot.plot(self.unwrapped)
            #print(x)
            x[np.isnan(x)] = 0.
            print('Sum of coefficients', np.sum(d*x))
            print(np.std(x), np.std(self.a0))
            #C = np.zeros((100,100))
            #for i in range(3):
            #    C[:,:] += self.a0[i*100:(i+1)*100, np.newaxis].dot(self.a0[np.newaxis,i*100:(i+1)*100])/30.

            #c = np.zeros(100)
            #for i in range(c.size):
            #    c[:] += np.roll(C[i,:],-i)/float(c.size)
                
            #f = np.fft.fftfreq(self.noffsets, d=self.offsetTime)
            #psNoise = self.sigma**2*(1. + (self.fknee/np.abs(f))**self.alpha)
            #psNoise[0] = psNoise[1]
            #psNoise = psNoise  #/ self.noffsets
            
            #psNoise = psNoise*1. + 0j
            #pyplot.plot(f, np.abs(np.fft.fft(self.a0))**2 )
            #pyplot.plot(f, np.real(psNoise)/self.offsetTime*2.)
            #pyplot.plot(f, self.noiseLong)
            #pyplot.yscale('log')
            #pyplot.xscale('log')
            #pyplot.plot(self.ps2/np.std(self.ps2)*np.std(c))
            #pyplot.figure()
            #pyplot.plot(c)
            #pyplot.plot(np.real(np.fft.ifft(np.fft.fft(c))))
            #pyplot.plot(np.real(np.fft.ifft(psNoise)))
            
            #pyplot.show()
            #pyplot.plot(self.unwrapped2)#/np.std(self.unwrapped2))
            #pyplot.plot(self.unwrapped)#/np.std(self.unwrapped))
            
            #pyplot.show()

            #t#emp = np.zeros(self.AxVec.size, dtype=float)
            #temp2 = self.AxVec*1.
            self.unwrapped2 *= 0.
            self.Unwrap(x, self.offsets, self.unwrapped2)
            self.SumTo(self.unwrapped2, self.offsets, self.AxVec, self.weights)
            #self.SumTo(self.unwrapped2, self.offsets, temp, self.weights)
            
            #pyplot.plot(temp2)
            #pyplot.plot(temp,alpha=0.25)
            #pyplot.plot(self.AxVec-temp,alpha=0.25)
            #pyplot.show()

    
    def ReturnMap(self, d= None):
        self.sumMap *= 0.
        self.weiMap *= 0.
        if isinstance(d, type(None)):
            self.SumTo(self.tod, self.pixels, self.sumMap, self.weights)
        else:
            self.SumTo(d, self.pixels, self.sumMap, self.weights)
                
        self.SumTo(self.weights, self.pixels, self.weiMap)
        self.outMap *= 0.
        self.outMap += self.sumMap
        self.outMap /= self.weiMap
        self.outMap[np.isnan(self.outMap)] = 0.

        return self.outMap*1.
        
    def ReturnDestripedMap(self):
    
        self.unwrapped *= 0.
        self.Unwrap(self.a0, self.offsets, self.unwrapped)
        self.sumMap *= 0.
        self.weiMap *= 0.
        self.SumTo(self.unwrapped, self.pixels, self.sumMap, self.weights)
        self.SumTo(self.weights, self.pixels, self.weiMap)
        self.outMap *= 0.
        self.outMap += self.sumMap
        self.outMap /= self.weiMap
        self.outMap[np.isnan(self.outMap)] = 0.

        return self.outMap*1.
        
    def EstimatePower(self):
        
        f = np.fft.fftfreq(self.noffsets, d=self.offsetTime)
        ps = np.abs(np.fft.fft(self.a0))**2
        
        fEdges = 10**np.linspace(np.log10(f[1]), np.log10(np.max(f)),20)
        fMids = (fEdges[1:] + fEdges[:-1])/2.
        
        z, h = np.histogram(f[1:f.size//2], fEdges, weights=ps[1:f.size//2])
        w, h = np.histogram(f[1:f.size//2], fEdges)
        
        psbin = z/w
        
        pyplot.plot(f[1:f.size//2], ps[1:f.size//2])
        pyplot.plot(fMids, psbin,'o')
        pyplot.yscale('log')
        pyplot.xscale('log')
        pyplot.show()
        
        
    
class DataTest(object):
    def __init__(self):
        self.tod = None
        self.a0 = np.zeros(4)
        self.AxVec = np.zeros(4)
        #self.bVec = np.array([2., -8.])
        self.xVec = np.array([1., -8., 3., 4.])
        
        #self.A = np.array([[3.,2.],[2.,6.]])
        self.A = np.random.uniform(size=(4,4))
        self.A = self.A + self.A.T + 4*np.identity(4)
        #np.array([[3.,2.,5.,8.],
        #                   [2.,6.,3.,5.],
        #                   [1.,3.,3.,5.],
        #                   [2.,5.,3.,5.]])
        print(la.eigh(self.A))
        self.bVec = self.A.dot(self.xVec[:,np.newaxis]).flatten()
        
    def b(self, dumb):
        
        self.bVec  *= 1.
        
    def Ax(self, r):
            
        self.AxVec = self.A.dot(r[:,np.newaxis]).flatten()

if __name__ == "__main__":
    
    # Read in parameter file
    #inputParameters = sys.argv[1]
    #Parameters = ConfigParser.ConfigParser()
    #Parameters.read(inputParameters)
    import sys
    import h5py
    filename = sys.argv[1]
    d = h5py.File(filename)
    tod = d['TOD'][...].flatten().astype(float)
    tod -= np.median(tod)
    
    ra = d['RA'][...].flatten()
    dec = d['DEC'][...].flatten()
    d.close()

    # Setup some basic mapping here?

#Standard modules:
import numpy as np
try:
    from mpi4py import MPI
    f_found=True
    import MPI_tools
except ImportError:
    f_found=False

from CGM import CGM
from MPIFuncs import MPISum2Root, MPIRoot2Nodes
import binning


#Destriper modules:
from DesFuncs import bFunc,AXFunc

class Destriper:

    def __init__(self, tod, weights, pix, baselines, npix, maxiter=300, Nodes=None, threads=1):
        """
        Arguments
        tod -- Time-ordered single-dish radio telescope scanning data. (Array)
        weights -- Weight for each tod sample.
        pix -- Time-ordered pixel coordinate data. (Array)
        bl  -- Length of baselines used to Destripe the tod. (Integer)
        npix -- Total number of pixels for output map.

        Setup data containers
        """
        
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        self.threads = threads


        # Make an array of prior 'long' baselines from medians
        self.nBaselines     = int(np.max(baselines)) + 1

        # Make sure everything is the correct type!
        self.tod = tod.astype('d')
        self.weights = weights.astype('d')
        self.a0 = np.zeros(self.nBaselines).astype('d') #Variations around zero

        self.FtZd = np.zeros(self.nBaselines).astype('d') #Variations around zero
        self.FtZa = np.zeros(self.nBaselines).astype('d') #Variations around zero
        self.Ft_local = np.zeros((self.threads, self.nBaselines)).astype('d') #Variations around zero


        self.pix = pix.astype('int')
        self.npix= int(npix)
        self.blpix = baselines # get our mapping for baselines
        self.size = tod.size
        self.maxiter = int(maxiter)

        # Setup the maps
        self.fullmap = np.zeros(npix, dtype='d')
        self.varmap  = np.zeros(npix, dtype='d')
        self.sw   = np.zeros(npix, dtype='d')
        self.w    = np.zeros(npix, dtype='d')
        self.hits = np.zeros(npix, dtype='d')
        self.m_local   = np.zeros((self.threads, npix), dtype='d')

        if rank == 0:
            self.swroot  = np.zeros(npix, dtype='d')
            self.wroot  = np.zeros(npix, dtype='d')
        else:
            self.swroot = None
            self.wroot  = None

        # Setup nodes
        if isinstance(Nodes,type(None)):
            self.Nodes = np.arange(size, dtype='i')
        else:
            self.Nodes = Nodes.astype('i')
        
        # Bin edges for making maps
        #self.mapbins = np.arange(0, npix+1)
        #self.hits = np.histogram(self.pix, bins=self.mapbins)[0]
        #self.w = np.histogram(self.pix, bins=self.mapbins, weights=self.weights)[0]
        self.m_local *= 0.
        binning.weight(self.weights, self.pix, self.w, self.m_local, self.threads)

        self.m_local *= 0.
        binning.hits(self.pix, self.hits, self.m_local, self.threads)

        self.goodpix = (self.w != 0)

        # Baseline bin edges
        #self.offsetbins = np.arange(0, self.nBaselines+1)
        #self.offsetindex= np.repeat(self.offsetbins, self.bl)[:self.tod.size]
        
    def BinMap(self, d, w):
        # Calculate first the signal weight map
        self.sw *= 0.
        self.m_local *= 0.
        binning.histweight(d, w, self.pix, self.sw, self.m_local, self.threads)

        # then the full sky map
        self.fullmap *= 0.
        self.fullmap[self.goodpix] = self.sw[self.goodpix]/self.w[self.goodpix]
        self.fullmap[np.isnan(self.fullmap)] = 0.

    def BinVarMap(self, d, w):
        # Calculate first the signal weight map
        self.sw *= 0.
        self.m_local *= 0.
        binning.histweight(d, w, self.pix, self.sw, self.m_local, self.threads)

        # then the full sky map
        self.fullmap *= 0.
        self.fullmap[self.goodpix] = self.sw[self.goodpix]/self.w[self.goodpix]
        self.fullmap[np.isnan(self.fullmap)] = 0.

        # variance map
        self.sw *= 0.
        self.m_local *= 0.
        binning.histweight(d**2, w, self.pix, self.sw, self.m_local, self.threads)
        self.varmap *= 0.
        self.varmap[self.goodpix] = self.sw[self.goodpix]/self.w[self.goodpix] - self.fullmap[self.goodpix]**2
        self.varmap[np.isnan(self.fullmap)] = 0.


    def FtC(self, d, w, Ft, index1, index2):
        # Calculate first the signal weight map
        #ft_bin = np.histogram(self.offsetindex, bins=self.offsetbins, weights=d*w)[0]
        self.Ft_local *= 0.

        binning.FtC(d, w, Ft, self.Ft_local, index1, index2, self.threads)
        #return self.Ft

    def GuessA0(self):
        # Loop through median values of tod and estimate initial baseline values
        #binning.medians(self.tod, self.a0, self.bl)
        #self.a0 = np.median(np.reshape(self.tod, (self.nBaselines, self.bl)), axis=1)
        self.a0[:] = np.median(self.tod)
        
        #for i in range(self.nBaselines): 
        #    self.a0[i] = np.median(self.tod[i*self.bl:(i+1)*self.bl])





#------------------------------------------------------------------------#
#----------------------DESTRIPER FUNCTIONS-------------------------------#
#------------------------------------------------------------------------#

def Run(tod, weights, pix, baselines, npix, maxiter=300, Nodes=None, substring='', threads=1, submean=True, noavg=False,destripe=True, verbose=False ):
    '''
    Return Destriped maps for a given set of TOD.

    Arguments
    tod -- Time-ordered single-dish radio telescope scanning data. (Array)
    weights -- Weight for each tod sample.
    pix -- Time-ordered pixel coordinate data. (Array)
    bl  -- Length of baselines used to Destripe the tod. (Integer)
    npix -- Total number of pixels for output map.

    Keyword Arguments
    maxiter -- Maximum CGM iterations to perform

    '''

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if isinstance(Nodes, type(None)):
        Nodes = np.arange(size, dtype='int')
    
    todmean  = MPI_tools.MPI_sum(comm,tod)/MPI_tools.MPI_len(comm,tod)
    if submean:
        tod -= todmean

    #Generate Maps:
    if verbose:
        print 'PREGENERATING MAP INFORMATION'
    Data = Destriper(tod, weights, pix, baselines, npix, threads=threads)

    if verbose:
        print 'NBASELINES', Data.nBaselines
        print 'MAKING INITIAL GUESS'
    Data.GuessA0()

    if verbose:
        print 'BINNING MAP'
    Data.BinMap(Data.tod, Data.weights)
    MPISum2Root(Data.sw, Data.swroot, Data.Nodes)
    MPISum2Root(Data.w , Data.wroot, Data.Nodes)


    if rank==0:
        gd = (Data.wroot != 0)
        if noavg:
            inputMap = Data.swroot * 1.
            Data.fullmap[gd] = Data.swroot[gd]
        else:
            Data.fullmap[gd] = Data.swroot[gd]/Data.wroot[gd]
            inputMap = Data.fullmap * 1.
    if verbose:        
        print 'STARTING CGM'
    if destripe:
        CGM(Data, substring)

    if verbose:
        print 'CGM COMPLETED'


    Data.BinVarMap(Data.tod-Data.a0[Data.blpix], Data.weights) 
    Data.BinMap(Data.a0[Data.blpix], Data.weights) 
    MPISum2Root(Data.sw, Data.swroot, Data.Nodes)
    MPISum2Root(Data.w , Data.wroot, Data.Nodes)
    if rank==0:
        gd = (Data.sw != 0)
        Data.fullmap[gd] = Data.swroot[gd]/Data.wroot[gd]

        return inputMap, inputMap - Data.fullmap, Data.a0, Data.fullmap, Data.varmap
    else:
        return None

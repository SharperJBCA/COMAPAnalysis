#Holds all the function for solving Ax and b.
#Standard modules:
import numpy as np
from scipy.interpolate import interp1d
from MPIFuncs import MPISum2Root, MPIRoot2Nodes

try:
    from mpi4py import MPI
    f_found=True
    import MPI_tools
except ImportError:
    f_found=False

#Map-making modules:
#from ..Tools.Mapping import MapsClass
#from ..Tools import nBinning as Binning
#from ..Tools import fBinning

import time

def bFunc(Data):
    '''
    Returns solution for Ft Z d. 

    Arguments
    a0 -- Offsets, not used.
    tod -- input data
    bl  -- baseline length
    pix -- pixel coordinate vector (tod.size)
    cn -- estimated white-noise variance vector
    Maps -- Object holding map data
    
    '''
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
        

    Data.BinMap(Data.tod, Data.weights)


    MPISum2Root(Data.sw, Data.swroot, Data.Nodes)
    MPISum2Root(Data.w , Data.wroot, Data.Nodes)
    if rank==0:
        gd = (Data.wroot != 0)
        Data.fullmap[gd] = Data.swroot[gd]/Data.wroot[gd]


    MPIRoot2Nodes(Data.fullmap, Data.Nodes)

    #FtZd = Data.FtC(Data.tod, Data.weights) - Data.FtC(Data.fullmap[Data.pix], Data.weights)
    Data.FtZd *= 0.
    Data.FtC(Data.tod, Data.weights, Data.FtZd, Data.blpix, np.arange(Data.size)) 
    print 'i1', np.sum(Data.FtZd)
    Data.FtC(-Data.fullmap, Data.weights, Data.FtZd, Data.blpix, Data.pix)
    print 'i2', np.sum(Data.FtZd)

    
    # - Data.FtC(, Data.weights)    


    return Data.FtZd #np.reshape(FtZd, (FtZd.size, 1))


def AXFunc(x, Data, comm=None):
    '''
    Returns solution for Ft Z F a

    Arguments
    a0 -- Offsets for this CGM iteration.
    FtZFa -- This iterations guess at the conjugate vector to a0. Modified in place.

    tod -- input data
    bl  -- baseline length
    pix -- pixel coordinate vector (tod.size)
    cn -- estimated white-noise variance vector
    Maps -- Object holding map data
    

    '''
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    #Make a map of the baselines:  
    Data.BinMap(x[Data.blpix], Data.weights)
    MPISum2Root(Data.sw, Data.swroot, Data.Nodes)
    MPISum2Root(Data.w , Data.wroot, Data.Nodes)
    if rank==0:
        gd = (Data.wroot != 0)
        Data.fullmap[gd] = Data.swroot[gd]/Data.wroot[gd]

    MPIRoot2Nodes(Data.fullmap, Data.Nodes)



    #FtZa = Data.FtC(x[Data.offsetindex], Data.weights) - Data.FtC(Data.fullmap[Data.pix], Data.weights)    
    #FtZa = Data.FtC(x[Data.offsetindex]-Data.fullmap[Data.pix], Data.weights)# - Data.FtC(, Data.weights)  
    Data.FtZa *= 0.
    Data.FtC(x, Data.weights,Data.FtZa , Data.blpix, Data.blpix) 
    Data.FtC(-Data.fullmap, Data.weights,Data.FtZa , Data.blpix, Data.pix)# - Data.FtC(, Data.weights)    

    #FtZa -= MPI_tools.MPI_sum(comm,x)
    Data.FtZa -= MPI_tools.MPI_sum(comm,x)

    #Now subtract the prior information (this is a gaussian prior)
    #FtZFa[:,0] += Ft_ext(a[:,0], bl, cn*noiseRatio**2, mask)
    return Data.FtZa #np.reshape(FtZa, (FtZa.size, 1))


def FtP(m, p, bl, cn, hits, asize, mask):
    '''
    Returns stretched out map binned into baselines

    Arguments
    m -- map vector
    p -- pixel vector
    bl -- baseline length
    cn -- white-noise variances vector
    hits -- min hits (always set to 0)
    '''

    limit = 0
    x = fBinning.bin_to_baselines(m   ,
                                  p   ,
                                  int(bl)         ,
                                  mask,
                                  cn  ,
                                  asize           )
        
    return x

def Ft(x, bl, cn, mask):
    '''
    Return bin data into baselines

    x -- tod to be binned into baselines
    bl -- baseline length
    C_N -- white-noise variances vector
    '''

    #BIN TOD TO BASELINES
    n = int(np.ceil(len(x)/float(bl)))
    out = fBinning.bin_ft(x, cn, bl, n, mask)
    return out


    return out


def Ft_ext(x, bl, cn, mask):
    '''
    Return bin data into baselines

    x -- tod to be binned into baselines
    bl -- baseline length
    C_N -- white-noise variances vector
    '''

    #BIN TOD TO BASELINES
    out = fBinning.bin_ft_ext(x, cn, bl, mask)
    return out

import numpy as np
#try:
#    from mpi4py import MPI
#    f_found=True
#    from ..Tools import MPI_tools
#except ImportError:
#    f_found=False
from mpi4py import MPI

from matplotlib import pyplot
import math

#------------------------------------------------------------------------#
# Some MPI utilities

def MPI_sum(comm,x):
    # Sum values from all nodes and broadcast back the result

    size = comm.Get_size()
    rank = comm.Get_rank()

    x2 = np.array([np.sum(x)], dtype=np.result_type(x))

    #MPITypes = 
    MPIType = MPI.DOUBLE#MPITypes[np.result_type(x2)]


    # if np.result_type(x2) == float:
    #     MPIType = MPI.FLOAT
    # else:
    #     MPIType = MPI.INT


    s = np.zeros(x2.size, dtype=np.result_type(x))

    comm.Allreduce( [x2, MPIType], [s, MPIType], op=MPI.SUM)

    return np.sum(s)

def MPI_concat(comm,x):
    # Concatenate values from all nodes and broadcast back the result

    size = comm.Get_size()
    rank = comm.Get_rank()

    vals = comm.gather(x,root=0)
    
    if rank==0:
        s = np.concatenate(vals)
    else:
        s = None
            
    s = np.array(comm.bcast(s,root=0))

    return s

def MPI_len(comm,x):
    # Concatenate values from all nodes and broadcast back the result
    size = comm.Get_size()
    rank = comm.Get_rank()


    datasize = x.size
    vals = comm.gather(datasize,root=0)
    
    s=0
    if rank==0:
        s = math.fsum(vals)
        #for val in vals:
        #    s += val
        #    print s,val
    else:
        s = None
            
    s = comm.bcast(s,root=0)

    return s

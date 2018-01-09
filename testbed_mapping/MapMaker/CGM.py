import numpy as np
from MPIFuncs import MPISum2Root, MPIRoot2Nodes
from DesFuncs import bFunc,AXFunc
try:
    from mpi4py import MPI
    f_found=True
    import MPI_tools
except ImportError:
    f_found=False

cutoff = 1e-19

import time
def CGM(Data, substring=''):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()


    FtZd = bFunc(Data)

    FtZa = AXFunc(Data.a0, Data)

    #Ensure b and x0 are column vectors:
    #r = np.reshape(FtZd - FtZa, (Data.nBaselines, 1))
    #d = np.reshape(FtZd - FtZa, (Data.nBaselines, 1))
    r = Data.FtZd - Data.FtZa
    d = Data.FtZd - Data.FtZa

    #Initial Threshold:
    del0 = np.sum(r**2)#r.T.dot(r)
    if f_found:
        del0  = MPI_tools.MPI_sum(comm,del0)
    else:
        del0 = np.sum(del0)

    

    dnew = np.copy(del0)

    #errors = np.zeros((Data.maxiter, Data.nBaselines))

    lastLim = 1e32 #Initial value
    for i in range(Data.maxiter):
        # Convergence analysis
        #errors[i,:] = Data.a0
        ######################
        t0 = time.time()
        #Generate a new conjugate search vector Ad using d:
        FtZa = AXFunc(d, Data)
        
        t1 = time.time()
        # Calculate search vector:
        dTq = np.sum(d*Data.FtZa)

        if f_found:
           dTq = MPI_tools.MPI_sum(comm,dTq)
        else:
           dTq = np.sum(dTq)

        alpha = dnew/dTq
        Data.a0 +=  alpha*d

        if np.mod(i+1,50) == 0:
           FtZa = AXFunc(Data.a0, Data)
           r = Data.FtZd - Data.FtZa

        else:
           r = r - alpha*Data.FtZa

        dold = dnew*1.
        dnew = np.sum(r**2)# r.T.dot(r) 
        if f_found:
            dnew= MPI_tools.MPI_sum(comm,dnew)
        else:
            dnew = np.sum(dnew)

        beta = dnew/dold

        print 'ITERATION TIME', time.time()-t0, time.time()-t1, t1-t0
        d = r + beta*d
        if (rank == 0):
            print 'iteration: ', i, '{:.2f} % Converged'.format(1./(1.-np.log10(cutoff * del0/dnew))*100. )
        if  cutoff * del0/dnew > 1:
            ilast = i
            break


    # Convergence analysis
    #errors -= Data.a0
    #e = np.sqrt(np.sum(errors[:ilast,:]**2, axis=1))
    #FileTools.WriteH5Py('CGM_ERRORS_{}deg.hdf5'.format(substring), {'errors':e})
    ######################

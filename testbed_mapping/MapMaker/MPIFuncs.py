import numpy as np
from mpi4py import MPI

def MPISum2Root(dshare, droot, Nodes):
    # Switch on MPI 
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        droot *= 0.
        droot += dshare
        _dshare = np.copy(dshare)
    for node in Nodes:
        if node == 0:
            continue

        if rank == node:
            comm.Send([dshare, MPI.DOUBLE], dest=0, tag=node)
        if rank == 0:
            comm.Recv([dshare, MPI.DOUBLE], source=node, tag=node)
            droot += dshare
    if rank == 0:
        dshare[:] = _dshare[:]

def MPIRoot2Nodes(dshare, Nodes):
    # Switch on MPI 
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    for node in Nodes:
        if node == 0:
            continue

        if rank == 0:
            comm.Send([dshare, MPI.DOUBLE], dest=node, tag=node)
        if rank == node:
            comm.Recv([dshare, MPI.DOUBLE], source=0, tag=node)

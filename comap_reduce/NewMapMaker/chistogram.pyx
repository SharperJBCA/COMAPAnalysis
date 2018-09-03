import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

ITYPE = np.int
ctypedef np.int_t ITYPE_t


cimport cython
from cpython.array cimport array, clone


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def histogram(ITYPE_t[:] locations, DTYPE_t[:] weights, DTYPE_t[:] output):
    
    cdef int i
    
    for i in range(locations.size):
        output[locations[i]] += weights[i]

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def histogram_weights(ITYPE_t[:] locations, DTYPE_t[:] data, DTYPE_t[:] weights, DTYPE_t[:] output):
    
    cdef int i
    
    for i in range(locations.size):
        output[locations[i]] += weights[i]*data[i]

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def unwrap(ITYPE_t[:] locations, DTYPE_t[:] data, DTYPE_t[:] output):
    
    cdef int i
    
    for i in range(locations.size):
        output[i] = data[locations[i]]
        

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def dotcovar(DTYPE_t[:] clong, DTYPE_t[:] data, DTYPE_t[:] output):
    
    cdef int i, j
    
    for i in range(clong.size):
        for j in range(clong.size):
            output[i] += data[(j+i) % clong.size]
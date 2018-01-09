cdef extern from "stdlib.h":
    double drand48()
    void srand48(long int seedval)

cdef extern from "time.h":
    long int time(int)

cdef extern from "math.h":
    double M_PI
    double sin(double x)
    double cos(double x)
    double log(double x)
    double sqrt(double x)

from cython.parallel cimport prange
from cython cimport parallel
cimport cython
from libc.stdio cimport printf
cimport openmp

#import pyximport
#pyximport.install(pyimport = True)

import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

ITYPE = np.int
ctypedef np.int64_t ITYPE_t



cimport cython

def medians(double[:] x,
            double[:] a,
            int bl):

    cdef int nsamples = x.shape[0]
    cdef int nbaselines = a.shape[0]
    
    for i in range(nbaselines):
        a[i] = np.median(x[i*bl:(i+1)*bl])

@cython.boundscheck(False)
@cython.wraparound(False)
def histweight(double[:] x,
               double[:] w,
               long[:] pix,
               double[:] sw,
               double[:,:] sw_local,
               int threads):

    cdef int nsamples = x.shape[0]
    cdef int pixels = sw.shape[0]
    cdef int i
    cdef long p = 0

    cdef int threadid = -1

    # bin tod locally
    for i in prange(nsamples, nogil=True, num_threads=threads):
        threadid = openmp.omp_get_thread_num()
        if (pix[i] >= 0) and (pix[i] < pixels):
            sw_local[threadid, pix[i]] += x[i]*w[i]

    # then bin into the global map
    with nogil:
        for p in prange(pixels, num_threads=threads):
            for threadid in range(threads):
                sw[p] += sw_local[threadid, p]


@cython.boundscheck(False)
@cython.wraparound(False)
def weight(double[:] w,
           long[:] pix,
           double[:] sw,
           double[:,:] sw_local,
           int threads):

    cdef int nsamples = w.shape[0]
    cdef int pixels = sw.shape[0]
    cdef int i = 0
    
    cdef long p = 0

    cdef int threadid = -1
    
    with nogil:
        for i in prange(nsamples, num_threads=threads, schedule='static'):
            p = pix[i]
            threadid = openmp.omp_get_thread_num()
            if (p >= 0) and (p < pixels):
                sw_local[threadid, p] += w[i]

    with nogil:
        for p in prange(pixels, num_threads=threads):
            for threadid in range(threads):
                sw[p] += sw_local[threadid, p]


@cython.boundscheck(False)
@cython.wraparound(False)
def hits(long[:] pix,
         double[:] sw,
         double[:,:] sw_local,
         int threads):

    cdef int nsamples = pix.shape[0]
    cdef int pixels = sw.shape[0]
    cdef int i

    cdef long p = 0

    cdef int threadid = -1

    for i in prange(nsamples, nogil=True, num_threads=threads):
        threadid = openmp.omp_get_thread_num()
        if (pix[i] >= 0) and (pix[i] < pixels):
            sw_local[threadid, pix[i]] += 1

    with nogil:
        for p in prange(pixels, num_threads=threads):
            for threadid in range(threads):
                sw[p] += sw_local[threadid, p]


@cython.boundscheck(False)
@cython.wraparound(False)
def FtC(double[:] d,
        double[:] w,
        double[:] Ft,
        double[:,:] Ft_local,
        long[:] I1,
        long[:] I2,
        int threads):
    
    cdef int nbaselines = Ft.shape[0]
    cdef int nsamples = w.shape[0]
    cdef int i
    cdef long p = 0
    cdef int threadid = -1

    #for i in prange(nsamples, nogil=True, num_threads=threads):
    #    Ft[I1[i]] += d[I2[i]] * w[i]


    for i in prange(nsamples, nogil=True, num_threads=threads):
        threadid = openmp.omp_get_thread_num()
        Ft_local[threadid, I1[i]] += d[I2[i]] * w[i]

    with nogil:
        for p in prange(nbaselines, num_threads=threads):
            for threadid in range(threads):
                Ft[p] += Ft_local[threadid, p]

    #for p in range(nbaselines):
    #    for threadid in range(threads):
    #        Ft[p] += Ft_local[threadid, p]
                

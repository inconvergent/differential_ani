#!/usr/bin/python

from __future__ import division

import numpy as np
cimport numpy as np
cimport cython


DINT = np.int
ctypedef np.int_t DINT_t
DFLOAT = np.float
ctypedef np.float_t DFLOAT_t

@cython.wraparound(False)
@cython.boundscheck(False)
def pyx_collision_reject(l,np.ndarray[DFLOAT_t,ndim=2] sx,farl):
#def pyx_collision_reject(l,double[:,::1] sx,farl):

  cdef DINT_t vnum = l.vnum
  cdef DFLOAT_t cfarl = farl

  cdef np.ndarray[DFLOAT_t,ndim=1] X = l.X[:vnum,0]
  cdef np.ndarray[DFLOAT_t,ndim=1] Y = l.X[:vnum,1]

  near = l.get_all_near_vertices(cfarl)

  cdef np.ndarray[DINT_t,ndim=1] n_near = np.array([len(a) for a in near],DINT)
  cdef DINT_t ncols = np.max(n_near)

  cdef DINT_t k
  cdef DINT_t c
  cdef DINT_t j
  cdef DINT_t ind

  cdef DFLOAT_t x
  cdef DFLOAT_t y
  cdef DFLOAT_t dx
  cdef DFLOAT_t dy
  cdef DFLOAT_t dd
  cdef DFLOAT_t ddsquare
  cdef DFLOAT_t force
  cdef DFLOAT_t resx
  cdef DFLOAT_t resy

  for j in range(vnum):
    k = n_near[j]
    resx = 0.
    resy = 0.
    x = -X[j]
    y = -Y[j]
    for c in range(k):
      ind = <unsigned int>near[j][c]
      if ind == k:
        continue
      dx = x+X[ind]
      dy = y+Y[ind]
      ddsquare = dx*dx+dy*dy
      if ddsquare <= 0.:
        continue
      dd = ddsquare**0.5
      force = (cfarl-dd)/dd
      resx -= dx*force
      resy -= dy*force

    sx[j,0] += resx
    sx[j,1] += resy

